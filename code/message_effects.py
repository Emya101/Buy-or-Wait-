import pandas as pd

from evidence import DATASET, load_evidence_cache


# --------------------------------------------------
# Load message metadata
# --------------------------------------------------

messages = pd.read_csv(DATASET / "messages.csv")

messages["_sent_at"] = pd.to_datetime(
    messages["sent_at"],
    errors="coerce",
    utc=True,
)

# requests.csv gives a date rather than a decision time,
# so a message sent on the request date is considered known.
messages["_sent_date"] = (
    messages["_sent_at"]
    .dt.tz_convert(None)
    .dt.normalize()
)


# --------------------------------------------------
# Small helpers
# --------------------------------------------------

def _date(value):

    if value is None or pd.isna(value):
        return None

    timestamp = pd.Timestamp(
        value
    )

    # Keep the forecasting engine consistently
    # timezone-naive. Message timestamps are UTC,
    # while request/event dates are date-only.
    if timestamp.tzinfo is not None:
        timestamp = (
            timestamp.tz_convert(
                None
            )
        )

    return timestamp.normalize()


def _effect_date(effect):
    return (
        _date(effect.get("effective_date"))
        or _date(effect.get("sent_at"))
    )


def _replacement_date(effect):
    return _date(effect.get("replacement_date"))


def _convert_amount(
    amount,
    currency,
    event_date,
    home_currency,
    normalize_amount,
):
    if amount is None:
        return None

    pseudo_event = {
        "amount": float(amount),
        "currency": currency or home_currency,
        "event_date": pd.Timestamp(event_date),
    }

    return float(
        normalize_amount(
            pseudo_event,
            home_currency,
        )
    )


def _sort_timeline(events):
    def sort_key(event):
        return (
            pd.Timestamp(event["date"]),
            0 if event["direction"] == "debit" else 1,
            event.get("category") or "",
        )

    return sorted(events, key=sort_key)


# --------------------------------------------------
# Only use messages available at decision time
# --------------------------------------------------

def get_applicable_message_effects(
    user_id,
    request_id,
    request_date,
):
    request_date = pd.Timestamp(
        request_date
    ).normalize()

    user_messages = messages[
    (
        messages["user_id"]
        == user_id
    )
    & (
        messages[
            "_sent_date"
        ].notna()
    )
    & (
        messages[
            "_sent_date"
        ]
        <= request_date
    )
    & (
        messages[
            "request_id"
        ].isna()
        |
        (
            messages[
                "request_id"
            ]
            == request_id
        )
    )
].sort_values(
    "_sent_at"
)

    cache = load_evidence_cache()
    resolved = cache.get("messages", {})

    effects = []

    for _, message in user_messages.iterrows():
        message_id = message["message_id"]
        result = resolved.get(message_id)

        if not result:
            continue

        if not result.get(
            "financially_relevant",
            False,
        ):
            continue

        for raw_effect in result.get(
            "effects",
            [],
        ):
            effect = dict(raw_effect)
            effect["message_id"] = message_id
            effect["sent_at"] = message[
                "_sent_at"
            ]
            effects.append(effect)

    return effects


# --------------------------------------------------
# Concrete CSV-event amendments
# --------------------------------------------------

def apply_targeted_event_effects(
    user_events,
    effects,
):
    result = user_events.copy()

    for effect in effects:
        effect_type = effect.get(
            "effect_type"
        )

        if effect_type not in {
            "event_date_change",
            "event_amount_change",
            "event_status_change",
        }:
            continue

        target_event_id = effect.get(
            "target_event_id"
        )

        if not target_event_id:
            continue

        mask = (
            result["event_id"]
            == target_event_id
        )

        if not mask.any():
            continue

        if effect_type == "event_date_change":
            new_date = _replacement_date(
                effect
            )

            if new_date is None:
                continue

            result.loc[
                mask,
                "event_date",
            ] = new_date

            # The message is changing when the
            # cash movement occurs, so move the
            # settlement date too.
            if "settlement_date" in result.columns:
                result.loc[
                    mask,
                    "settlement_date",
                ] = new_date

        elif effect_type == "event_amount_change":
            amount = effect.get("amount")

            if amount is None:
                continue

            result.loc[
                mask,
                "amount",
            ] = float(amount)

            currency = effect.get("currency")

            if currency:
                result.loc[
                    mask,
                    "currency",
                ] = currency

        elif effect_type == "event_status_change":
            new_status = effect.get(
                "new_status"
            )

            if not new_status:
                continue

            result.loc[
                mask,
                "status",
            ] = new_status

    return result


# --------------------------------------------------
# Locate one projected recurring stream
# --------------------------------------------------

def _matching_stream_indexes(
    projected_events,
    effect,
):
    category = effect.get("category")
    direction = effect.get("direction")
    target_event_id = effect.get(
        "target_event_id"
    )

    candidates = []

    for index, event in enumerate(
        projected_events
    ):
        if (
            category
            and event.get("category")
            != category
        ):
            continue

        if (
            direction
            and event.get("direction")
            != direction
        ):
            continue

        if (
            target_event_id
            and event.get("anchor_event_id")
            != target_event_id
        ):
            continue

        candidates.append(index)

    if target_event_id:
        return candidates

    # With no explicit target, only modify a
    # category/direction when it represents one
    # detected recurring stream. Never guess across
    # two salary streams, two rents, etc.
    anchors = {
        projected_events[index].get(
            "anchor_event_id"
        )
        for index in candidates
    }
    anchors.discard(None)

    if len(anchors) > 1:
        return None

    return candidates


# --------------------------------------------------
# Recurring changes / stops / resumes
# --------------------------------------------------

def apply_recurring_message_effects(
    projected_events,
    effects,
    request_date,
    end_date,
    home_currency,
    normalize_amount,
):
    projected = [
        dict(event)
        for event in projected_events
    ]

    request_date = _date(request_date)
    end_date = _date(end_date)

    for effect in effects:
        effect_type = effect.get(
            "effect_type"
        )

        if effect_type not in {
            "recurring_amount_change",
            "recurring_stop",
            "recurring_start",
        }:
            continue

        if not effect.get("confirmed", False):
            continue

        effective_date = (
            _effect_date(effect)
            or request_date
        )

        indexes = _matching_stream_indexes(
            projected,
            effect,
        )

        # None means multiple possible streams and
        # no target_event_id: do not guess.
        if indexes is None:
            continue

        if effect_type == "recurring_stop":
            if not indexes:
                continue

            target_indexes = set(indexes)

            projected = [
                event
                for index, event
                in enumerate(projected)
                if not (
                    index in target_indexes
                    and _date(event["date"])
                    >= effective_date
                )
            ]
            continue

        if effect_type == "recurring_amount_change":
            amount = effect.get("amount")

            if amount is None or not indexes:
                continue

            for index in indexes:
                event = projected[index]

                if (
                    _date(event["date"])
                    < effective_date
                ):
                    continue

                event["amount"] = _convert_amount(
                    amount=amount,
                    currency=effect.get("currency"),
                    event_date=event["date"],
                    home_currency=home_currency,
                    normalize_amount=normalize_amount,
                )
                event["source"] = (
                    "message_recurring_amount_change"
                )
                event["message_id"] = effect.get(
                    "message_id"
                )

            continue

        if effect_type == "recurring_start":
            amount = effect.get("amount")

            # Never invent an amount for messages
            # such as "childcare begins this month".
            if amount is None:
                continue

            if indexes:
                target_indexes = set(indexes)

                # A stream that "resumes on X" should
                # not be projected before X.
                projected = [
                    event
                    for index, event
                    in enumerate(projected)
                    if not (
                        index in target_indexes
                        and request_date
                        <= _date(event["date"])
                        < effective_date
                    )
                ]

                indexes = _matching_stream_indexes(
                    projected,
                    effect,
                )

                if indexes is None:
                    continue

                for index in indexes:
                    event = projected[index]

                    if (
                        _date(event["date"])
                        < effective_date
                    ):
                        continue

                    event["amount"] = _convert_amount(
                        amount=amount,
                        currency=effect.get("currency"),
                        event_date=event["date"],
                        home_currency=home_currency,
                        normalize_amount=normalize_amount,
                    )
                    event["source"] = (
                        "message_recurring_start"
                    )
                    event["message_id"] = effect.get(
                        "message_id"
                    )

                continue

            # If a target was named but no projected
            # stream matches it, do not fabricate one.
            if effect.get("target_event_id"):
                continue

            # No detected historical stream. Only
            # synthesize a monthly recurrence when an
            # explicit start date and amount are known.
            explicit_start = _date(
                effect.get("effective_date")
            )

            if explicit_start is None:
                continue

            current = explicit_start

            while current <= end_date:
                if current >= request_date:
                    projected.append({
                        "date": current,
                        "category": effect.get(
                            "category"
                        ) or "other",
                        "direction": effect.get(
                            "direction"
                        ) or "debit",
                        "amount": _convert_amount(
                            amount=amount,
                            currency=effect.get(
                                "currency"
                            ),
                            event_date=current,
                            home_currency=home_currency,
                            normalize_amount=normalize_amount,
                        ),
                        "source": (
                            "message_recurring_start"
                        ),
                        "anchor_event_id": (
                            "message:"
                            f"{effect.get('message_id')}"
                        ),
                        "message_id": effect.get(
                            "message_id"
                        ),
                    })

                current = (
                    current
                    + pd.DateOffset(months=1)
                )

    return _sort_timeline(projected)


# --------------------------------------------------
# Confirmed one-time message cashflows
# --------------------------------------------------

def build_one_time_message_events(
    effects,
    request_date,
    end_date,
    home_currency,
    normalize_amount,
):
    request_date = _date(request_date)
    end_date = _date(end_date)

    result = []

    for effect in effects:
        effect_type = effect.get(
            "effect_type"
        )

        if effect_type not in {
            "one_time_credit",
            "one_time_debit",
        }:
            continue

        if not effect.get("confirmed", False):
            continue

        amount = effect.get("amount")

        if amount is None:
            continue

        event_date = (
            _replacement_date(effect)
            or _date(effect.get("effective_date"))
        )

        # A confirmed amount without a known
        # settlement/payment date cannot be used.
        if event_date is None:
            continue

        if not (
            request_date
            <= event_date
            <= end_date
        ):
            continue

        direction = (
            "credit"
            if effect_type == "one_time_credit"
            else "debit"
        )

        result.append({
            "date": event_date,
            "event_id": (
                "message:"
                f"{effect.get('message_id')}"
            ),
            "category": effect.get(
                "category"
            ) or "other",
            "direction": direction,
            "amount": _convert_amount(
                amount=amount,
                currency=effect.get("currency"),
                event_date=event_date,
                home_currency=home_currency,
                normalize_amount=normalize_amount,
            ),
            "status": "confirmed",
            "source": "message_one_time",
            "message_id": effect.get(
                "message_id"
            ),
        })

    return _sort_timeline(result)
