from pathlib import Path
from datetime import timedelta

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


# --------------------------------------------------
# Load data
# --------------------------------------------------

requests = pd.read_csv(DATASET / "requests.csv")
profiles = pd.read_csv(DATASET / "financial_profiles.csv")
events = pd.read_csv(DATASET / "financial_events.csv")
exchange_rates = pd.read_csv(DATASET / "exchange_rates.csv")


# --------------------------------------------------
# Convert date columns
# --------------------------------------------------

requests["request_date"] = pd.to_datetime(
    requests["request_date"]
)

requests["desired_completion_date"] = pd.to_datetime(
    requests["desired_completion_date"]
)

events["event_date"] = pd.to_datetime(
    events["event_date"]
)

events["settlement_date"] = pd.to_datetime(
    events["settlement_date"],
    errors="coerce"
)

exchange_rates["rate_date"] = pd.to_datetime(
    exchange_rates["rate_date"]
)


# --------------------------------------------------
# Does this event affect real cashflow?
# --------------------------------------------------

def include_in_cashflow(event):

    if event["status"] in {
        "cancelled",
        "failed",
        "unrealized"
    }:
        return False

    if event["direction"] == "non_cash":
        return False

    if (
        event["status"] == "pending"
        and event["direction"] == "credit"
    ):
        return False

    return True


# --------------------------------------------------
# Convert event amount into home currency
# --------------------------------------------------

def normalize_amount(event, home_currency):

    amount = event["amount"]

    # Image-derived amounts will later be
    # inserted before this point.
    if pd.isna(amount):
        return None

    if event["currency"] == home_currency:
        return float(amount)

    rate_match = exchange_rates[
        (exchange_rates["from_currency"] == event["currency"])
        & (exchange_rates["to_currency"] == home_currency)
        & (exchange_rates["rate_date"] == event["event_date"])
    ]

    if rate_match.empty:
        raise ValueError(
            f"No exchange rate for "
            f"{event['currency']} -> {home_currency} "
            f"on {event['event_date'].date()}"
        )

    rate = rate_match.iloc[0]["rate"]

    return float(amount) * float(rate)


# --------------------------------------------------
# Load one request's financial state
# --------------------------------------------------

def get_request_state(request_id):

    request_match = requests[
        requests["request_id"] == request_id
    ]

    if request_match.empty:
        raise ValueError(
            f"Unknown request_id: {request_id}"
        )

    request = request_match.iloc[0]

    profile = profiles[
        profiles["user_id"]
        == request["user_id"]
    ].iloc[0]

    user_events = events[
        events["user_id"]
        == request["user_id"]
    ].copy()

    user_events["include_in_cashflow"] = (
        user_events.apply(
            include_in_cashflow,
            axis=1
        )
    )

    user_events["normalized_amount"] = (
        user_events.apply(
            lambda event: normalize_amount(
                event,
                profile["home_currency"]
            ),
            axis=1
        )
    )

    return request, profile, user_events


# --------------------------------------------------
# TEST 4A
# --------------------------------------------------

def test_request_loading():

    request, profile, user_events = (
        get_request_state("request_26")
    )

    # Correct relationship
    assert request["user_id"] == profile["user_id"]

    # We found events
    assert len(user_events) > 0

    # Every returned event belongs to this request's user
    assert (
        user_events["user_id"]
        == request["user_id"]
    ).all()

    # Required profile values exist
    assert pd.notna(
        profile["current_available_balance"]
    )

    assert pd.notna(
        profile["minimum_balance_to_keep"]
    )

    assert pd.notna(
        profile["home_currency"]
    )

    # Normalized usable events should have valid amounts
    usable = user_events[
        user_events["include_in_cashflow"]
        & user_events["amount"].notna()
    ]

    assert (
        usable["normalized_amount"].notna()
    ).all()

    print("✓ 4A request loading passed")


    # --------------------------------------------------
# Historical information available at request time
# --------------------------------------------------

def historical_events(user_events, request_date):

    history = user_events[
        (user_events["event_date"] < request_date)
        & (user_events["status"] == "settled")
        & (user_events["include_in_cashflow"])
        & (
            user_events["direction"].isin(
                ["debit", "credit"]
            )
        )
        & (
            user_events[
                "normalized_amount"
            ].notna()
        )
    ].copy()

    return history


# --------------------------------------------------
# Detect monthly recurrence
# --------------------------------------------------

def find_monthly_patterns(history):

    patterns = []

    history = history.copy()

    history["day_of_month"] = (
        history["event_date"].dt.day
    )

    grouped = history.groupby(
        [
            "category",
            "direction",
            "day_of_month"
        ]
    )

    for (
        category,
        direction,
        day
    ), group in grouped:

        group = group.sort_values(
            "event_date"
        )

        if len(group) < 3:
            continue

        dates = group[
            "event_date"
        ].tolist()

        consecutive_months = 0

        for i in range(
            1,
            len(dates)
        ):
            previous = dates[i - 1]
            current = dates[i]

            month_difference = (
                (current.year - previous.year) * 12
                + current.month
                - previous.month
            )

            if month_difference == 1:
                consecutive_months += 1

        if consecutive_months < 2:
            continue

        patterns.append({
            "category": category,
            "direction": direction,
            "recurrence_type": "monthly",
            "day_of_month": int(day),
            "estimated_amount": (
                group[
                    "normalized_amount"
                ].mean()
            ),
            "last_event_date": (
                group[
                    "event_date"
                ].max()
            ),
        })

    return patterns


# --------------------------------------------------
# Detect exact fixed intervals
# --------------------------------------------------

def find_fixed_interval_patterns(history):

    patterns = []

    grouped = history.groupby(
        [
            "category",
            "direction"
        ]
    )

    for (
        category,
        direction
    ), group in grouped:

        group = group.sort_values(
            "event_date"
        )

        dates = group[
            "event_date"
        ].tolist()

        # 4 events gives us 3 intervals.
        if len(dates) < 4:
            continue

        gaps = [
            (
                dates[i]
                - dates[i - 1]
            ).days
            for i in range(
                1,
                len(dates)
            )
        ]

        if len(set(gaps)) != 1:
            continue

        interval = gaps[0]

        if not 2 <= interval <= 27:
            continue

        patterns.append({
            "category": category,
            "direction": direction,
            "recurrence_type": (
                "fixed_interval"
            ),
            "interval_days": interval,
            "estimated_amount": (
                group[
                    "normalized_amount"
                ].mean()
            ),
            "last_event_date": (
                group[
                    "event_date"
                ].max()
            ),
        })

    return patterns


# --------------------------------------------------
# Project fixed-interval recurrence
# --------------------------------------------------

def project_fixed_interval(
    pattern,
    start_date,
    end_date
):

    projected = []

    next_date = (
        pattern["last_event_date"]
        + timedelta(
            days=pattern[
                "interval_days"
            ]
        )
    )

    while next_date <= end_date:

        if next_date >= start_date:

            projected.append({
                "date": next_date,
                "category": (
                    pattern["category"]
                ),
                "direction": (
                    pattern["direction"]
                ),
                "amount": (
                    pattern[
                        "estimated_amount"
                    ]
                ),
                "source": (
                    "projected_fixed_interval"
                ),
            })

        next_date += timedelta(
            days=pattern[
                "interval_days"
            ]
        )

    return projected


# --------------------------------------------------
# Project monthly recurrence
# --------------------------------------------------

def project_monthly(
    pattern,
    start_date,
    end_date
):

    projected = []

    current_month = (
        start_date.replace(day=1)
    )

    while current_month <= end_date:

        try:

            event_date = (
                current_month.replace(
                    day=pattern[
                        "day_of_month"
                    ]
                )
            )

        except ValueError:

            current_month = (
                current_month
                + pd.DateOffset(
                    months=1
                )
            )

            continue

        if (
            start_date
            <= event_date
            <= end_date
        ):

            projected.append({
                "date": event_date,
                "category": (
                    pattern[
                        "category"
                    ]
                ),
                "direction": (
                    pattern[
                        "direction"
                    ]
                ),
                "amount": (
                    pattern[
                        "estimated_amount"
                    ]
                ),
                "source": (
                    "projected_monthly"
                ),
            })

        current_month = (
            current_month
            + pd.DateOffset(
                months=1
            )
        )

    return projected


# --------------------------------------------------
# TEST 4B
# --------------------------------------------------

def test_recurrence_detection():

    request, profile, user_events = (
        get_request_state(
            "request_26"
        )
    )

    history = historical_events(
        user_events,
        request["request_date"]
    )

    # No future leakage.
    assert (
        history["event_date"].max()
        < request["request_date"]
    )

    monthly = find_monthly_patterns(
        history
    )

    intervals = (
        find_fixed_interval_patterns(
            history
        )
    )

    monthly_lookup = {
        (
            p["category"],
            p["direction"],
            p["day_of_month"]
        )
        for p in monthly
    }

    assert (
        "rent",
        "debit",
        3
    ) in monthly_lookup

    assert (
        "utilities",
        "debit",
        7
    ) in monthly_lookup

    assert (
        "salary",
        "credit",
        8
    ) in monthly_lookup

    assert (
        "salary",
        "credit",
        22
    ) in monthly_lookup

    assert (
        "streaming",
        "debit",
        10
    ) in monthly_lookup

    interval_lookup = {
        (
            p["category"],
            p["direction"]
        ): p["interval_days"]
        for p in intervals
    }

    assert (
        interval_lookup[
            ("groceries", "debit")
        ]
        == 10
    )

    assert (
        interval_lookup[
            ("dining", "debit")
        ]
        == 21
    )

    assert (
        interval_lookup[
            ("transport", "debit")
        ]
        == 21
    )

    print(
        "✓ 4B recurrence detection passed"
    )
# --------------------------------------------------
# Get explicitly supplied future events
# --------------------------------------------------

def get_explicit_future_events(
    user_events,
    request_date,
    end_date
):

    explicit_events = []

    eligible = user_events[
        (user_events["include_in_cashflow"])
        & (
            user_events["direction"].isin(
                ["debit", "credit"]
            )
        )
        & (
            user_events[
                "normalized_amount"
            ].notna()
        )
    ].copy()

    for _, event in eligible.iterrows():

        event_date = event["event_date"]
        settlement_date = event["settlement_date"]
        direction = event["direction"]
        status = event["status"]

        # ------------------------------------------
        # DEBITS
        # ------------------------------------------

        if direction == "debit":

            # Pending money already committed before
            # the request must be reserved immediately.
            if (
                status == "pending"
                and event_date < request_date
            ):
                cashflow_date = request_date

            # Historical completed debit is already
            # reflected in current_available_balance.
            elif event_date < request_date:
                continue

            else:
                cashflow_date = event_date

        # ------------------------------------------
        # CREDITS
        # ------------------------------------------

        elif direction == "credit":

            # Pending credits were already excluded
            # by include_in_cashflow().
            #
            # Confirmed money is usable only when it
            # actually settles.
            if pd.isna(settlement_date):
                continue
            cashflow_date = settlement_date

            # Already received before request.
            if cashflow_date < request_date:
                continue

        else:
            continue

        # Must fall inside our 90-day forecast.
        if not (
            request_date
            <= cashflow_date
            <= end_date
        ):
            continue

        explicit_events.append({
            "date": cashflow_date,
            "event_id": event["event_id"],
            "category": event["category"],
            "direction": direction,
            "amount": event["normalized_amount"],
            "status": status,
            "source": "explicit_future_event",
        })

    return explicit_events


# --------------------------------------------------
# TEST 4C
# --------------------------------------------------

def test_explicit_future_events():

    start = pd.Timestamp(
        "2025-08-03"
    )

    end = (
        start
        + timedelta(days=90)
    )

    test_events = pd.DataFrame([
        {
            "event_id": "test_scheduled_debit",
            "event_date": pd.Timestamp("2025-08-08"),
            "settlement_date": pd.NaT,
            "category": "utilities",
            "direction": "debit",
            "status": "scheduled",
            "normalized_amount": 100.0,
        },
        {
            "event_id": "test_pending_debit",
            "event_date": pd.Timestamp("2025-08-01"),
            "settlement_date": pd.Timestamp("2025-08-07"),
            "category": "rent",
            "direction": "debit",
            "status": "pending",
            "normalized_amount": 200.0,
        },
        {
            "event_id": "test_pending_credit",
            "event_date": pd.Timestamp("2025-08-05"),
            "settlement_date": pd.Timestamp("2025-08-06"),
            "category": "salary",
            "direction": "credit",
            "status": "pending",
            "normalized_amount": 300.0,
        },
        {
            "event_id": "test_scheduled_credit",
            "event_date": pd.Timestamp("2025-08-05"),
            "settlement_date": pd.Timestamp("2025-08-10"),
            "category": "salary",
            "direction": "credit",
            "status": "scheduled",
            "normalized_amount": 400.0,
        },
        {
            "event_id": "test_failed_debit",
            "event_date": pd.Timestamp("2025-08-06"),
            "settlement_date": pd.NaT,
            "category": "utilities",
            "direction": "debit",
            "status": "failed",
            "normalized_amount": 500.0,
        },
        {
            "event_id": "test_historical_debit",
            "event_date": pd.Timestamp("2025-08-01"),
            "settlement_date": pd.Timestamp("2025-08-01"),
            "category": "groceries",
            "direction": "debit",
            "status": "settled",
            "normalized_amount": 600.0,
        },
        {
    "event_id": "test_unsettled_credit",
    "event_date": pd.Timestamp("2025-08-07"),
    "settlement_date": pd.NaT,
    "category": "salary",
    "direction": "credit",
    "status": "scheduled",
    "normalized_amount": 350.0,
},
    ])

    test_events["include_in_cashflow"] = (
        test_events.apply(
            include_in_cashflow,
            axis=1
        )
    )

    future = get_explicit_future_events(
        test_events,
        start,
        end
    )

    by_id = {
        event["event_id"]: event
        for event in future
    }

    assert "test_scheduled_debit" in by_id
    assert "test_pending_debit" in by_id
    assert "test_scheduled_credit" in by_id

    assert "test_pending_credit" not in by_id
    assert "test_failed_debit" not in by_id
    assert "test_historical_debit" not in by_id
    assert "test_unsettled_credit" not in by_id

    assert (
        by_id["test_scheduled_debit"]["date"]
        == pd.Timestamp("2025-08-08")
    )

    assert (
        by_id["test_pending_debit"]["date"]
        == start
    )

    assert (
        by_id["test_scheduled_credit"]["date"]
        == pd.Timestamp("2025-08-10")
    )

    print(
        "✓ 4C explicit-event handling passed"
    )

def test_request_26_has_no_explicit_future_events():

    request, profile, user_events = (
        get_request_state(
            "request_26"
        )
    )

    start = request["request_date"]
    end = start + timedelta(days=90)

    future = get_explicit_future_events(
        user_events,
        start,
        end
    )

    assert len(future) == 0

    print(
        "✓ request_26 explicit-event sanity check passed"
    )

    # --------------------------------------------------
# Project all detected recurring events
# --------------------------------------------------

def project_recurring_events(
    history,
    start_date,
    end_date
):

    projected_events = []

    monthly_patterns = (
        find_monthly_patterns(
            history
        )
    )

    interval_patterns = (
        find_fixed_interval_patterns(
            history
        )
    )

    for pattern in monthly_patterns:

        projected_events.extend(
            project_monthly(
                pattern,
                start_date,
                end_date
            )
        )

    for pattern in interval_patterns:

        projected_events.extend(
            project_fixed_interval(
                pattern,
                start_date,
                end_date
            )
        )

    return projected_events


# --------------------------------------------------
# Combine projected and explicit future events
# --------------------------------------------------

def combine_timelines(
    projected_events,
    explicit_events
):

    # Explicit data overrides an inferred
    # recurrence for the same financial event.
    explicit_keys = {
        (
            pd.Timestamp(
                event["date"]
            ).normalize(),
            event["category"],
            event["direction"]
        )
        for event in explicit_events
    }

    combined = list(
        explicit_events
    )

    for event in projected_events:

        key = (
            pd.Timestamp(
                event["date"]
            ).normalize(),
            event["category"],
            event["direction"]
        )

        # Do not double-count a recurrence
        # when an explicit event exists.
        if key in explicit_keys:
            continue

        combined.append(
            event
        )

    # Conservative same-day ordering:
    # debit before credit.
    def sort_key(event):

        direction_priority = (
            0
            if event["direction"] == "debit"
            else 1
        )

        return (
            pd.Timestamp(
                event["date"]
            ),
            direction_priority,
            event["category"]
        )

    combined.sort(
        key=sort_key
    )

    return combined


# --------------------------------------------------
# Build complete 90-day financial timeline
# --------------------------------------------------

def build_financial_timeline(
    request_id
):

    request, profile, user_events = (
        get_request_state(
            request_id
        )
    )

    start_date = (
        request["request_date"]
    )

    end_date = (
        start_date
        + timedelta(days=90)
    )

    # Only information available before
    # the request is used for recurrence.
    history = historical_events(
        user_events,
        start_date
    )

    projected_events = (
        project_recurring_events(
            history,
            start_date,
            end_date
        )
    )

    explicit_events = (
        get_explicit_future_events(
            user_events,
            start_date,
            end_date
        )
    )

    timeline = combine_timelines(
        projected_events,
        explicit_events
    )

    return (
        request,
        profile,
        timeline
    )
# --------------------------------------------------
# TEST 4D — synthetic timeline combination
# --------------------------------------------------

def test_timeline_combination():

    projected = [
        {
            "date": pd.Timestamp("2025-08-03"),
            "category": "rent",
            "direction": "debit",
            "amount": 1000.0,
            "source": "projected_monthly",
        },
        {
            "date": pd.Timestamp("2025-08-07"),
            "category": "utilities",
            "direction": "debit",
            "amount": 100.0,
            "source": "projected_monthly",
        },
        {
            "date": pd.Timestamp("2025-08-08"),
            "category": "salary",
            "direction": "credit",
            "amount": 500.0,
            "source": "projected_monthly",
        },
        {
            "date": pd.Timestamp("2025-08-10"),
            "category": "streaming",
            "direction": "debit",
            "amount": 50.0,
            "source": "projected_monthly",
        },
    ]

    explicit = [
        {
            "date": pd.Timestamp("2025-08-07"),
            "event_id": "explicit_utilities",
            "category": "utilities",
            "direction": "debit",
            "amount": 120.0,
            "status": "scheduled",
            "source": "explicit_future_event",
        },
        {
            "date": pd.Timestamp("2025-08-08"),
            "event_id": "explicit_fee",
            "category": "fee",
            "direction": "debit",
            "amount": 25.0,
            "status": "scheduled",
            "source": "explicit_future_event",
        },
    ]

    timeline = combine_timelines(
        projected,
        explicit
    )

    assert len(timeline) == 5

    utilities = [
        event
        for event in timeline
        if (
            event["category"] == "utilities"
            and event["date"]
            == pd.Timestamp("2025-08-07")
        )
    ]

    assert len(utilities) == 1
    assert utilities[0]["amount"] == 120.0
    assert (
        utilities[0]["source"]
        == "explicit_future_event"
    )

    dates = [
        event["date"]
        for event in timeline
    ]

    assert dates == sorted(dates)

    august_8 = [
        event
        for event in timeline
        if event["date"]
        == pd.Timestamp("2025-08-08")
    ]

    assert august_8[0]["direction"] == "debit"
    assert august_8[1]["direction"] == "credit"

    print("✓ 4D timeline combination passed")
# --------------------------------------------------
# TEST 4D — real request timeline
# --------------------------------------------------

def test_request_26_timeline():

    request, profile, timeline = (
        build_financial_timeline(
            "request_26"
        )
    )

    start = request[
        "request_date"
    ]

    end = (
        start
        + timedelta(days=90)
    )

    assert len(timeline) > 0

    # Every event must remain inside
    # the 90-day forecast.
    assert all(
        start
        <= event["date"]
        <= end
        for event in timeline
    )

    # Timeline must already be sorted.
    dates = [
        event["date"]
        for event in timeline
    ]

    assert dates == sorted(dates)

    # request_26 begins on Aug 3.
    # We know rent recurs on day 3.
    rent_august = [
        event
        for event in timeline
        if (
            event["category"]
            == "rent"
            and event["direction"]
            == "debit"
            and event["date"]
            == pd.Timestamp(
                "2025-08-03"
            )
        )
    ]

    assert len(
        rent_august
    ) == 1

    # We detected salary on day 8.
    salary_august = [
        event
        for event in timeline
        if (
            event["category"]
            == "salary"
            and event["direction"]
            == "credit"
            and event["date"]
            == pd.Timestamp(
                "2025-08-08"
            )
        )
    ]

    assert len(
        salary_august
    ) == 1

    print(
        "✓ request_26 90-day timeline passed"
    )

if __name__ == "__main__":

    # ==================================================
    # 4A — REQUEST LOADING / NORMALIZATION
    # ==================================================

    print("\n========================================")
    print("4A — REQUEST LOADING")
    print("========================================")

    test_request_loading()

    request, profile, user_events = (
        get_request_state("request_26")
    )

    print("\nTest data:")
    print("Request ID:", request["request_id"])
    print("User ID:", request["user_id"])
    print("Home currency:", profile["home_currency"])
    print(
        "Starting balance:",
        profile["current_available_balance"]
    )
    print(
        "Minimum balance:",
        profile["minimum_balance_to_keep"]
    )
    print(
        "Events loaded:",
        len(user_events)
    )

    print("\nFirst 3 normalized events:")

    print(
        user_events[
            [
                "event_id",
                "category",
                "direction",
                "amount",
                "currency",
                "normalized_amount"
            ]
        ].head(3)
    )

    print("\n✓ 4A PASSED")


    # ==================================================
    # 4B — RECURRENCE DETECTION
    # ==================================================

    print("\n========================================")
    print("4B — RECURRENCE DETECTION")
    print("========================================")

    test_recurrence_detection()

    history = historical_events(
        user_events,
        request["request_date"]
    )

    monthly = find_monthly_patterns(
        history
    )

    intervals = find_fixed_interval_patterns(
        history
    )

    print("\nMonthly patterns detected:")

    for pattern in monthly:

        print(
            pattern["category"],
            "→ day",
            pattern["day_of_month"],
            "|",
            pattern["direction"],
            "| amount:",
            round(
                pattern["estimated_amount"],
                2
            )
        )

    print("\nFixed interval patterns detected:")

    for pattern in intervals:

        print(
            pattern["category"],
            "→ every",
            pattern["interval_days"],
            "days",
            "|",
            pattern["direction"],
            "| amount:",
            round(
                pattern["estimated_amount"],
                2
            )
        )

    print("\n✓ 4B PASSED")


    # ==================================================
    # 4C — EXPLICIT FUTURE EVENTS
    # ==================================================

    print("\n========================================")
    print("4C — EXPLICIT FUTURE EVENTS")
    print("========================================")

    test_explicit_future_events()

    print("\nSynthetic rules verified:")
    print("✓ scheduled future debit → INCLUDED")
    print("✓ pending debit → INCLUDED immediately")
    print("✓ scheduled settled credit → INCLUDED")
    print("✓ pending credit → EXCLUDED")
    print("✓ failed debit → EXCLUDED")
    print("✓ historical settled debit → EXCLUDED")
    print("✓ unsettled credit → EXCLUDED")

    print("\n✓ 4C SYNTHETIC TEST PASSED")


    # ==================================================
    # 4C — REAL DATA SANITY CHECK
    # ==================================================

    print("\n========================================")
    print("4C — REAL DATA CHECK")
    print("========================================")

    test_request_26_has_no_explicit_future_events()

    start = request["request_date"]
    end = start + timedelta(days=90)

    future = get_explicit_future_events(
        user_events,
        start,
        end
    )

    print("\nRequest:", request["request_id"])
    print("Forecast start:", start.date())
    print("Forecast end:", end.date())
    print(
        "Explicit future events found:",
        len(future)
    )

    if future:

        for event in future:

            print(
                event["date"].date(),
                event["direction"],
                event["category"],
                round(
                    event["amount"],
                    2
                )
            )

    else:

        print(
            "No explicit future events "
            "for this request."
        )

    print("\n✓ 4C REAL DATA CHECK PASSED")

    # ==================================================
    # 4D — COMPLETE 90-DAY TIMELINE
    # ==================================================

    print("\n========================================")
    print("4D — FINANCIAL TIMELINE")
    print("========================================")

    test_timeline_combination()
    test_request_26_timeline()

    request, profile, timeline = (
        build_financial_timeline(
            "request_26"
        )
    )

    print("\nRequest:", request["request_id"])

    print(
        "Forecast:",
        request["request_date"].date(),
        "→",
        (
            request["request_date"]
            + timedelta(days=90)
        ).date()
    )

    print(
        "Timeline events:",
        len(timeline)
    )

    print(
        "\nFirst 15 timeline events:"
    )

    for event in timeline[:15]:

        print(
            event["date"].date(),
            "|",
            f"{event['direction']:<6}",
            "|",
            f"{event['category']:<15}",
            "|",
            f"{event['amount']:.2f}",
            "|",
            event["source"]
        )

    print("\n✓ 4D REAL TIMELINE PASSED")


    # ==================================================
    # FINAL RESULT
    # ==================================================

    print("\n========================================")
    print("✓ ALL STAGE 4A–4D TESTS PASSED")
    print("========================================")