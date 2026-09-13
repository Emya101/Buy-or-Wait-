from datetime import timedelta

import pandas as pd

from data import (
    get_request_state,
    get_explicit_future_events,
    include_in_cashflow,
    normalize_amount,
)

from message_effects import (
    get_applicable_message_effects,
    apply_targeted_event_effects,
    apply_recurring_message_effects,
    build_one_time_message_events,
)

from recurrence import (
    historical_events,
    project_recurring_events,
)


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
        request[
            "request_date"
        ]
    )

    end_date = (
        start_date
        + timedelta(days=90)
    )

    # --------------------------------------------------
    # Messages known at request time
    # --------------------------------------------------

    message_effects = (
        get_applicable_message_effects(
            request[
                "user_id"
            ],
            request[
                "request_id"
            ],
            start_date
        )
    )

    # --------------------------------------------------
    # Apply message changes to concrete CSV events
    # --------------------------------------------------

    user_events = (
        apply_targeted_event_effects(
            user_events,
            message_effects
        )
    )

    # Message evidence may have changed:
    # amount, date, or status.
    # Recalculate these columns before forecasting.

    user_events[
        "include_in_cashflow"
    ] = user_events.apply(
        include_in_cashflow,
        axis=1
    )

    user_events[
        "normalized_amount"
    ] = user_events.apply(
        lambda event: normalize_amount(
            event,
            profile[
                "home_currency"
            ]
        ),
        axis=1
    )

    # --------------------------------------------------
    # Historical recurrence
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Apply message changes to recurring streams
    # --------------------------------------------------

    projected_events = (
        apply_recurring_message_effects(
            projected_events,
            message_effects,
            start_date,
            end_date,
            profile[
                "home_currency"
            ],
            normalize_amount
        )
    )

    # --------------------------------------------------
    # Explicit future CSV events
    # --------------------------------------------------

    explicit_events = (
        get_explicit_future_events(
            user_events,
            start_date,
            end_date
        )
    )

    # --------------------------------------------------
    # Confirmed one-time events from messages
    # --------------------------------------------------

    message_events = (
        build_one_time_message_events(
            message_effects,
            start_date,
            end_date,
            profile[
                "home_currency"
            ],
            normalize_amount
        )
    )

    # --------------------------------------------------
    # Final timeline
    # --------------------------------------------------

    timeline = combine_timelines(
        projected_events,
        (
            explicit_events
            + message_events
        )
    )

    return (
        request,
        profile,
        timeline
    )


# --------------------------------------------------
# Simulate running balance across timeline
# --------------------------------------------------

def simulate_balance(
    starting_balance,
    minimum_balance,
    timeline
):

    balance = float(
        starting_balance
    )

    minimum_balance = float(
        minimum_balance
    )

    lowest_balance = balance
    lowest_balance_date = None

    safe = True

    simulated_timeline = []

    for event in timeline:

        amount = float(
            event["amount"]
        )

        if event["direction"] == "debit":

            balance -= amount

        elif event["direction"] == "credit":

            balance += amount

        else:

            raise ValueError(
                f"Unexpected direction: "
                f"{event['direction']}"
            )

        # Track lowest point reached.
        if balance < lowest_balance:

            lowest_balance = balance
            lowest_balance_date = (
                event["date"]
            )

        above_minimum = (
            balance >= minimum_balance
        )

        if not above_minimum:

            safe = False

        simulated_timeline.append({
            **event,
            "balance_after": balance,
            "above_minimum": above_minimum,
        })

    return {
        "safe": safe,
        "starting_balance": float(
            starting_balance
        ),
        "minimum_balance": (
            minimum_balance
        ),
        "ending_balance": balance,
        "lowest_balance": (
            lowest_balance
        ),
        "lowest_balance_date": (
            lowest_balance_date
        ),
        "timeline": (
            simulated_timeline
        ),
    }