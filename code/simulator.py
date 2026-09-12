from datetime import timedelta

import pandas as pd

from data import (
    get_request_state,
    get_explicit_future_events,
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