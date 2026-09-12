from datetime import timedelta

import pandas as pd

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
