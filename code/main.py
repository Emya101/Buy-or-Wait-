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
# Run Stage 4A + 4B
# --------------------------------------------------

if __name__ == "__main__":

    test_request_loading()
    test_recurrence_detection()

    request, profile, user_events = (
        get_request_state("request_26")
    )

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

    print("\nMONTHLY PATTERNS")

    for pattern in monthly:
        print(pattern)

    print("\nFIXED INTERVAL PATTERNS")

    for pattern in intervals:
        print(pattern)