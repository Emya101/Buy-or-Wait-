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


if __name__ == "__main__":

    test_request_loading()

    request, profile, user_events = (
        get_request_state("request_26")
    )

    print("\nREQUEST")
    print(request)

    print("\nPROFILE")
    print(profile)

    print("\nFIRST 5 EVENTS")
    print(user_events.head())