from pathlib import Path

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
request_payment_options = pd.read_csv(
    DATASET / "request_payment_options.csv"
)

request_payment_options[
    "first_payment_date"
] = pd.to_datetime(
    request_payment_options[
        "first_payment_date"
    ]
)

requests = pd.read_csv(
    DATASET / "requests.csv"
)

profiles = pd.read_csv(
    DATASET / "financial_profiles.csv"
)

events = pd.read_csv(
    DATASET / "financial_events.csv"
)

exchange_rates = pd.read_csv(
    DATASET / "exchange_rates.csv"
)

request_payment_options = pd.read_csv(
    DATASET / "request_payment_options.csv"
)

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