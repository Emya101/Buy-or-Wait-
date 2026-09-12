import re

import pandas as pd

from payment_plans import evaluate_request_payment_plans


# --------------------------------------------------
# Formatting helpers
# --------------------------------------------------

def format_money(amount):

    value = round(float(amount), 2)

    if abs(value - round(value)) < 0.001:
        return str(int(round(value)))

    return f"{value:.2f}"


def format_date(value):

    if value is None or pd.isna(value):
        return ""

    return pd.Timestamp(value).strftime(
        "%Y-%m-%d"
    )


def format_payment_plan(payments):

    if not payments:
        return "none"

    payments = sorted(
        payments,
        key=lambda payment: pd.Timestamp(
            payment["date"]
        ),
    )

    return "|".join(
        (
            f"{format_date(payment['date'])}:"
            f"{format_money(payment['amount'])}"
        )
        for payment in payments
    )


def format_spending_changes(changes):

    if not changes:
        return "none"

    formatted = []

    for change in changes:

        if change["action"] == "stop":

            formatted.append(
                f"stop:{change['event_id']}"
            )

        elif change["action"] == "reduce_to":

            formatted.append(
                (
                    f"reduce_to:"
                    f"{change['event_id']}:"
                    f"{format_money(change['new_amount'])}"
                )
            )

        else:

            raise ValueError(
                "Unknown spending change: "
                f"{change['action']}"
            )

    return "|".join(formatted)


# --------------------------------------------------
# payment_option_id tie-breaker
# --------------------------------------------------

def payment_option_rank(
    payment_option_id
):

    if payment_option_id is None:
        return float("inf")

    match = re.search(
        r"(\d+)$",
        str(payment_option_id),
    )

    if match is None:
        return float("inf")

    return int(
        match.group(1)
    )


# --------------------------------------------------
# Official candidate ranking
# --------------------------------------------------

def candidate_rank_key(candidate):

    payments = candidate.get(
        "payments",
        [],
    )

    spending_changes = candidate.get(
        "spending_changes",
        [],
    )

    start_date = min(
        (
            pd.Timestamp(
                payment["date"]
            )
            for payment in payments
        ),
        default=pd.Timestamp.max,
    )

    return (
        # 1. Complete by deadline.
        0
        if candidate.get(
            "completes_by_deadline",
            False,
        )
        else 1,

        # 2. Require no spending changes.
        0
        if not spending_changes
        else 1,

        # 3. Minimize total paid.
        float(
            candidate.get(
                "total_paid",
                float("inf"),
            )
        ),

        # 4. Start earlier.
        start_date,

        # 5. Fewer payments.
        len(payments),

        # 6. Lowest payment option ID.
        payment_option_rank(
            candidate.get(
                "payment_option_id"
            )
        ),
    )


def select_best_candidate(analysis):

    normal_candidates = (
        analysis[
            "valid_without_changes"
        ]
    )

    if normal_candidates:

        return min(
            normal_candidates,
            key=candidate_rank_key,
        )

    changed_candidates = (
        analysis[
            "changed_candidates"
        ]
    )

    if changed_candidates:

        return min(
            changed_candidates,
            key=candidate_rank_key,
        )

    return None


# --------------------------------------------------
# Convert winning candidate into affordability status
# --------------------------------------------------

def determine_affordability_status(
    candidate
):

    if candidate is None:
        return "not_affordable"

    method = candidate["method"]

    spending_changes = (
        candidate.get(
            "spending_changes",
            [],
        )
    )

    if (
        method == "full_payment"
        and not spending_changes
    ):
        return "affordable_now"

    if (
        method == "wait"
        and not spending_changes
    ):
        return "affordable_later"

    return "affordable_with_plan"


# --------------------------------------------------
# Spending-change wording
# --------------------------------------------------

def describe_spending_changes(
    changes,
    currency
):

    if not changes:
        return ""

    descriptions = []

    for change in changes:

        category = (
            change[
                "category"
            ]
            .replace("_", " ")
        )

        if change["action"] == "stop":

            descriptions.append(
                f"stop the recurring "
                f"{category} expense"
            )

        elif (
            change["action"]
            == "reduce_to"
        ):

            descriptions.append(
                (
                    f"reduce the recurring "
                    f"{category} expense to "
                    f"{currency} "
                    f"{format_money(change['new_amount'])}"
                )
            )

    if not descriptions:
        return ""

    if len(descriptions) == 1:
        return descriptions[0]

    return (
        ", ".join(
            descriptions[:-1]
        )
        + " and "
        + descriptions[-1]
    )


# --------------------------------------------------
# Build short recommendation explanation
# --------------------------------------------------

def build_explanation(
    analysis,
    candidate
):

    request = analysis["request"]
    profile = analysis["profile"]

    currency = profile[
        "home_currency"
    ]

    requested_amount = (
        format_money(
            request[
                "requested_amount"
            ]
        )
    )

    minimum_balance = (
        format_money(
            profile[
                "minimum_balance_to_keep"
            ]
        )
    )

    deadline = format_date(
        request[
            "desired_completion_date"
        ]
    )

    # ----------------------------------------------
    # No safe eligible recommendation
    # ----------------------------------------------

    if candidate is None:

        earliest = analysis[
            "earliest_date_for_full_payment"
        ]

        financially_safe_candidates = [
            plan
            for plan
            in analysis["candidates"]
            if (
                plan.get(
                    "plan_safe",
                    False,
                )
                and plan.get(
                    "completes_by_deadline",
                    False,
                )
            )
        ]

        if financially_safe_candidates:

            return (
                "No safe payment method matching "
                "the user's accepted payment "
                f"preferences completes the request "
                f"by {deadline}."
            )

        if earliest is None:

            return (
                f"Do not proceed with the "
                f"{currency} {requested_amount} "
                "request. The full amount does not "
                "become safe within the 90-day "
                f"forecast while protecting the "
                f"{currency} {minimum_balance} "
                "minimum balance."
            )

        if (
            earliest
            > request[
                "desired_completion_date"
            ]
        ):

            return (
                f"Do not proceed by {deadline}. "
                f"Full payment first becomes safe "
                f"on {format_date(earliest)}, "
                "after the requested completion "
                "date."
            )

        return (
            f"Do not proceed with the "
            f"{currency} {requested_amount} "
            "request. None of the eligible payment "
            "options keeps the required minimum "
            "balance protected."
        )

    # ----------------------------------------------
    # Safe recommendation
    # ----------------------------------------------

    method = candidate["method"]

    changes = candidate.get(
        "spending_changes",
        [],
    )

    change_text = (
        describe_spending_changes(
            changes,
            currency,
        )
    )

    simulation = candidate.get(
        "simulation"
    )

    lowest_balance = None

    if simulation is not None:

        lowest_balance = (
            simulation[
                "lowest_balance"
            ]
        )

    safety_text = ""

    if lowest_balance is not None:

        safety_text = (
            f" This leaves at least "
            f"{currency} "
            f"{format_money(lowest_balance)} "
            "available during the 90-day forecast."
        )

    prefix = ""

    if change_text:

        prefix = (
            change_text[0].upper()
            + change_text[1:]
            + ", then "
        )

    # ----------------------------------------------
    # Full payment
    # ----------------------------------------------

    if method == "full_payment":

        if prefix:

            return (
                f"{prefix}pay "
                f"{currency} "
                f"{requested_amount} today."
                f"{safety_text}"
            )

        return (
            f"Pay {currency} "
            f"{requested_amount} today."
            f"{safety_text}"
        )

    # ----------------------------------------------
    # Wait
    # ----------------------------------------------

    if method == "wait":

        payment = (
            candidate[
                "payments"
            ][0]
        )

        payment_date = (
            format_date(
                payment["date"]
            )
        )

        return (
            f"Wait until {payment_date}, "
            f"then pay {currency} "
            f"{requested_amount} in full. "
            "Paying earlier would put the "
            f"{currency} {minimum_balance} "
            "minimum balance at risk."
        )

    # ----------------------------------------------
    # Partial payment
    # ----------------------------------------------

    if method == "partial_payment":

        first = (
            candidate[
                "payments"
            ][0]
        )

        second = (
            candidate[
                "payments"
            ][1]
        )

        return (
            f"{prefix}"
            f"pay {currency} "
            f"{format_money(first['amount'])} "
            "today and the remaining "
            f"{currency} "
            f"{format_money(second['amount'])} "
            f"on {format_date(second['date'])}. "
            "This completes the full request "
            "while maintaining the required "
            "minimum balance."
        )

    # ----------------------------------------------
    # Installments
    # ----------------------------------------------

    if method == "installments":

        payments = candidate[
            "payments"
        ]

        return (
            f"{prefix}"
            f"use {len(payments)} installments "
            f"starting "
            f"{format_date(payments[0]['date'])}. "
            f"Total payable is {currency} "
            f"{format_money(candidate['total_paid'])}."
            f"{safety_text}"
        )

    raise ValueError(
        f"Unknown payment method: {method}"
    )


# --------------------------------------------------
# Final Stage 6 recommendation
# --------------------------------------------------

def build_recommendation(
    request_id
):

    analysis = (
        evaluate_request_payment_plans(
            request_id
        )
    )

    request = analysis[
        "request"
    ]

    winner = (
        select_best_candidate(
            analysis
        )
    )

    affordability_status = (
        determine_affordability_status(
            winner
        )
    )

    if winner is None:

        recommended_method = (
            "not_recommended"
        )

        payment_plan = "none"

        spending_changes = "none"

    else:

        recommended_method = (
            winner[
                "method"
            ]
        )

        payment_plan = (
            format_payment_plan(
                winner[
                    "payments"
                ]
            )
        )

        spending_changes = (
            format_spending_changes(
                winner.get(
                    "spending_changes",
                    [],
                )
            )
        )

    earliest = analysis[
        "earliest_date_for_full_payment"
    ]

    return {
        "request_id":
            request[
                "request_id"
            ],

        "amount_safe_to_pay":
            analysis[
                "amount_safe_to_pay"
            ],

        "affordability_status":
            affordability_status,

        "recommended_payment_method":
            recommended_method,

        "payment_plan":
            payment_plan,

        "earliest_date_for_full_payment":
            format_date(
                earliest
            ),

        "spending_changes_needed":
            spending_changes,

        "decision_explanation":
            build_explanation(
                analysis,
                winner,
            ),
    }