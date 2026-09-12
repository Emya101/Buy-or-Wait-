from datetime import timedelta
from itertools import combinations

import pandas as pd

from data import (
    get_request_state,
    request_payment_options,
)

from simulator import (
    build_financial_timeline,
    simulate_balance,
)


MONEY_TOLERANCE = 0.05


# --------------------------------------------------
# Small value helpers
# --------------------------------------------------

def pipe_values(value):

    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    }


def boolean_value(value):

    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        == "true"
    )


# --------------------------------------------------
# Add request payments into financial timeline
# --------------------------------------------------

def add_payments_to_timeline(
    timeline,
    payments
):

    combined = [
        dict(event)
        for event in timeline
    ]

    for index, payment in enumerate(
        payments,
        start=1
    ):

        combined.append({
            "date": pd.Timestamp(
                payment["date"]
            ),
            "category": "request_payment",
            "direction": "debit",
            "amount": float(
                payment["amount"]
            ),
            "source": "request_payment",
            "payment_number": index,
        })

    # Conservative ordering:
    # all debits happen before credits
    # on the same date.
    def sort_key(event):

        direction_priority = (
            0
            if event["direction"] == "debit"
            else 1
        )

        request_payment_priority = (
            0
            if event.get("source")
            == "request_payment"
            else 1
        )

        return (
            pd.Timestamp(
                event["date"]
            ),
            direction_priority,
            request_payment_priority,
            event.get(
                "category",
                ""
            ),
        )

    combined.sort(
        key=sort_key
    )

    return combined


# --------------------------------------------------
# Deterministically evaluate one payment schedule
# --------------------------------------------------

def evaluate_payments(
    request,
    profile,
    timeline,
    payments,
    expected_total=None
):

    if not payments:

        return {
            "plan_safe": False,
            "cashflow_safe": False,
            "inside_forecast": False,
            "total_matches": False,
            "completes_by_deadline": False,
            "total_paid": 0.0,
            "completion_date": None,
            "simulation": None,
        }

    payments = sorted(
        payments,
        key=lambda payment: pd.Timestamp(
            payment["date"]
        )
    )

    request_date = (
        request["request_date"]
    )

    forecast_end = (
        request_date
        + timedelta(days=90)
    )

    payment_dates = [
        pd.Timestamp(
            payment["date"]
        )
        for payment in payments
    ]

    inside_forecast = all(
        request_date
        <= payment_date
        <= forecast_end
        for payment_date
        in payment_dates
    )

    total_paid = sum(
        float(payment["amount"])
        for payment in payments
    )

    if expected_total is None:

        total_matches = True

    else:

        total_matches = (
            abs(
                total_paid
                - float(expected_total)
            )
            <= MONEY_TOLERANCE
        )

    completion_date = max(
        payment_dates
    )

    completes_by_deadline = (
        completion_date
        <= request[
            "desired_completion_date"
        ]
    )

    if not inside_forecast:

        return {
            "plan_safe": False,
            "cashflow_safe": False,
            "inside_forecast": False,
            "total_matches": total_matches,
            "completes_by_deadline":
                completes_by_deadline,
            "total_paid": total_paid,
            "completion_date":
                completion_date,
            "simulation": None,
        }

    payment_timeline = (
        add_payments_to_timeline(
            timeline,
            payments
        )
    )

    simulation = simulate_balance(
        starting_balance=(
            profile[
                "current_available_balance"
            ]
        ),
        minimum_balance=(
            profile[
                "minimum_balance_to_keep"
            ]
        ),
        timeline=payment_timeline,
    )

    cashflow_safe = (
        simulation["safe"]
    )

    plan_safe = (
        cashflow_safe
        and total_matches
        and inside_forecast
    )

    return {
        "plan_safe": plan_safe,
        "cashflow_safe": cashflow_safe,
        "inside_forecast": inside_forecast,
        "total_matches": total_matches,
        "completes_by_deadline":
            completes_by_deadline,
        "total_paid": total_paid,
        "completion_date":
            completion_date,
        "simulation": simulation,
    }


# ==================================================
# 5A — FULL PAYMENT TODAY
# ==================================================

def build_full_payment_candidate(
    request,
    profile,
    timeline,
    spending_changes=None
):

    payments = [{
        "date": request[
            "request_date"
        ],
        "amount": float(
            request[
                "requested_amount"
            ]
        ),
    }]

    result = evaluate_payments(
        request=request,
        profile=profile,
        timeline=timeline,
        payments=payments,
        expected_total=(
            request[
                "requested_amount"
            ]
        ),
    )

    accepted_methods = pipe_values(
        profile[
            "payment_methods_user_will_consider"
        ]
    )

    preference_eligible = (
        "full_payment"
        in accepted_methods
    )

    return {
        "method": "full_payment",
        "payment_option_id": None,
        "payments": payments,
        "spending_changes":
            spending_changes or [],
        "preference_eligible":
            preference_eligible,
        **result,
        "eligible": (
            preference_eligible
            and result["plan_safe"]
            and result[
                "completes_by_deadline"
            ]
        ),
    }


# ==================================================
# 5B — MAXIMUM SAFE PAYMENT TODAY
# ==================================================

def calculate_amount_safe_to_pay(
    request,
    profile,
    timeline
):

    baseline = simulate_balance(
        starting_balance=(
            profile[
                "current_available_balance"
            ]
        ),
        minimum_balance=(
            profile[
                "minimum_balance_to_keep"
            ]
        ),
        timeline=timeline,
    )

    headroom = (
        baseline["lowest_balance"]
        - baseline["minimum_balance"]
    )

    safe_amount = max(
        0.0,
        min(
            float(
                request[
                    "requested_amount"
                ]
            ),
            headroom,
        )
    )

    return round(
        safe_amount,
        2
    )


# ==================================================
# 5C — EARLIEST SAFE FULL PAYMENT
# ==================================================

def find_earliest_full_payment_date(
    request,
    profile,
    timeline
):

    request_date = (
        request[
            "request_date"
        ]
    )

    forecast_end = (
        request_date
        + timedelta(days=90)
    )

    current_date = request_date

    while (
        current_date
        <= forecast_end
    ):

        payments = [{
            "date": current_date,
            "amount": float(
                request[
                    "requested_amount"
                ]
            ),
        }]

        result = evaluate_payments(
            request=request,
            profile=profile,
            timeline=timeline,
            payments=payments,
            expected_total=(
                request[
                    "requested_amount"
                ]
            ),
        )

        # Deliberately do NOT check
        # desired_completion_date here.
        #
        # earliest_date_for_full_payment
        # measures financial capacity
        # across the full forecast.
        if result["plan_safe"]:
            return current_date

        current_date += timedelta(
            days=1
        )

    return None


# --------------------------------------------------
# Build wait candidate
# --------------------------------------------------

def build_wait_candidate(
    request,
    profile,
    timeline,
    earliest_date,
    spending_changes=None
):

    if earliest_date is None:
        return None

    if (
        earliest_date
        <= request[
            "request_date"
        ]
    ):
        return None

    payments = [{
        "date": earliest_date,
        "amount": float(
            request[
                "requested_amount"
            ]
        ),
    }]

    result = evaluate_payments(
        request=request,
        profile=profile,
        timeline=timeline,
        payments=payments,
        expected_total=(
            request[
                "requested_amount"
            ]
        ),
    )

    accepted_methods = pipe_values(
        profile[
            "payment_methods_user_will_consider"
        ]
    )

    preference_eligible = (
        "full_payment"
        in accepted_methods
    )

    return {
        "method": "wait",
        "payment_option_id": None,
        "payments": payments,
        "spending_changes":
            spending_changes or [],
        "preference_eligible":
            preference_eligible,
        **result,
        "eligible": (
            preference_eligible
            and result["plan_safe"]
            and result[
                "completes_by_deadline"
            ]
        ),
    }


# ==================================================
# 5D — INSTALLMENT OPTIONS
# ==================================================

def build_installment_schedule(
    option
):

    if (
        option["payment_method"]
        != "installments"
    ):

        raise ValueError(
            "Payment option is not "
            "an installment option."
        )

    first_date = pd.Timestamp(
        option[
            "first_payment_date"
        ]
    )

    payment_amount = float(
        option[
            "payment_amount"
        ]
    )

    payment_count = int(
        option[
            "number_of_payments"
        ]
    )

    frequency_days = int(
        option[
            "payment_frequency_days"
        ]
    )

    return [
        {
            "date": (
                first_date
                + timedelta(
                    days=(
                        index
                        * frequency_days
                    )
                )
            ),
            "amount":
                payment_amount,
        }
        for index
        in range(payment_count)
    ]


def evaluate_installment_options(
    request,
    profile,
    timeline,
    spending_changes=None
):

    options = request_payment_options[
        (
            request_payment_options[
                "request_id"
            ]
            == request[
                "request_id"
            ]
        )
        & (
            request_payment_options[
                "payment_method"
            ]
            == "installments"
        )
    ]

    accepted_methods = pipe_values(
        profile[
            "payment_methods_user_will_consider"
        ]
    )

    accepts_installments = (
        "installments"
        in accepted_methods
    )

    max_months = profile[
        "max_installment_months"
    ]

    candidates = []

    for _, option in options.iterrows():

        schedule = (
            build_installment_schedule(
                option
            )
        )

        result = evaluate_payments(
            request=request,
            profile=profile,
            timeline=timeline,
            payments=schedule,
            expected_total=(
                option[
                    "total_payable_amount"
                ]
            ),
        )

        if pd.isna(max_months):

            term_eligible = True

        else:

            # Payment frequencies are monthly
            # style: 28, 30, or 31 days.
            term_eligible = (
                int(
                    option[
                        "number_of_payments"
                    ]
                )
                <= int(max_months)
            )

        preference_eligible = (
            accepts_installments
        )

        candidate = {
            "method": "installments",
            "payment_option_id": (
                option[
                    "payment_option_id"
                ]
            ),
            "payments": schedule,
            "spending_changes":
                spending_changes or [],
            "financing_fee": float(
                option[
                    "financing_fee"
                ]
            ),
            "total_payable_amount":
                float(
                    option[
                        "total_payable_amount"
                    ]
                ),
            "term_eligible":
                term_eligible,
            "preference_eligible":
                preference_eligible,
            **result,
        }

        candidate["eligible"] = (
            preference_eligible
            and term_eligible
            and result["plan_safe"]
            and result[
                "completes_by_deadline"
            ]
        )

        candidates.append(
            candidate
        )

    return candidates


# ==================================================
# 5E — PARTIAL PAYMENT
# ==================================================

def build_partial_payment_candidate(
    request,
    profile,
    timeline,
    amount_safe_to_pay,
    earliest_full_payment_date,
    spending_changes=None
):

    accepted_methods = pipe_values(
        profile[
            "payment_methods_user_will_consider"
        ]
    )

    allows_partial = boolean_value(
        request[
            "allows_partial_payment"
        ]
    )

    requested_amount = float(
        request[
            "requested_amount"
        ]
    )

    preference_eligible = (
        "partial_payment"
        in accepted_methods
    )

    if not (
        allows_partial
        and preference_eligible
        and 0
        < amount_safe_to_pay
        < requested_amount
        and earliest_full_payment_date
        is not None
        and earliest_full_payment_date
        <= request[
            "desired_completion_date"
        ]
    ):
        return None

    remaining = round(
        requested_amount
        - amount_safe_to_pay,
        2
    )

    payments = [
        {
            "date": request[
                "request_date"
            ],
            "amount":
                amount_safe_to_pay,
        },
        {
            "date":
                earliest_full_payment_date,
            "amount":
                remaining,
        },
    ]

    result = evaluate_payments(
        request=request,
        profile=profile,
        timeline=timeline,
        payments=payments,
        expected_total=
            requested_amount,
    )

    return {
        "method": "partial_payment",
        "payment_option_id": None,
        "payments": payments,
        "spending_changes":
            spending_changes or [],
        "preference_eligible":
            preference_eligible,
        **result,
        "eligible": (
            result["plan_safe"]
            and result[
                "completes_by_deadline"
            ]
        ),
    }


# ==================================================
# 5F — FLEXIBLE SPENDING CHANGES
# ==================================================

def get_flexible_recurring_actions(
    request,
    profile,
    user_events,
    timeline
):

    willing_to_reduce = pipe_values(
        profile[
            "expense_categories_user_is_willing_to_reduce"
        ]
    )

    willing_to_stop = pipe_values(
        profile[
            "expense_categories_user_is_willing_to_stop"
        ]
    )

    # We may only change expenses that
    # Stage 4 actually classified as recurring.
    recurring_categories = {
        event["category"]
        for event in timeline
        if (
            event["direction"]
            == "debit"
            and str(
                event["source"]
            ).startswith(
                "projected_"
            )
        )
    }

    history = user_events[
        (
            user_events[
                "event_date"
            ]
            < request[
                "request_date"
            ]
        )
        & (
            user_events["status"]
            == "settled"
        )
        & (
            user_events[
                "direction"
            ]
            == "debit"
        )
    ].copy()

    actions = []

    for category in sorted(
        recurring_categories
    ):

        category_history = history[
            history["category"]
            == category
        ]

        if category_history.empty:
            continue

        latest = (
            category_history
            .sort_values(
                "event_date"
            )
            .iloc[-1]
        )

        flexibility = (
            str(
                latest[
                    "flexibility"
                ]
            )
            .strip()
            .lower()
        )

        event_id = latest[
            "event_id"
        ]

        # ------------------------------------------
        # STOP
        # ------------------------------------------

        can_stop = (
            category
            in willing_to_stop
            and flexibility
            in {
                "stoppable",
                "reducible_or_stoppable",
            }
        )

        if can_stop:

            actions.append({
                "action": "stop",
                "event_id":
                    event_id,
                "category":
                    category,
                "new_amount":
                    None,
            })

        # ------------------------------------------
        # REDUCE
        # ------------------------------------------

        can_reduce = (
            category
            in willing_to_reduce
            and flexibility
            in {
                "reducible",
                "reducible_or_stoppable",
            }
            and pd.notna(
                latest[
                    "minimum_allowed_amount"
                ]
            )
        )

        if can_reduce:

            source_amount = latest[
                "amount"
            ]

            normalized_amount = latest[
                "normalized_amount"
            ]

            minimum_raw = float(
                latest[
                    "minimum_allowed_amount"
                ]
            )

            # Convert minimum_allowed_amount
            # into home currency using the same
            # conversion factor as the event.
            if (
                pd.notna(source_amount)
                and float(source_amount)
                != 0
            ):

                conversion_factor = (
                    float(
                        normalized_amount
                    )
                    / float(
                        source_amount
                    )
                )

            else:

                conversion_factor = 1.0

            minimum_home = (
                minimum_raw
                * conversion_factor
            )

            projected_amounts = [
                float(
                    event["amount"]
                )
                for event in timeline
                if (
                    event[
                        "direction"
                    ]
                    == "debit"
                    and event[
                        "category"
                    ]
                    == category
                    and str(
                        event["source"]
                    ).startswith(
                        "projected_"
                    )
                )
            ]

            # It only counts as a reduction
            # if the forecast is currently above
            # the allowed minimum.
            if (
                projected_amounts
                and max(
                    projected_amounts
                )
                > (
                    minimum_home
                    + MONEY_TOLERANCE
                )
            ):

                actions.append({
                    "action":
                        "reduce_to",
                    "event_id":
                        event_id,
                    "category":
                        category,
                    "new_amount":
                        round(
                            minimum_home,
                            2
                        ),
                })

    return actions


# --------------------------------------------------
# Apply one set of optional changes
# --------------------------------------------------

def apply_spending_changes(
    timeline,
    changes
):

    actions_by_category = {
        change["category"]: change
        for change in changes
    }

    adjusted = []

    for event in timeline:

        change = (
            actions_by_category.get(
                event["category"]
            )
        )

        affects_recurring_debit = (
            change is not None
            and event["direction"]
            == "debit"
            and (
                str(
                    event["source"]
                ).startswith(
                    "projected_"
                )
                or event[
                    "source"
                ]
                == "explicit_future_event"
            )
        )

        if not affects_recurring_debit:

            adjusted.append(
                dict(event)
            )

            continue

        if (
            change["action"]
            == "stop"
        ):

            # Future recurring debit disappears.
            continue

        if (
            change["action"]
            == "reduce_to"
        ):

            changed_event = dict(
                event
            )

            changed_event["amount"] = (
                float(
                    change[
                        "new_amount"
                    ]
                )
            )

            changed_event["source"] = (
                "spending_change"
            )

            adjusted.append(
                changed_event
            )

            continue

        raise ValueError(
            "Unknown spending change: "
            f"{change['action']}"
        )

    return adjusted


# --------------------------------------------------
# Generate up to three compatible changes
# --------------------------------------------------

def generate_spending_change_sets(
    actions,
    max_changes=3
):

    change_sets = []

    maximum = min(
        max_changes,
        len(actions)
    )

    for size in range(
        1,
        maximum + 1
    ):

        for combo in combinations(
            actions,
            size
        ):

            event_ids = [
                change["event_id"]
                for change in combo
            ]

            categories = [
                change["category"]
                for change in combo
            ]

            # Cannot stop and reduce
            # the same event/category.
            if (
                len(set(event_ids))
                != len(event_ids)
            ):
                continue

            if (
                len(set(categories))
                != len(categories)
            ):
                continue

            change_sets.append(
                list(combo)
            )

    return change_sets


# --------------------------------------------------
# Earliest safe full payment on an arbitrary timeline
# --------------------------------------------------

def earliest_safe_date_for_timeline(
    request,
    profile,
    timeline,
    end_date
):

    current_date = (
        request[
            "request_date"
        ]
    )

    while current_date <= end_date:

        result = evaluate_payments(
            request=request,
            profile=profile,
            timeline=timeline,
            payments=[{
                "date":
                    current_date,
                "amount": float(
                    request[
                        "requested_amount"
                    ]
                ),
            }],
            expected_total=(
                request[
                    "requested_amount"
                ]
            ),
        )

        if result["plan_safe"]:
            return current_date

        current_date += timedelta(
            days=1
        )

    return None


# --------------------------------------------------
# Test valid payment methods after spending changes
# --------------------------------------------------

def evaluate_spending_change_plans(
    request,
    profile,
    user_events,
    timeline,
    amount_safe_to_pay,
    baseline_earliest_date
):

    actions = (
        get_flexible_recurring_actions(
            request,
            profile,
            user_events,
            timeline
        )
    )

    change_sets = (
        generate_spending_change_sets(
            actions,
            max_changes=3
        )
    )

    valid_candidates = []

    for changes in change_sets:

        changed_timeline = (
            apply_spending_changes(
                timeline,
                changes
            )
        )

        # ------------------------------------------
        # FULL PAYMENT TODAY
        # ------------------------------------------

        full = (
            build_full_payment_candidate(
                request,
                profile,
                changed_timeline,
                spending_changes=
                    changes,
            )
        )

        if full["eligible"]:

            valid_candidates.append(
                full
            )

        # ------------------------------------------
        # INSTALLMENTS
        # ------------------------------------------

        installments = (
            evaluate_installment_options(
                request,
                profile,
                changed_timeline,
                spending_changes=
                    changes,
            )
        )

        valid_candidates.extend(
            candidate
            for candidate
            in installments
            if candidate[
                "eligible"
            ]
        )

        # ------------------------------------------
        # PARTIAL PAYMENT
        # ------------------------------------------

        partial = (
            build_partial_payment_candidate(
                request,
                profile,
                changed_timeline,
                amount_safe_to_pay,
                baseline_earliest_date,
                spending_changes=
                    changes,
            )
        )

        if (
            partial is not None
            and partial["eligible"]
        ):

            valid_candidates.append(
                partial
            )

        # ------------------------------------------
        # WAIT + CHANGES
        # ------------------------------------------

        changed_earliest = (
            earliest_safe_date_for_timeline(
                request,
                profile,
                changed_timeline,
                min(
                    request[
                        "desired_completion_date"
                    ],
                    request[
                        "request_date"
                    ]
                    + timedelta(days=90)
                )
            )
        )

        wait = build_wait_candidate(
            request,
            profile,
            changed_timeline,
            changed_earliest,
            spending_changes=
                changes,
        )

        if (
            wait is not None
            and wait["eligible"]
        ):

            valid_candidates.append(
                wait
            )

    return valid_candidates


# ==================================================
# COMPLETE STAGE 5 ANALYSIS
# ==================================================

def evaluate_request_payment_plans(
    request_id
):

    request, profile, timeline = (
        build_financial_timeline(
            request_id
        )
    )

    _, _, user_events = (
        get_request_state(
            request_id
        )
    )

    # ----------------------------------------------
    # Baseline measures — NO spending changes
    # ----------------------------------------------

    amount_safe = (
        calculate_amount_safe_to_pay(
            request,
            profile,
            timeline
        )
    )

    earliest_full_date = (
        find_earliest_full_payment_date(
            request,
            profile,
            timeline
        )
    )

    # ----------------------------------------------
    # Normal candidate plans
    # ----------------------------------------------

    candidates = []

    full = build_full_payment_candidate(
        request,
        profile,
        timeline
    )

    candidates.append(
        full
    )

    wait = build_wait_candidate(
        request,
        profile,
        timeline,
        earliest_full_date
    )

    if wait is not None:

        candidates.append(
            wait
        )

    installment_candidates = (
        evaluate_installment_options(
            request,
            profile,
            timeline
        )
    )

    candidates.extend(
        installment_candidates
    )

    partial = (
        build_partial_payment_candidate(
            request,
            profile,
            timeline,
            amount_safe,
            earliest_full_date,
        )
    )

    if partial is not None:

        candidates.append(
            partial
        )

    valid_without_changes = [
        candidate
        for candidate in candidates
        if candidate[
            "eligible"
        ]
    ]

    # Ranking rules prefer plans without
    # optional spending changes.
    #
    # Therefore we only search spending changes
    # when no normal eligible plan works.
    changed_candidates = []

    if not valid_without_changes:

        changed_candidates = (
            evaluate_spending_change_plans(
                request,
                profile,
                user_events,
                timeline,
                amount_safe,
                earliest_full_date,
            )
        )

    return {
        "request": request,
        "profile": profile,
        "timeline": timeline,
        "amount_safe_to_pay":
            amount_safe,
        "earliest_date_for_full_payment":
            earliest_full_date,
        "candidates":
            candidates,
        "valid_without_changes":
            valid_without_changes,
        "changed_candidates":
            changed_candidates,
    }