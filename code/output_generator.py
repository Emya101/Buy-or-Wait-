from pathlib import Path

import pandas as pd

from data import (
    ROOT,
    requests,
    events,
    request_payment_options,
)

from decision import (
    build_recommendation,
    format_payment_plan,
)

from payment_plans import (
    build_installment_schedule,
)


OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


ALLOWED_STATUSES = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}


ALLOWED_METHODS = {
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
}


MONEY_TOLERANCE = 0.05


# --------------------------------------------------
# Parse payment_plan back into structured payments
# --------------------------------------------------

def parse_payment_plan(payment_plan):

    if payment_plan == "none":
        return []

    payments = []

    for item in payment_plan.split("|"):

        try:

            date_text, amount_text = (
                item.split(":", 1)
            )

            date = pd.Timestamp(
                date_text
            )

            amount = float(
                amount_text
            )

        except Exception as error:

            raise ValueError(
                f"Invalid payment plan item: {item}"
            ) from error

        if amount <= 0:

            raise ValueError(
                "Payment amounts must be positive."
            )

        payments.append({
            "date": date,
            "amount": amount,
        })

    return payments


# --------------------------------------------------
# Validate spending_changes_needed syntax
# --------------------------------------------------

def validate_spending_changes(
    spending_changes
):

    if spending_changes == "none":
        return

    changes = (
        spending_changes.split("|")
    )

    if len(changes) > 3:

        raise ValueError(
            "More than three spending "
            "changes were recommended."
        )

    used_event_ids = set()

    for change in changes:

        parts = change.split(":")

        action = parts[0]

        if action == "stop":

            if len(parts) != 2:

                raise ValueError(
                    f"Invalid stop change: {change}"
                )

            event_id = parts[1]

        elif action == "reduce_to":

            if len(parts) != 3:

                raise ValueError(
                    f"Invalid reduce change: {change}"
                )

            event_id = parts[1]

            new_amount = float(
                parts[2]
            )

            if new_amount < 0:

                raise ValueError(
                    "Reduced amount cannot "
                    "be negative."
                )

        else:

            raise ValueError(
                f"Unknown spending change: {change}"
            )

        if event_id in used_event_ids:

            raise ValueError(
                f"Event {event_id} was changed "
                "more than once."
            )

        used_event_ids.add(
            event_id
        )

        event_match = events[
            events["event_id"]
            == event_id
        ]

        if event_match.empty:

            raise ValueError(
                f"Unknown spending-change "
                f"event: {event_id}"
            )

        event = event_match.iloc[0]

        if event["direction"] != "debit":

            raise ValueError(
                f"{event_id} is not a debit."
            )

        flexibility = (
            str(
                event["flexibility"]
            )
            .strip()
            .lower()
        )

        if (
            action == "stop"
            and flexibility
            not in {
                "stoppable",
                "reducible_or_stoppable",
            }
        ):

            raise ValueError(
                f"{event_id} cannot be stopped."
            )

        if (
            action == "reduce_to"
            and flexibility
            not in {
                "reducible",
                "reducible_or_stoppable",
            }
        ):

            raise ValueError(
                f"{event_id} cannot be reduced."
            )


# --------------------------------------------------
# Validate one recommendation
# --------------------------------------------------

def validate_recommendation(
    recommendation,
    request
):

    missing_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column not in recommendation
    ]

    if missing_columns:

        raise ValueError(
            "Missing output columns: "
            f"{missing_columns}"
        )

    request_id = recommendation[
        "request_id"
    ]

    if (
        request_id
        != request["request_id"]
    ):

        raise ValueError(
            f"{request_id}: request ID mismatch"
        )

    requested_amount = float(
        request["requested_amount"]
    )

    safe_amount = float(
        recommendation[
            "amount_safe_to_pay"
        ]
    )

    # ------------------------------------------
    # amount_safe_to_pay
    # ------------------------------------------

    if not (
        0
        <= safe_amount
        <= requested_amount
        + MONEY_TOLERANCE
    ):

        raise ValueError(
            f"{request_id}: invalid "
            "amount_safe_to_pay"
        )

    # ------------------------------------------
    # Allowed output values
    # ------------------------------------------

    status = recommendation[
        "affordability_status"
    ]

    method = recommendation[
        "recommended_payment_method"
    ]

    if status not in ALLOWED_STATUSES:

        raise ValueError(
            f"{request_id}: invalid status "
            f"{status}"
        )

    if method not in ALLOWED_METHODS:

        raise ValueError(
            f"{request_id}: invalid method "
            f"{method}"
        )

    # ------------------------------------------
    # Earliest full-payment date
    # ------------------------------------------

    earliest_text = recommendation[
        "earliest_date_for_full_payment"
    ]

    if earliest_text:

        earliest = pd.Timestamp(
            earliest_text
        )

        forecast_end = (
            request["request_date"]
            + pd.Timedelta(days=90)
        )

        if not (
            request["request_date"]
            <= earliest
            <= forecast_end
        ):

            raise ValueError(
                f"{request_id}: earliest full "
                "payment date outside forecast"
            )

    else:

        earliest = None

    # ------------------------------------------
    # Parse payment plan
    # ------------------------------------------

    payments = parse_payment_plan(
        recommendation[
            "payment_plan"
        ]
    )

    has_spending_changes = (
        recommendation[
            "spending_changes_needed"
        ]
        != "none"
    )

    payment_dates = [
        payment["date"]
        for payment in payments
    ]

    if (
        payment_dates
        != sorted(payment_dates)
    ):

        raise ValueError(
            f"{request_id}: payment plan "
            "is not chronological"
        )

    total_paid = sum(
        payment["amount"]
        for payment in payments
    )

    # ------------------------------------------
    # Status/method relationships
    # ------------------------------------------

    if status == "affordable_now":

        if method != "full_payment":

            raise ValueError(
                f"{request_id}: affordable_now "
                "must use full_payment"
            )

        if earliest != request[
            "request_date"
        ]:

            raise ValueError(
                f"{request_id}: affordable_now "
                "must have request_date as "
                "earliest full-payment date"
            )

    if status == "affordable_later":

        if method != "wait":

            raise ValueError(
                f"{request_id}: affordable_later "
                "must use wait"
            )

        if (
            earliest is None
            or earliest
            <= request["request_date"]
        ):

            raise ValueError(
                f"{request_id}: wait requires "
                "a later safe date"
            )

    if status == "not_affordable":

        if method != "not_recommended":

            raise ValueError(
                f"{request_id}: not_affordable "
                "must use not_recommended"
            )

        if payments:

            raise ValueError(
                f"{request_id}: not_affordable "
                "cannot contain payments"
            )

    # ------------------------------------------
    # Method-specific validation
    # ------------------------------------------

    if method == "not_recommended":

        if (
            recommendation[
                "payment_plan"
            ]
            != "none"
        ):

            raise ValueError(
                f"{request_id}: "
                "not_recommended plan "
                "must be none"
            )

    elif method in {
        "full_payment",
        "wait",
    }:

        if len(payments) != 1:

            raise ValueError(
                f"{request_id}: {method} "
                "must contain one payment"
            )

        if (
            abs(
                total_paid
                - requested_amount
            )
            > MONEY_TOLERANCE
        ):

            raise ValueError(
                f"{request_id}: full amount "
                "was not paid"
            )

        if method == "full_payment":

            if (
                payments[0]["date"]
                != request["request_date"]
            ):

                raise ValueError(
                    f"{request_id}: full payment "
                    "must occur on request date"
                )

        if method == "wait":

            payment_date = (
                payments[0]["date"]
            )

            # Wait must actually occur later.
            if (
                payment_date
                <= request[
                    "request_date"
                ]
            ):

                raise ValueError(
                    f"{request_id}: wait payment "
                    "must occur after request date"
                )

            # Must still finish by the user's deadline.
            if (
                payment_date
                > request[
                    "desired_completion_date"
                ]
            ):

                raise ValueError(
                    f"{request_id}: wait payment "
                    "occurs after completion deadline"
                )

            # Without spending changes, the wait
            # payment must equal the baseline earliest
            # safe full-payment date. With changes,
            # the modified timeline can make a different
            # payment date safe.
            if not has_spending_changes:

                if earliest is None:

                    raise ValueError(
                        f"{request_id}: wait has "
                        "no earliest safe date"
                    )

                if payment_date != earliest:

                    raise ValueError(
                        f"{request_id}: wait payment "
                        "must equal earliest safe date"
                    )

    elif method == "partial_payment":

        if len(payments) != 2:

            raise ValueError(
                f"{request_id}: partial payment "
                "must have exactly two payments"
            )

        if (
            payments[0]["date"]
            != request["request_date"]
        ):

            raise ValueError(
                f"{request_id}: first partial "
                "payment must be today"
            )

        if (
            abs(
                payments[0]["amount"]
                - safe_amount
            )
            > MONEY_TOLERANCE
        ):

            raise ValueError(
                f"{request_id}: first partial "
                "payment must equal "
                "amount_safe_to_pay"
            )

        if (
            abs(
                total_paid
                - requested_amount
            )
            > MONEY_TOLERANCE
        ):

            raise ValueError(
                f"{request_id}: partial payments "
                "do not equal requested amount"
            )

        if earliest is None:

            raise ValueError(
                f"{request_id}: partial payment "
                "requires an earliest full date"
            )

        if (
            payments[1]["date"]
            != earliest
        ):

            raise ValueError(
                f"{request_id}: second partial "
                "payment date must equal "
                "earliest full-payment date"
            )

    elif method == "installments":

        options = request_payment_options[
            (
                request_payment_options[
                    "request_id"
                ]
                == request_id
            )
            & (
                request_payment_options[
                    "payment_method"
                ]
                == "installments"
            )
        ]

        supplied_plans = set()

        for _, option in options.iterrows():

            schedule = (
                build_installment_schedule(
                    option
                )
            )

            supplied_plans.add(
                format_payment_plan(
                    schedule
                )
            )

        if (
            recommendation[
                "payment_plan"
            ]
            not in supplied_plans
        ):

            raise ValueError(
                f"{request_id}: installment "
                "schedule does not exactly match "
                "a supplied payment option"
            )

    # ------------------------------------------
    # Spending changes
    # ------------------------------------------

    validate_spending_changes(
        recommendation[
            "spending_changes_needed"
        ]
    )

    # ------------------------------------------
    # Explanation
    # ------------------------------------------

    explanation = str(
        recommendation[
            "decision_explanation"
        ]
    ).strip()

    if not explanation:

        raise ValueError(
            f"{request_id}: empty explanation"
        )

    return True


# --------------------------------------------------
# Build recommendations for all requests
# --------------------------------------------------

def build_output_dataframe(
    request_ids=None,
    show_progress=True
):

    if request_ids is None:

        request_ids = (
            requests[
                "request_id"
            ].tolist()
        )

    rows = []

    failures = []

    total = len(request_ids)

    for index, request_id in enumerate(
        request_ids,
        start=1
    ):

        try:

            recommendation = (
                build_recommendation(
                    request_id
                )
            )

            request = requests[
                requests[
                    "request_id"
                ]
                == request_id
            ].iloc[0]

            validate_recommendation(
                recommendation,
                request
            )

            rows.append(
                recommendation
            )

            if show_progress:

                print(
                    f"[{index}/{total}] "
                    f"{request_id} ✓"
                )

        except Exception as error:

            failures.append(
                (
                    request_id,
                    str(error)
                )
            )

            if show_progress:

                print(
                    f"[{index}/{total}] "
                    f"{request_id} ✗ "
                    f"{error}"
                )

    if failures:

        details = "\n".join(
            (
                f"- {request_id}: "
                f"{message}"
            )
            for request_id, message
            in failures
        )

        raise RuntimeError(
            "Stage 7 failed for "
            f"{len(failures)} request(s):\n"
            f"{details}"
        )

    return pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )


# --------------------------------------------------
# Validate complete output table
# --------------------------------------------------

def validate_output_dataframe(
    output
):

    if list(output.columns) != (
        OUTPUT_COLUMNS
    ):

        raise ValueError(
            "Output columns are not "
            "in the required order."
        )

    if len(output) != len(requests):

        raise ValueError(
            "Output row count does not "
            "match requests.csv."
        )

    if (
        output[
            "request_id"
        ].duplicated().any()
    ):

        raise ValueError(
            "Duplicate request IDs "
            "in output."
        )

    expected_ids = set(
        requests[
            "request_id"
        ]
    )

    output_ids = set(
        output[
            "request_id"
        ]
    )

    if output_ids != expected_ids:

        missing = (
            expected_ids
            - output_ids
        )

        extra = (
            output_ids
            - expected_ids
        )

        raise ValueError(
            f"Request IDs mismatch. "
            f"Missing={missing}, "
            f"extra={extra}"
        )

    for _, row in output.iterrows():

        request = requests[
            requests[
                "request_id"
            ]
            == row["request_id"]
        ].iloc[0]

        validate_recommendation(
            row.to_dict(),
            request
        )

    return True


# --------------------------------------------------
# Final-input safety check
# --------------------------------------------------

def unresolved_event_amounts():

    unresolved = events[
        events["amount"].isna()
    ][
        [
            "event_id",
            "user_id",
            "category",
        ]
    ].copy()

    return unresolved


def require_resolved_inputs():

    unresolved = (
        unresolved_event_amounts()
    )

    if not unresolved.empty:

        event_ids = ", ".join(
            unresolved[
                "event_id"
            ].astype(str)
        )

        raise RuntimeError(
            "Cannot produce final submission yet. "
            "Financial events still have blank "
            "amounts that must be resolved from "
            "their linked images: "
            f"{event_ids}"
        )


# --------------------------------------------------
# Generate output.csv
# --------------------------------------------------

def generate_output(
    output_path=None,
    require_complete_inputs=True
):

    if require_complete_inputs:

        require_resolved_inputs()

    output = (
        build_output_dataframe()
    )

    validate_output_dataframe(
        output
    )

    if output_path is None:

        output_path = (
            ROOT
            / "output.csv"
        )

    output.to_csv(
        output_path,
        index=False,
    )

    # Read it back to prove the serialized
    # CSV itself is valid.
    saved_output = pd.read_csv(
        output_path,
        keep_default_na=False,
    )

    validate_output_dataframe(
        saved_output
    )

    print(
        "\n========================================"
    )

    print(
        "✓ STAGE 7 OUTPUT GENERATED"
    )

    print(
        "Rows:",
        len(saved_output)
    )

    print(
        "File:",
        output_path
    )

    print(
        "========================================"
    )

    return saved_output