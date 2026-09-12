import pandas as pd

from decision import (
    format_money,
    format_payment_plan,
    format_spending_changes,
    payment_option_rank,
    candidate_rank_key,
    determine_affordability_status,
    build_recommendation,
)


# --------------------------------------------------
# 6A — formatting
# --------------------------------------------------

def test_formatting():

    assert (
        format_money(15656000.0)
        == "15656000"
    )

    assert (
        format_money(620.4)
        == "620.40"
    )

    payments = [
        {
            "date":
                pd.Timestamp(
                    "2025-09-07"
                ),
            "amount":
                300.0,
        },
        {
            "date":
                pd.Timestamp(
                    "2025-08-07"
                ),
            "amount":
                300.0,
        },
    ]

    assert (
        format_payment_plan(
            payments
        )
        ==
        (
            "2025-08-07:300|"
            "2025-09-07:300"
        )
    )

    changes = [
        {
            "action":
                "stop",
            "event_id":
                "event_14",
        },
        {
            "action":
                "reduce_to",
            "event_id":
                "event_21",
            "new_amount":
                100.5,
        },
    ]

    assert (
        format_spending_changes(
            changes
        )
        ==
        (
            "stop:event_14|"
            "reduce_to:event_21:100.50"
        )
    )

    print(
        "✓ 6A output formatting passed"
    )


# --------------------------------------------------
# 6B — status rules
# --------------------------------------------------

def test_affordability_status():

    full = {
        "method":
            "full_payment",
        "spending_changes":
            [],
    }

    assert (
        determine_affordability_status(
            full
        )
        == "affordable_now"
    )

    wait = {
        "method":
            "wait",
        "spending_changes":
            [],
    }

    assert (
        determine_affordability_status(
            wait
        )
        == "affordable_later"
    )

    installment = {
        "method":
            "installments",
        "spending_changes":
            [],
    }

    assert (
        determine_affordability_status(
            installment
        )
        == "affordable_with_plan"
    )

    changed_full = {
        "method":
            "full_payment",
        "spending_changes": [
            {
                "action":
                    "stop",
                "event_id":
                    "event_test",
            }
        ],
    }

    assert (
        determine_affordability_status(
            changed_full
        )
        == "affordable_with_plan"
    )

    assert (
        determine_affordability_status(
            None
        )
        == "not_affordable"
    )

    print(
        "✓ 6B affordability statuses passed"
    )


# --------------------------------------------------
# 6C — ranking rules
# --------------------------------------------------

def test_candidate_ranking():

    base = {
        "completes_by_deadline":
            True,
        "spending_changes":
            [],
        "total_paid":
            1000.0,
        "payments": [
            {
                "date":
                    pd.Timestamp(
                        "2025-01-01"
                    ),
                "amount":
                    500.0,
            },
            {
                "date":
                    pd.Timestamp(
                        "2025-02-01"
                    ),
                "amount":
                    500.0,
            },
        ],
    }

    option_10 = {
        **base,
        "payment_option_id":
            "payment_option_10",
    }

    option_2 = {
        **base,
        "payment_option_id":
            "payment_option_02",
    }

    assert (
        candidate_rank_key(
            option_2
        )
        <
        candidate_rank_key(
            option_10
        )
    )

    changed = {
        **base,
        "spending_changes": [
            {
                "action":
                    "stop",
                "event_id":
                    "event_test",
            }
        ],
        "payment_option_id":
            "payment_option_01",
    }

    assert (
        candidate_rank_key(
            option_2
        )
        <
        candidate_rank_key(
            changed
        )
    )

    cheaper = {
        **base,
        "total_paid":
            900.0,
        "payment_option_id":
            "payment_option_99",
    }

    assert (
        candidate_rank_key(
            cheaper
        )
        <
        candidate_rank_key(
            option_2
        )
    )

    print(
        "✓ 6C candidate ranking passed"
    )


# --------------------------------------------------
# 6D — real request_26
# --------------------------------------------------

def test_request_26_recommendation():

    result = build_recommendation(
        "request_26"
    )

    assert (
        result[
            "request_id"
        ]
        == "request_26"
    )

    assert (
        result[
            "amount_safe_to_pay"
        ]
        == 15656000.0
    )

    assert (
        result[
            "affordability_status"
        ]
        == "affordable_now"
    )

    assert (
        result[
            "recommended_payment_method"
        ]
        == "full_payment"
    )

    assert (
        result[
            "payment_plan"
        ]
        ==
        "2025-08-03:15656000"
    )

    assert (
        result[
            "earliest_date_for_full_payment"
        ]
        == "2025-08-03"
    )

    assert (
        result[
            "spending_changes_needed"
        ]
        == "none"
    )

    assert (
        len(
            result[
                "decision_explanation"
            ]
        )
        > 0
    )

    print(
        "✓ request_26 recommendation passed"
    )


if __name__ == "__main__":

    print(
        "\n========================================"
    )
    print(
        "STAGE 6 — DECISION TESTS"
    )
    print(
        "========================================"
    )

    test_formatting()
    test_affordability_status()
    test_candidate_ranking()
    test_request_26_recommendation()

    result = build_recommendation(
        "request_26"
    )

    print(
        "\nFinal Stage 6 result:"
    )

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    print(
        "\n========================================"
    )
    print(
        "✓ ALL STAGE 6 TESTS PASSED"
    )
    print(
        "========================================"
    )