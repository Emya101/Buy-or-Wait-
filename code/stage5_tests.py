from datetime import timedelta

import pandas as pd

from data import (
    get_request_state,
    request_payment_options,
)

from simulator import (
    simulate_balance,
)

from payment_plans import (
    add_payments_to_timeline,
    calculate_amount_safe_to_pay,
    find_earliest_full_payment_date,
    build_installment_schedule,
    evaluate_request_payment_plans,
)


# --------------------------------------------------
# 5A — payment injection
# --------------------------------------------------

def test_payment_injection():

    timeline = [
        {
            "date":
                pd.Timestamp(
                    "2025-01-01"
                ),
            "category":
                "rent",
            "direction":
                "debit",
            "amount":
                200.0,
            "source":
                "test",
        },
        {
            "date":
                pd.Timestamp(
                    "2025-01-01"
                ),
            "category":
                "salary",
            "direction":
                "credit",
            "amount":
                500.0,
            "source":
                "test",
        },
    ]

    combined = (
        add_payments_to_timeline(
            timeline,
            [{
                "date":
                    pd.Timestamp(
                        "2025-01-01"
                    ),
                "amount":
                    300.0,
            }]
        )
    )

    # Payment + rent must happen before salary.
    assert (
        combined[0]["direction"]
        == "debit"
    )

    assert (
        combined[1]["direction"]
        == "debit"
    )

    assert (
        combined[2]["direction"]
        == "credit"
    )

    print(
        "✓ 5A payment injection passed"
    )


# --------------------------------------------------
# 5B — safe amount calculation
# --------------------------------------------------

def test_amount_safe_to_pay():

    request = pd.Series({
        "requested_amount": 600.0,
    })

    profile = pd.Series({
        "current_available_balance":
            1000.0,
        "minimum_balance_to_keep":
            300.0,
    })

    timeline = [{
        "date":
            pd.Timestamp(
                "2025-01-01"
            ),
        "category":
            "bill",
        "direction":
            "debit",
        "amount":
            200.0,
        "source":
            "test",
    }]

    safe_amount = (
        calculate_amount_safe_to_pay(
            request,
            profile,
            timeline
        )
    )

    # Baseline low:
    # 1000 - 200 = 800
    #
    # Minimum = 300
    #
    # Headroom = 500
    assert safe_amount == 500.0

    print(
        "✓ 5B safe-payment amount passed"
    )


# --------------------------------------------------
# 5C — earliest full-payment date
# --------------------------------------------------

def test_earliest_full_payment_date():

    request = pd.Series({
        "request_date":
            pd.Timestamp(
                "2025-01-01"
            ),
        "requested_amount":
            600.0,
        "desired_completion_date":
            pd.Timestamp(
                "2025-02-01"
            ),
    })

    profile = pd.Series({
        "current_available_balance":
            1000.0,
        "minimum_balance_to_keep":
            300.0,
    })

    timeline = [
        {
            "date":
                pd.Timestamp(
                    "2025-01-01"
                ),
            "category":
                "bill",
            "direction":
                "debit",
            "amount":
                200.0,
            "source":
                "test",
        },
        {
            "date":
                pd.Timestamp(
                    "2025-01-02"
                ),
            "category":
                "salary",
            "direction":
                "credit",
            "amount":
                200.0,
            "source":
                "test",
        },
    ]

    earliest = (
        find_earliest_full_payment_date(
            request,
            profile,
            timeline
        )
    )

    # Jan 1:
    # 1000 - 600 - 200 = 200
    # unsafe.
    #
    # Jan 2:
    # payment debit occurs before salary:
    # 800 - 600 = 200
    # unsafe.
    #
    # Jan 3:
    # salary has already settled:
    # 1000 - 600 = 400
    # safe.
    assert earliest == pd.Timestamp(
        "2025-01-03"
    )

    print(
        "✓ 5C earliest full-payment date passed"
    )


# --------------------------------------------------
# 5D — supplied installment schedule
# --------------------------------------------------

def test_installment_schedule():

    option = (
        request_payment_options[
            request_payment_options[
                "payment_option_id"
            ]
            == "payment_option_05"
        ]
        .iloc[0]
    )

    schedule = (
        build_installment_schedule(
            option
        )
    )

    assert len(schedule) == 3

    assert (
        schedule[0]["date"]
        == pd.Timestamp(
            "2025-08-08"
        )
    )

    assert (
        schedule[1]["date"]
        == pd.Timestamp(
            "2025-09-07"
        )
    )

    assert (
        schedule[2]["date"]
        == pd.Timestamp(
            "2025-10-07"
        )
    )

    assert (
        schedule[0]["amount"]
        == 15952906.67
    )

    print(
        "✓ 5D supplied installment schedule passed"
    )


# --------------------------------------------------
# 5A–5F — real request_26
# --------------------------------------------------

def test_request_26_stage5():

    analysis = (
        evaluate_request_payment_plans(
            "request_26"
        )
    )

    request = analysis[
        "request"
    ]

    assert (
        analysis[
            "amount_safe_to_pay"
        ]
        == float(
            request[
                "requested_amount"
            ]
        )
    )

    assert (
        analysis[
            "earliest_date_for_full_payment"
        ]
        == request[
            "request_date"
        ]
    )

    full = next(
        candidate
        for candidate
        in analysis["candidates"]
        if candidate[
            "method"
        ]
        == "full_payment"
    )

    assert full["plan_safe"]
    assert full["eligible"]

    # request_26 does not permit
    # partial payment.
    partials = [
        candidate
        for candidate
        in analysis["candidates"]
        if candidate[
            "method"
        ]
        == "partial_payment"
    ]

    assert len(partials) == 0

    installment_candidates = [
        candidate
        for candidate
        in analysis["candidates"]
        if candidate[
            "method"
        ]
        == "installments"
    ]

    assert len(
        installment_candidates
    ) == 3

    # user_26 accepts installments,
    # but all supplied terms exceed
    # the user's five-month limit
    # and/or the request deadline.
    assert not any(
        candidate["eligible"]
        for candidate
        in installment_candidates
    )

    # Full payment works without
    # changing spending, so Stage 5
    # should not search unnecessary
    # spending reductions.
    assert (
        analysis[
            "changed_candidates"
        ]
        == []
    )

    print(
        "✓ request_26 Stage 5 analysis passed"
    )


if __name__ == "__main__":

    print(
        "\n========================================"
    )
    print(
        "STAGE 5 — PAYMENT PLAN TESTS"
    )
    print(
        "========================================"
    )

    test_payment_injection()
    test_amount_safe_to_pay()
    test_earliest_full_payment_date()
    test_installment_schedule()
    test_request_26_stage5()

    analysis = (
        evaluate_request_payment_plans(
            "request_26"
        )
    )

    print(
        "\nRequest:",
        analysis[
            "request"
        ]["request_id"]
    )

    print(
        "Requested amount:",
        analysis[
            "request"
        ]["requested_amount"]
    )

    print(
        "Amount safe today:",
        analysis[
            "amount_safe_to_pay"
        ]
    )

    earliest = analysis[
        "earliest_date_for_full_payment"
    ]

    print(
        "Earliest full-payment date:",
        (
            earliest.date()
            if earliest is not None
            else "none"
        )
    )

    print(
        "\nCandidate plans:"
    )

    for candidate in analysis[
        "candidates"
    ]:

        option_id = (
            candidate[
                "payment_option_id"
            ]
            or "-"
        )

        print(
            f"{candidate['method']:<16}"
            f"| option={option_id:<18}"
            f"| safe={candidate['plan_safe']!s:<5}"
            f"| deadline="
            f"{candidate['completes_by_deadline']!s:<5}"
            f"| preference="
            f"{candidate['preference_eligible']!s:<5}"
            f"| eligible="
            f"{candidate['eligible']}"
        )

    print(
        "\nChanged-plan candidates:",
        len(
            analysis[
                "changed_candidates"
            ]
        )
    )

    print(
        "\n========================================"
    )
    print(
        "✓ ALL STAGE 5 TESTS PASSED"
    )
    print(
        "========================================"
    )