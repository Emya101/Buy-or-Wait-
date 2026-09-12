import pandas as pd

from data import (
    requests,
)

from decision import (
    build_recommendation,
)

from output_generator import (
    OUTPUT_COLUMNS,
    parse_payment_plan,
    validate_recommendation,
    build_output_dataframe,
)


# --------------------------------------------------
# 7A — payment-plan parsing
# --------------------------------------------------

def test_payment_plan_parser():

    payments = parse_payment_plan(
        (
            "2025-08-03:100|"
            "2025-09-03:200"
        )
    )

    assert len(payments) == 2

    assert (
        payments[0]["date"]
        == pd.Timestamp(
            "2025-08-03"
        )
    )

    assert (
        payments[0]["amount"]
        == 100.0
    )

    assert (
        parse_payment_plan(
            "none"
        )
        == []
    )

    print(
        "✓ 7A payment-plan parser passed"
    )


# --------------------------------------------------
# 7B — request_26 final validation
# --------------------------------------------------

def test_request_26_validation():

    result = build_recommendation(
        "request_26"
    )

    request = requests[
        requests[
            "request_id"
        ]
        == "request_26"
    ].iloc[0]

    assert validate_recommendation(
        result,
        request
    )

    print(
        "✓ 7B request_26 validation passed"
    )


# --------------------------------------------------
# 7C — required schema
# --------------------------------------------------

def test_required_schema():

    output = (
        build_output_dataframe(
            request_ids=[
                "request_26"
            ],
            show_progress=False,
        )
    )

    assert (
        list(output.columns)
        == OUTPUT_COLUMNS
    )

    assert len(output) == 1

    print(
        "✓ 7C required schema passed"
    )


# --------------------------------------------------
# 7D — full dataset smoke test
# --------------------------------------------------

def test_full_dataset_build():

    output = (
        build_output_dataframe(
            show_progress=True
        )
    )

    assert (
        len(output)
        == len(requests)
    )

    assert (
        output[
            "request_id"
        ].is_unique
    )

    assert set(
        output[
            "request_id"
        ]
    ) == set(
        requests[
            "request_id"
        ]
    )

    print(
        "✓ 7D full dataset build passed"
    )


if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "STAGE 7 — OUTPUT TESTS"
    )

    print(
        "========================================"
    )

    test_payment_plan_parser()

    test_request_26_validation()

    test_required_schema()

    print(
        "\nRunning every request..."
    )

    test_full_dataset_build()

    print(
        "\n========================================"
    )

    print(
        "✓ ALL STAGE 7 TESTS PASSED"
    )

    print(
        "========================================"
    )