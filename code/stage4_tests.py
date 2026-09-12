from datetime import timedelta

import pandas as pd

from data import (
    events,
    profiles,
    exchange_rates,
    include_in_cashflow,
    normalize_amount,
    get_request_state,
    get_explicit_future_events,
)

from recurrence import (
    historical_events,
    find_monthly_patterns,
    find_fixed_interval_patterns,
)

from simulator import (
    combine_timelines,
    build_financial_timeline,
    simulate_balance,
)

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

def test_explicit_future_events():

    start = pd.Timestamp(
        "2025-08-03"
    )

    end = (
        start
        + timedelta(days=90)
    )

    test_events = pd.DataFrame([
        {
            "event_id": "test_scheduled_debit",
            "event_date": pd.Timestamp("2025-08-08"),
            "settlement_date": pd.NaT,
            "category": "utilities",
            "direction": "debit",
            "status": "scheduled",
            "normalized_amount": 100.0,
        },
        {
            "event_id": "test_pending_debit",
            "event_date": pd.Timestamp("2025-08-01"),
            "settlement_date": pd.Timestamp("2025-08-07"),
            "category": "rent",
            "direction": "debit",
            "status": "pending",
            "normalized_amount": 200.0,
        },
        {
            "event_id": "test_pending_credit",
            "event_date": pd.Timestamp("2025-08-05"),
            "settlement_date": pd.Timestamp("2025-08-06"),
            "category": "salary",
            "direction": "credit",
            "status": "pending",
            "normalized_amount": 300.0,
        },
        {
            "event_id": "test_scheduled_credit",
            "event_date": pd.Timestamp("2025-08-05"),
            "settlement_date": pd.Timestamp("2025-08-10"),
            "category": "salary",
            "direction": "credit",
            "status": "scheduled",
            "normalized_amount": 400.0,
        },
        {
            "event_id": "test_failed_debit",
            "event_date": pd.Timestamp("2025-08-06"),
            "settlement_date": pd.NaT,
            "category": "utilities",
            "direction": "debit",
            "status": "failed",
            "normalized_amount": 500.0,
        },
        {
            "event_id": "test_historical_debit",
            "event_date": pd.Timestamp("2025-08-01"),
            "settlement_date": pd.Timestamp("2025-08-01"),
            "category": "groceries",
            "direction": "debit",
            "status": "settled",
            "normalized_amount": 600.0,
        },
        {
    "event_id": "test_unsettled_credit",
    "event_date": pd.Timestamp("2025-08-07"),
    "settlement_date": pd.NaT,
    "category": "salary",
    "direction": "credit",
    "status": "scheduled",
    "normalized_amount": 350.0,
},
    ])

    test_events["include_in_cashflow"] = (
        test_events.apply(
            include_in_cashflow,
            axis=1
        )
    )

    future = get_explicit_future_events(
        test_events,
        start,
        end
    )

    by_id = {
        event["event_id"]: event
        for event in future
    }

    assert "test_scheduled_debit" in by_id
    assert "test_pending_debit" in by_id
    assert "test_scheduled_credit" in by_id

    assert "test_pending_credit" not in by_id
    assert "test_failed_debit" not in by_id
    assert "test_historical_debit" not in by_id
    assert "test_unsettled_credit" not in by_id

    assert (
        by_id["test_scheduled_debit"]["date"]
        == pd.Timestamp("2025-08-08")
    )

    assert (
        by_id["test_pending_debit"]["date"]
        == start
    )

    assert (
        by_id["test_scheduled_credit"]["date"]
        == pd.Timestamp("2025-08-10")
    )

    print(
        "✓ 4C explicit-event handling passed"
    )

def test_request_26_has_no_explicit_future_events():

    request, profile, user_events = (
        get_request_state(
            "request_26"
        )
    )

    start = request["request_date"]
    end = start + timedelta(days=90)

    future = get_explicit_future_events(
        user_events,
        start,
        end
    )

    assert len(future) == 0

    print(
        "✓ request_26 explicit-event sanity check passed"
    )

def test_request_26_has_no_explicit_future_events():

    request, profile, user_events = (
        get_request_state(
            "request_26"
        )
    )

    start = request["request_date"]
    end = start + timedelta(days=90)

    future = get_explicit_future_events(
        user_events,
        start,
        end
    )

    assert len(future) == 0

    print(
        "✓ request_26 explicit-event sanity check passed"
    )


def test_timeline_combination():

    projected = [
        {
            "date": pd.Timestamp("2025-08-03"),
            "category": "rent",
            "direction": "debit",
            "amount": 1000.0,
            "source": "projected_monthly",
        },
        {
            "date": pd.Timestamp("2025-08-07"),
            "category": "utilities",
            "direction": "debit",
            "amount": 100.0,
            "source": "projected_monthly",
        },
        {
            "date": pd.Timestamp("2025-08-08"),
            "category": "salary",
            "direction": "credit",
            "amount": 500.0,
            "source": "projected_monthly",
        },
        {
            "date": pd.Timestamp("2025-08-10"),
            "category": "streaming",
            "direction": "debit",
            "amount": 50.0,
            "source": "projected_monthly",
        },
    ]

    explicit = [
        {
            "date": pd.Timestamp("2025-08-07"),
            "event_id": "explicit_utilities",
            "category": "utilities",
            "direction": "debit",
            "amount": 120.0,
            "status": "scheduled",
            "source": "explicit_future_event",
        },
        {
            "date": pd.Timestamp("2025-08-08"),
            "event_id": "explicit_fee",
            "category": "fee",
            "direction": "debit",
            "amount": 25.0,
            "status": "scheduled",
            "source": "explicit_future_event",
        },
    ]

    timeline = combine_timelines(
        projected,
        explicit
    )

    assert len(timeline) == 5

    utilities = [
        event
        for event in timeline
        if (
            event["category"] == "utilities"
            and event["date"]
            == pd.Timestamp("2025-08-07")
        )
    ]

    assert len(utilities) == 1
    assert utilities[0]["amount"] == 120.0
    assert (
        utilities[0]["source"]
        == "explicit_future_event"
    )

    dates = [
        event["date"]
        for event in timeline
    ]

    assert dates == sorted(dates)

    august_8 = [
        event
        for event in timeline
        if event["date"]
        == pd.Timestamp("2025-08-08")
    ]

    assert august_8[0]["direction"] == "debit"
    assert august_8[1]["direction"] == "credit"

    print("✓ 4D timeline combination passed")


def test_request_26_timeline():

    request, profile, timeline = (
        build_financial_timeline(
            "request_26"
        )
    )

    start = request[
        "request_date"
    ]

    end = (
        start
        + timedelta(days=90)
    )

    assert len(timeline) > 0

    # Every event must remain inside
    # the 90-day forecast.
    assert all(
        start
        <= event["date"]
        <= end
        for event in timeline
    )

    # Timeline must already be sorted.
    dates = [
        event["date"]
        for event in timeline
    ]

    assert dates == sorted(dates)

    # request_26 begins on Aug 3.
    # We know rent recurs on day 3.
    rent_august = [
        event
        for event in timeline
        if (
            event["category"]
            == "rent"
            and event["direction"]
            == "debit"
            and event["date"]
            == pd.Timestamp(
                "2025-08-03"
            )
        )
    ]

    assert len(
        rent_august
    ) == 1

    # We detected salary on day 8.
    salary_august = [
        event
        for event in timeline
        if (
            event["category"]
            == "salary"
            and event["direction"]
            == "credit"
            and event["date"]
            == pd.Timestamp(
                "2025-08-08"
            )
        )
    ]

    assert len(
        salary_august
    ) == 1

    print(
        "✓ request_26 90-day timeline passed"
    )

def test_balance_simulator():

    # ------------------------------------------
    # SAFE SCENARIO
    # ------------------------------------------

    safe_timeline = [
        {
            "date": pd.Timestamp(
                "2025-08-01"
            ),
            "category": "test_expense",
            "direction": "debit",
            "amount": 200.0,
            "source": "test",
        },
        {
            "date": pd.Timestamp(
                "2025-08-02"
            ),
            "category": "test_income",
            "direction": "credit",
            "amount": 100.0,
            "source": "test",
        },
        {
            "date": pd.Timestamp(
                "2025-08-03"
            ),
            "category": "test_expense",
            "direction": "debit",
            "amount": 400.0,
            "source": "test",
        },
    ]

    safe_result = simulate_balance(
        starting_balance=1000,
        minimum_balance=300,
        timeline=safe_timeline
    )

    # 1000
    # -200 = 800
    # +100 = 900
    # -400 = 500

    assert (
        safe_result["ending_balance"]
        == 500.0
    )

    assert (
        safe_result["lowest_balance"]
        == 500.0
    )

    assert (
        safe_result["safe"]
        is True
    )


    # ------------------------------------------
    # UNSAFE SCENARIO
    # ------------------------------------------

    unsafe_timeline = [
        {
            "date": pd.Timestamp(
                "2025-08-01"
            ),
            "category": "large_bill",
            "direction": "debit",
            "amount": 800.0,
            "source": "test",
        },
        {
            "date": pd.Timestamp(
                "2025-08-02"
            ),
            "category": "income",
            "direction": "credit",
            "amount": 600.0,
            "source": "test",
        },
    ]

    unsafe_result = simulate_balance(
        starting_balance=1000,
        minimum_balance=300,
        timeline=unsafe_timeline
    )

    # 1000
    # -800 = 200  ← below minimum
    # +600 = 800
    #
    # Ending balance is safe,
    # but the forecast itself is NOT.

    assert (
        unsafe_result["ending_balance"]
        == 800.0
    )

    assert (
        unsafe_result["lowest_balance"]
        == 200.0
    )

    assert (
        unsafe_result["safe"]
        is False
    )

    assert (
        unsafe_result[
            "lowest_balance_date"
        ]
        == pd.Timestamp(
            "2025-08-01"
        )
    )

    print(
        "✓ 4E balance simulator passed"
    )

def test_request_26_balance_simulation():

    request, profile, timeline = (
        build_financial_timeline(
            "request_26"
        )
    )

    result = simulate_balance(
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
        timeline=timeline
    )

    assert (
        len(result["timeline"])
        == len(timeline)
    )

    assert pd.notna(
        result["ending_balance"]
    )

    assert pd.notna(
        result["lowest_balance"]
    )

    # Verify simulator's safety result
    # agrees with every timeline row.
    expected_safe = all(
        event["above_minimum"]
        for event in result["timeline"]
    )

    assert (
        result["safe"]
        == expected_safe
    )

    print(
        "✓ request_26 baseline simulation passed"
    )

if __name__ == "__main__":

    # ==================================================
    # 4A — REQUEST LOADING / NORMALIZATION
    # ==================================================

    print("\n========================================")
    print("4A — REQUEST LOADING")
    print("========================================")

    test_request_loading()

    request, profile, user_events = (
        get_request_state("request_26")
    )

    print("\nTest data:")
    print("Request ID:", request["request_id"])
    print("User ID:", request["user_id"])
    print("Home currency:", profile["home_currency"])
    print(
        "Starting balance:",
        profile["current_available_balance"]
    )
    print(
        "Minimum balance:",
        profile["minimum_balance_to_keep"]
    )
    print(
        "Events loaded:",
        len(user_events)
    )

    print("\nFirst 3 normalized events:")

    print(
        user_events[
            [
                "event_id",
                "category",
                "direction",
                "amount",
                "currency",
                "normalized_amount"
            ]
        ].head(3)
    )

    print("\n✓ 4A PASSED")


    # ==================================================
    # 4B — RECURRENCE DETECTION
    # ==================================================

    print("\n========================================")
    print("4B — RECURRENCE DETECTION")
    print("========================================")

    test_recurrence_detection()

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

    print("\nMonthly patterns detected:")

    for pattern in monthly:

        print(
            pattern["category"],
            "→ day",
            pattern["day_of_month"],
            "|",
            pattern["direction"],
            "| amount:",
            round(
                pattern["estimated_amount"],
                2
            )
        )

    print("\nFixed interval patterns detected:")

    for pattern in intervals:

        print(
            pattern["category"],
            "→ every",
            pattern["interval_days"],
            "days",
            "|",
            pattern["direction"],
            "| amount:",
            round(
                pattern["estimated_amount"],
                2
            )
        )

    print("\n✓ 4B PASSED")


    # ==================================================
    # 4C — EXPLICIT FUTURE EVENTS
    # ==================================================

    print("\n========================================")
    print("4C — EXPLICIT FUTURE EVENTS")
    print("========================================")

    test_explicit_future_events()

    print("\nSynthetic rules verified:")
    print("✓ scheduled future debit → INCLUDED")
    print("✓ pending debit → INCLUDED immediately")
    print("✓ scheduled settled credit → INCLUDED")
    print("✓ pending credit → EXCLUDED")
    print("✓ failed debit → EXCLUDED")
    print("✓ historical settled debit → EXCLUDED")
    print("✓ unsettled credit → EXCLUDED")

    print("\n✓ 4C SYNTHETIC TEST PASSED")


    # ==================================================
    # 4C — REAL DATA SANITY CHECK
    # ==================================================

    print("\n========================================")
    print("4C — REAL DATA CHECK")
    print("========================================")

    test_request_26_has_no_explicit_future_events()

    start = request["request_date"]
    end = start + timedelta(days=90)

    future = get_explicit_future_events(
        user_events,
        start,
        end
    )

    print("\nRequest:", request["request_id"])
    print("Forecast start:", start.date())
    print("Forecast end:", end.date())
    print(
        "Explicit future events found:",
        len(future)
    )

    if future:

        for event in future:

            print(
                event["date"].date(),
                event["direction"],
                event["category"],
                round(
                    event["amount"],
                    2
                )
            )

    else:

        print(
            "No explicit future events "
            "for this request."
        )

    print("\n✓ 4C REAL DATA CHECK PASSED")

    # ==================================================
    # 4D — COMPLETE 90-DAY TIMELINE
    # ==================================================

    print("\n========================================")
    print("4D — FINANCIAL TIMELINE")
    print("========================================")

    test_timeline_combination()
    test_request_26_timeline()

    request, profile, timeline = (
        build_financial_timeline(
            "request_26"
        )
    )

    print("\nRequest:", request["request_id"])

    print(
        "Forecast:",
        request["request_date"].date(),
        "→",
        (
            request["request_date"]
            + timedelta(days=90)
        ).date()
    )

    print(
        "Timeline events:",
        len(timeline)
    )

    print(
        "\nFirst 15 timeline events:"
    )

    for event in timeline[:15]:

        print(
            event["date"].date(),
            "|",
            f"{event['direction']:<6}",
            "|",
            f"{event['category']:<15}",
            "|",
            f"{event['amount']:.2f}",
            "|",
            event["source"]
        )

    print("\n✓ 4D REAL TIMELINE PASSED")

        # ==================================================
    # 4E — RUNNING BALANCE SIMULATION
    # ==================================================

    print("\n========================================")
    print("4E — RUNNING BALANCE SIMULATION")
    print("========================================")

    test_balance_simulator()
    test_request_26_balance_simulation()

    request, profile, timeline = (
        build_financial_timeline(
            "request_26"
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
        timeline=timeline
    )

    print(
        "\nStarting balance:",
        f"{simulation['starting_balance']:.2f}"
    )

    print(
        "Minimum allowed:",
        f"{simulation['minimum_balance']:.2f}"
    )

    print("\nFirst 15 simulated events:")

    for event in simulation[
        "timeline"
    ][:15]:

        status = (
            "SAFE"
            if event["above_minimum"]
            else "BELOW MINIMUM"
        )

        print(
            event["date"].date(),
            "|",
            f"{event['direction']:<6}",
            "|",
            f"{event['category']:<15}",
            "|",
            f"{event['amount']:>12.2f}",
            "| balance:",
            f"{event['balance_after']:>12.2f}",
            "|",
            status
        )

    print(
        "\nLowest balance:",
        f"{simulation['lowest_balance']:.2f}"
    )

    if (
        simulation[
            "lowest_balance_date"
        ]
        is not None
    ):

        print(
            "Lowest balance date:",
            simulation[
                "lowest_balance_date"
            ].date()
        )

    print(
        "Ending balance:",
        f"{simulation['ending_balance']:.2f}"
    )

    print(
        "Minimum-balance headroom:",
        f"{simulation['lowest_balance'] - simulation['minimum_balance']:.2f}"
    )

    print(
        "Baseline forecast safe:",
        simulation["safe"]
    )

    print(
        "\n✓ 4E REAL BALANCE SIMULATION PASSED"
    )

    # ==================================================
    # FINAL RESULT
    # ==================================================

    print("\n========================================")
    print("✓ ALL STAGE 4A–4E TESTS PASSED")
    print("========================================")