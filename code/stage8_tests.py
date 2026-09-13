import pandas as pd

from data import normalize_amount
from message_effects import (
    get_applicable_message_effects,
    apply_targeted_event_effects,
    apply_recurring_message_effects,
    build_one_time_message_events,
)


def test_future_message_excluded():
    before = get_applicable_message_effects(
        "user_154",
        pd.Timestamp("2024-11-24"),
    )

    message_ids = {
        effect["message_id"]
        for effect in before
    }

    assert "message_119" not in message_ids
    print("✓ 8B2-A future-message exclusion passed")


def test_message_119_review_targets():
    effects = get_applicable_message_effects(
        "user_154",
        pd.Timestamp("2024-11-25"),
    )

    message_119 = [
        effect
        for effect in effects
        if effect["message_id"] == "message_119"
    ]

    stop = next(
        effect
        for effect in message_119
        if effect["effect_type"] == "recurring_stop"
    )

    change = next(
        effect
        for effect in message_119
        if effect["effect_type"] == "recurring_amount_change"
    )

    assert stop["target_event_id"] == "event_14173"
    assert change["target_event_id"] == "event_14180"
    assert float(change["amount"]) == 1628.0

    print("✓ 8B2-B message_119 manual review passed")


def test_targeted_event_change():
    events = pd.DataFrame([
        {
            "event_id": "event_test",
            "event_date": pd.Timestamp("2025-01-10"),
            "settlement_date": pd.Timestamp("2025-01-10"),
            "amount": 100.0,
            "currency": "USD",
            "status": "scheduled",
        }
    ])

    effects = [
        {
            "effect_type": "event_date_change",
            "target_event_id": "event_test",
            "replacement_date": "2025-01-15",
        }
    ]

    changed = apply_targeted_event_effects(
        events,
        effects,
    )

    assert changed.iloc[0]["event_date"] == pd.Timestamp("2025-01-15")
    assert changed.iloc[0]["settlement_date"] == pd.Timestamp("2025-01-15")

    print("✓ 8B2-C targeted event change passed")


def test_anchored_recurring_changes():
    projected = [
        {
            "date": pd.Timestamp("2024-12-15"),
            "category": "salary",
            "direction": "credit",
            "amount": 1009.36,
            "source": "projected_monthly",
            "anchor_event_id": "event_14180",
        },
        {
            "date": pd.Timestamp("2024-12-20"),
            "category": "salary",
            "direction": "credit",
            "amount": 600.0,
            "source": "projected_monthly",
            "anchor_event_id": "event_14173",
        },
        {
            "date": pd.Timestamp("2025-01-15"),
            "category": "salary",
            "direction": "credit",
            "amount": 1009.36,
            "source": "projected_monthly",
            "anchor_event_id": "event_14180",
        },
        {
            "date": pd.Timestamp("2025-01-20"),
            "category": "salary",
            "direction": "credit",
            "amount": 600.0,
            "source": "projected_monthly",
            "anchor_event_id": "event_14173",
        },
    ]

    effects = [
        {
            "effect_type": "recurring_stop",
            "target_event_id": "event_14173",
            "category": "salary",
            "direction": "credit",
            "amount": None,
            "currency": "EUR",
            "effective_date": "2024-11-25",
            "confirmed": True,
            "message_id": "message_119",
            "sent_at": pd.Timestamp("2024-11-25"),
        },
        {
            "effect_type": "recurring_amount_change",
            "target_event_id": "event_14180",
            "category": "salary",
            "direction": "credit",
            "amount": 1628,
            "currency": "EUR",
            "effective_date": "2024-11-25",
            "confirmed": True,
            "message_id": "message_119",
            "sent_at": pd.Timestamp("2024-11-25"),
        },
    ]

    changed = apply_recurring_message_effects(
        projected_events=projected,
        effects=effects,
        request_date=pd.Timestamp("2024-11-25"),
        end_date=pd.Timestamp("2025-02-23"),
        home_currency="EUR",
        normalize_amount=normalize_amount,
    )

    assert all(
        event.get("anchor_event_id") != "event_14173"
        for event in changed
    )

    primary = [
        event
        for event in changed
        if event.get("anchor_event_id") == "event_14180"
    ]

    assert len(primary) == 2
    assert all(event["amount"] == 1628.0 for event in primary)

    print("✓ 8B2-D anchored recurring changes passed")


def test_unknown_recurring_amount_not_created():
    changed = apply_recurring_message_effects(
        projected_events=[],
        effects=[
            {
                "effect_type": "recurring_start",
                "target_event_id": None,
                "category": "childcare",
                "direction": "debit",
                "amount": None,
                "currency": "EUR",
                "effective_date": "2025-08-01",
                "confirmed": True,
                "message_id": "message_test",
                "sent_at": pd.Timestamp("2025-07-20"),
            }
        ],
        request_date=pd.Timestamp("2025-07-25"),
        end_date=pd.Timestamp("2025-10-23"),
        home_currency="EUR",
        normalize_amount=normalize_amount,
    )

    assert changed == []
    print("✓ 8B2-E unknown recurring amount ignored safely")


def test_unconfirmed_one_time_not_added():
    events = build_one_time_message_events(
        effects=[
            {
                "effect_type": "one_time_credit",
                "amount": 500.0,
                "currency": "USD",
                "replacement_date": "2025-08-15",
                "confirmed": False,
                "message_id": "message_test",
            }
        ],
        request_date=pd.Timestamp("2025-08-01"),
        end_date=pd.Timestamp("2025-10-30"),
        home_currency="USD",
        normalize_amount=normalize_amount,
    )

    assert events == []
    print("✓ 8B2-F unconfirmed one-time credit excluded")


if __name__ == "__main__":
    print("\n========================================")
    print("STAGE 8B2 — MESSAGE APPLICATION TESTS")
    print("========================================")

    test_future_message_excluded()
    test_message_119_review_targets()
    test_targeted_event_change()
    test_anchored_recurring_changes()
    test_unknown_recurring_amount_not_created()
    test_unconfirmed_one_time_not_added()

    print("\n========================================")
    print("✓ ALL STAGE 8B2 TESTS PASSED")
    print("========================================")
