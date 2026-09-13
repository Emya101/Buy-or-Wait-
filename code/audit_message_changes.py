from decision import build_recommendation
from data import get_request_state
from message_effects import get_applicable_message_effects
import simulator


changed_ids = [
    "request_29",
    "request_36",
    "request_45",
    "request_49",
    "request_61",
    "request_68",
    "request_73",
    "request_75",
    "request_83",
    "request_87",
    "request_110",
    "request_111",
    "request_119",
    "request_122",
    "request_127",
    "request_130",
    "request_133",
    "request_154",
    "request_155",
    "request_157",
    "request_165",
    "request_166",
    "request_178",
    "request_193",
    "request_194",
    "request_201",
    "request_213",
    "request_216",
    "request_219",
    "request_237",
    "request_241",
    "request_246",
    "request_265",
]


def short(result):
    return {
        "safe": result[
            "amount_safe_to_pay"
        ],
        "status": result[
            "affordability_status"
        ],
        "method": result[
            "recommended_payment_method"
        ],
        "earliest": result[
            "earliest_date_for_full_payment"
        ],
        "changes": result[
            "spending_changes_needed"
        ],
    }


for request_id in changed_ids:

    request, profile, _ = (
        get_request_state(
            request_id
        )
    )

    effects = (
        get_applicable_message_effects(
            request["user_id"],
            request["request_id"],
            request["request_date"],
        )
    )

    # Message-aware result
    after = build_recommendation(
        request_id
    )

    # Temporarily disable messages
    original = (
        simulator
        .get_applicable_message_effects
    )

    simulator.get_applicable_message_effects = (
        lambda *args, **kwargs: []
    )

    try:
        before = build_recommendation(
            request_id
        )
    finally:
        simulator.get_applicable_message_effects = (
            original
        )

    print()
    print("=" * 70)
    print(request_id)
    print("=" * 70)

    print("BEFORE:", short(before))
    print("AFTER: ", short(after))

    print("MESSAGE EFFECTS:")

    for effect in effects:
        print(
            " ",
            effect.get("message_id"),
            "|",
            effect.get("effect_type"),
            "| category:",
            effect.get("category"),
            "| direction:",
            effect.get("direction"),
            "| amount:",
            effect.get("amount"),
            "| effective:",
            effect.get("effective_date"),
            "| replacement:",
            effect.get("replacement_date"),
            "| target:",
            effect.get("target_event_id"),
            "| status:",
            effect.get("new_status"),
            "| confirmed:",
            effect.get("confirmed"),
        )