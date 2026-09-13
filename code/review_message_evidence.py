import pandas as pd

from evidence import (
    DATASET,
    load_evidence_cache,
)


def main():

    cache = load_evidence_cache()

    messages = pd.read_csv(
        DATASET / "messages.csv"
    )

    message_lookup = (
        messages
        .set_index("message_id")
        .to_dict("index")
    )

    review_count = 0

    print(
        "\n========================================"
    )
    print(
        "MESSAGE EVIDENCE REVIEW"
    )
    print(
        "========================================"
    )

    for message_id, result in (
        cache["messages"].items()
    ):

        for effect in result["effects"]:

            if effect["confidence"] == "high":
                continue

            review_count += 1

            original = (
                message_lookup.get(
                    message_id,
                    {},
                )
            )

            print()
            print(
                "----------------------------------------"
            )
            print(
                f"{message_id}"
            )
            print(
                "----------------------------------------"
            )

            print(
                "Original message:"
            )
            print(
                original.get(
                    "message_text",
                    "",
                )
            )

            print(
                "\nExtracted effect:"
            )

            for key, value in (
                effect.items()
            ):
                print(
                    f"{key}: {value}"
                )

    print()
    print(
        "========================================"
    )
    print(
        f"Needs review: {review_count}"
    )
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()