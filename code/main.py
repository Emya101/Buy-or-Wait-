from simulator import (
    build_financial_timeline,
    simulate_balance,
)


def main():

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
        timeline=timeline,
    )

    print(
        f"{request['request_id']} | "
        f"safe={simulation['safe']} | "
        f"lowest_balance="
        f"{simulation['lowest_balance']:.2f}"
    )


if __name__ == "__main__":
    main()