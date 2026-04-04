def compute_reward(cash, late_fee, interest, credit_used, paid):
    return (
        0.01 * cash
        - 1.0 * late_fee
        - 0.5 * interest
        - 0.2 * credit_used
        + 2.0 * paid
    )