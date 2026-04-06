def compute_reward(cash, late_fee, interest, credit_used, paid):
    return (
        0.002 * cash              # reduced influence
        - 2.0 * late_fee          # stronger penalty
        - 1.5 * interest          # stronger penalty
        - 0.5 * credit_used
        + 10.0 * paid             # incentivize completion
    )