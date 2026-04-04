import numpy as np


# -------------------------
# Task Graders (score 0.0 → 1.0)
# -------------------------

def grade_easy(logs):
    """Lenient: only penalizes late fees."""
    late = sum(l['late_fee'] for l in logs)
    return max(0.0, 1.0 - late / 500.0)


def grade_medium(logs):
    """Moderate: penalizes both late fees and interest."""
    cost = sum(l['late_fee'] + l['interest'] for l in logs)
    return max(0.0, 1.0 - cost / 1000.0)


def grade_hard(logs, cash_hist):
    """Strict: penalizes costs AND rewards liquidity maintenance."""
    cost = sum(l['late_fee'] + l['interest'] for l in logs)
    liquidity = np.mean(cash_hist) if cash_hist else 0
    # Clamp to [0.0, 1.0]
    return min(1.0, max(0.0, 1.0 - cost / 1000.0 + liquidity / 2000.0))


# -------------------------
# Grader Dispatch
# -------------------------
GRADERS = {
    "easy": lambda logs, cash_hist: grade_easy(logs),
    "medium": lambda logs, cash_hist: grade_medium(logs),
    "hard": lambda logs, cash_hist: grade_hard(logs, cash_hist),
}


def grade_episode(difficulty, logs, cash_hist):
    """
    Grade a completed episode based on difficulty.

    Args:
        difficulty: "easy", "medium", or "hard"
        logs: list of dicts with 'late_fee' and 'interest' keys
        cash_hist: list of cash values at each step

    Returns:
        float: score between 0.0 and 1.0
    """
    grader = GRADERS.get(difficulty)
    if grader is None:
        raise ValueError(f"Unknown difficulty: {difficulty}. Must be one of: easy, medium, hard")
    return grader(logs, cash_hist)


# -------------------------
# Task Runner
# -------------------------
def run_task(difficulty, env, policy_fn, seed=42):
    """
    Run a complete task: reset environment, play episode, grade result.

    Args:
        difficulty: "easy", "medium", or "hard"
        env: CashflowmanagerEnvironment instance
        policy_fn: callable(observation, history) -> CashflowmanagerAction
        seed: random seed for reproducibility

    Returns:
        (logs, cash_hist, score, history)
    """
    obs = env.reset(difficulty=difficulty, seed=seed)
    done = False

    logs = []
    cash_hist = [obs.cash]
    history = []

    while not done:
        action = policy_fn(obs, history)
        prev_day = obs.day
        obs = env.step(action)
        done = obs.done

        # Store for in-context learning
        history.append({
            "day": prev_day,
            "action": "Skip" if action.type == 0 else "Min" if action.type == 1 else "Full",
            "invoice_id": action.invoice_id,
            "reward": obs.reward,
            "late_fee": obs.metadata.get("late_fee", 0),
            "interest": obs.metadata.get("interest", 0),
        })

        logs.append({
            "day": obs.day,
            "cash": obs.cash,
            "late_fee": obs.metadata.get("late_fee", 0),
            "interest": obs.metadata.get("interest", 0),
            "reward": obs.reward,
        })

        cash_hist.append(obs.cash)

    score = grade_episode(difficulty, logs, cash_hist)

    return logs, cash_hist, score, history