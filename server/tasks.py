import numpy as np
from collections import defaultdict

def grade_easy(logs):
    cost = sum(l['late_fee'] + 0.5 * l['interest'] for l in logs)

    penalty = min(1.0, cost / 500.0)

    score = 1.0 - penalty
    return max(0.0, min(1.0, score))

def grade_medium(logs, cash_hist):
    cost = sum(l['late_fee'] + l['interest'] for l in logs)
    penalty = min(1.0, cost / 1000.0)
    if cash_hist:
        liquidity = np.mean(cash_hist)
    else:
        liquidity = 0.0
    bonus = min(0.1, liquidity / 40000.0)
    score = 1.0 - penalty + bonus
    return max(0.0, min(1.0, score))


def grade_hard(logs, cash_hist):
    cost = sum(l['late_fee'] + l['interest'] for l in logs)
    penalty = min(1.0, cost / 3000.0)
    if cash_hist:
        liquidity = np.mean(cash_hist) - np.std(cash_hist)
        liquidity = max(0.0, liquidity)
    else:
        liquidity = 0.0
    bonus = min(0.15, liquidity / 50000.0)
    score = 1.0 - penalty + bonus
    return max(0.0, min(1.0, score))


GRADERS = {
    "easy":   lambda logs, cash_hist: grade_easy(logs),
    "medium": lambda logs, cash_hist: grade_medium(logs, cash_hist),
    "hard":   lambda logs, cash_hist: grade_hard(logs, cash_hist),
}


def grade_episode(difficulty, logs, cash_hist):
    grader = GRADERS.get(difficulty)
    if grader is None:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    return grader(logs, cash_hist)


def run_task(difficulty, env, policy_fn, seed=42):
    obs = env.reset(difficulty=difficulty, seed=seed)

    def serialize_invoices(invoices):
        return [inv.model_dump() for inv in invoices]

    done = False
    logs = []
    cash_hist = [obs.cash]
    history = []

    day_data = defaultdict(lambda: {
        "day": None,
        "initial_invoices": [],
        "actions": [],
        "end_cash": None,
        "late_fee": 0.0,
        "interest": 0.0,
    })

    current_day = obs.day
    day_data[current_day]["day"] = current_day
    day_data[current_day]["initial_invoices"] = serialize_invoices(obs.invoices)

    params = env.params
    max_steps = params["max_days"] * params["num_invoices"] + 10
    step_count = 0

    while not done and step_count < max_steps:
        action = policy_fn(obs, history)
        prev_day = obs.day
        current_invoice_idx = obs.metadata.get("daily_index", 0)
        obs = env.step(action)
        done = obs.done
        step_count += 1

        action_label = "Skip" if action.type == 0 else \
                       "Pay Min" if action.type == 1 else "Pay Full"

        # Log action into the day it was taken
        day_data[prev_day]["actions"].append({
            "invoice_id": current_invoice_idx,
            "action": action_label,
            "reward": round(obs.reward, 4),
            "late_fee": round(obs.metadata.get("late_fee", 0), 2),
            "interest": round(obs.metadata.get("interest", 0), 4),
            "cash_after": round(obs.cash, 2),
            "credit_used": round(obs.credit_used, 2),
        })
        day_data[prev_day]["late_fee"] += obs.metadata.get("late_fee", 0)
        day_data[prev_day]["interest"] += obs.metadata.get("interest", 0)
        day_data[prev_day]["end_cash"] = round(obs.cash, 2)

        # When day advances, capture new day's invoices
        if obs.day != prev_day and not done:
            day_data[obs.day]["day"] = obs.day
            day_data[obs.day]["initial_invoices"] = serialize_invoices(obs.invoices)

        history.append({
            "day": prev_day,
            "action": action_label,
            "invoice_id": action.invoice_id,
            "reward": obs.reward,
            "late_fee": obs.metadata.get("late_fee", 0),
            "interest": obs.metadata.get("interest", 0),
            "credit_used": obs.credit_used,
        })

        logs.append({
            "day": obs.day,
            "cash": obs.cash,
            "credit_used": obs.credit_used,
            "late_fee": obs.metadata.get("late_fee", 0),
            "interest": obs.metadata.get("interest", 0),
            "reward": obs.reward,
        })

        cash_hist.append(obs.cash)

    score = grade_episode(difficulty, logs, cash_hist)

    #day by day breakdpwn
    episode_days = [
        {
            "day": v["day"],
            "initial_invoices": v["initial_invoices"],
            "actions": v["actions"],
            "end_cash": v["end_cash"],
            "day_late_fee": round(v["late_fee"], 2),
            "day_interest": round(v["interest"], 4),
        }
        for k, v in sorted(day_data.items())
    ]

    return logs, cash_hist, score, history, episode_days