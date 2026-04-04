import sys
import os

# Add cashflowmanager directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cashflowmanager'))

from server.cashflowmanager_environment import CashflowmanagerEnvironment
from server.client import groq_policy
from server.tasks import run_task


def run_mode(difficulty):
    """Run a single difficulty mode and print results."""
    print(f"\n{'='*60}")
    print(f"  TASK: {difficulty.upper()}")
    print(f"{'='*60}")

    env = CashflowmanagerEnvironment(difficulty=difficulty)
    logs, cash_hist, score, history = run_task(
        difficulty=difficulty,
        env=env,
        policy_fn=groq_policy,
        seed=42,
    )

    # Print step-by-step
    print(f"\n{'DAY':<5} | {'CASH':<10} | {'LATE FEE':<10} | {'INTEREST':<10} | {'REWARD':<10}")
    print("-" * 55)

    total_reward = 0
    for day in logs:
        print(
            f"{day['day']:<5} | "
            f"{day['cash']:<10.2f} | "
            f"{day['late_fee']:<10.2f} | "
            f"{day['interest']:<10.2f} | "
            f"{day['reward']:<10.2f}"
        )
        total_reward += day['reward']

    print("-" * 55)
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Cash:   {cash_hist[-1]:.2f}")
    print(f"TASK SCORE:   {score:.4f}  (0.0 = fail, 1.0 = perfect)")

    return difficulty, score, total_reward


def main():
    print("=" * 60)
    print("  CASHFLOW RL SIMULATOR — ALL TASKS")
    print("  Using Groq LLM (LLama 3.1) as policy agent")
    print("=" * 60)

    results = []
    for diff in ["easy", "medium", "hard"]:
        try:
            result = run_mode(diff)
            results.append(result)
        except Exception as e:
            import traceback
            print(f"\n[ERROR in {diff}]")
            traceback.print_exc()
            results.append((diff, -1, 0))

    # Summary table
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'TASK':<10} | {'SCORE':<10} | {'TOTAL REWARD':<15}")
    print("-" * 40)
    for diff, score, reward in results:
        score_str = f"{score:.4f}" if score >= 0 else "ERROR"
        print(f"{diff:<10} | {score_str:<10} | {reward:<15.2f}")
    print("=" * 60)

    # Check all scores valid
    all_valid = all(0.0 <= s <= 1.0 for _, s, _ in results if s >= 0)
    if all_valid and len(results) == 3:
        print("\n✅ All 3 tasks completed with valid scores [0.0 — 1.0]")
    else:
        print("\n⚠️  Some tasks failed or produced invalid scores")

    print("\nDone.")


if __name__ == "__main__":
    main()
