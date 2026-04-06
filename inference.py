import os
import sys
import textwrap
import traceback
from typing import List, Optional

from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), 'cashflowmanager'))

from server.cashflowmanager_environment import CashflowmanagerEnvironment
from server.client import groq_policy, clear_action_cache
from server.tasks import grade_episode

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("GROQ_API_KEY")
BENCHMARK = "cashflowmanager"

SUCCESS_SCORE_THRESHOLD = 0.5 

def log_start(task: str, env: str, model: str) -> None:
    """Emits mandatory episode start log."""
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emits mandatory per-step log."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emits mandatory episode end log."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Note: Using 3 decimal places for score as per example although END says <score>
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_task(difficulty: str):
    """
    Executes a single financial strategy task and logs in strict telemetry format.
    """
    log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)
    
    env = CashflowmanagerEnvironment(difficulty=difficulty)
    obs = env.reset(seed=42) # Fixed seed for reproducible baseline
    done = False
    
    clear_action_cache()  # Reset stale LLM decisions from prior episodes
    
    history: List[dict] = []
    logs: List[dict] = []      # Observation-level logs for grading
    rewards: List[float] = []
    cash_hist: List[float] = [obs.cash]
    steps_taken = 0
    score = 0.0
    success = False
    
    max_steps = env.params["max_days"] * env.params["num_invoices"] + 10
    
    try:
        while not done and steps_taken < max_steps:
            steps_taken += 1
            
            action = groq_policy(obs, history)
            
            obs = env.step(action)
            done = obs.done
            
            reward = obs.reward
            rewards.append(reward)
            cash_hist.append(obs.cash)
            
            action_type = "Skip" if action.type == 0 else "Min" if action.type == 1 else "Full"
            action_str = f"{action_type}(inv={action.invoice_id})"
            
            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=None)
            
            # Internal history for LLM context (In-Context Learning)
            history.append({
                "day": obs.day - 1,
                "action": action_type,
                "invoice_id": action.invoice_id,
                "reward": reward,
                "late_fee": obs.metadata.get("late_fee", 0),
                "interest": obs.metadata.get("interest", 0),
            })
            
            # Observation-level logs for grading (matches tasks.py structure)
            logs.append({
                "day": obs.day,
                "cash": obs.cash,
                "credit_used": obs.credit_used,
                "late_fee": obs.metadata.get("late_fee", 0),
                "interest": obs.metadata.get("interest", 0),
                "reward": reward,
            })
            
            if done:
                break
                
        score = grade_episode(difficulty, logs, cash_hist)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception as e:
        print(f"[DEBUG] Runtime Error in {difficulty}: {e}", file=sys.stderr)
        traceback.print_exc()
        score = 0.0
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

def main():
    """Main entry point to run all baseline tasks."""
    for difficulty in ["easy", "medium", "hard"]:
        run_task(difficulty)

if __name__ == "__main__":
    main()
