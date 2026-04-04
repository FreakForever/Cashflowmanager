# =========================
# OpenEnv FastAPI + Gradio UI
# =========================

from fastapi import FastAPI
import threading

# ---- OpenEnv Server ----
try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with `uv sync`"
    ) from e

try:
    from ..models import CashflowmanagerAction, CashflowmanagerObservation
    from .cashflowmanager_environment import CashflowmanagerEnvironment
except ImportError:
    from models import CashflowmanagerAction, CashflowmanagerObservation
    from server.cashflowmanager_environment import CashflowmanagerEnvironment


# Create OpenEnv FastAPI app (REQUIRED)
app: FastAPI = create_app(
    CashflowmanagerEnvironment,
    CashflowmanagerAction,
    CashflowmanagerObservation,
    env_name="cashflowmanager",
    max_concurrent_envs=1,
)

# ---- OPTIONAL: Gradio UI ----
import gradio as gr
from server.cashflowmanager_environment import CashflowmanagerEnvironment
from server.client import groq_policy
from server.tasks import run_task, grade_episode


def run_simulation(difficulty="medium"):
    """
    Run a full RL episode with the Groq LLM policy at a given difficulty.
    Returns formatted results with the graded score.
    """
    env = CashflowmanagerEnvironment(difficulty=difficulty)
    logs, cash_hist, score, history = run_task(
        difficulty=difficulty,
        env=env,
        policy_fn=groq_policy,
        seed=42,
    )

    # Build result summary
    result = {
        "difficulty": difficulty,
        "score": round(score, 4),
        "total_reward": round(sum(l["reward"] for l in logs), 2),
        "final_cash": round(logs[-1]["cash"], 2) if logs else 0,
        "total_late_fees": round(sum(l["late_fee"] for l in logs), 2),
        "total_interest": round(sum(l["interest"] for l in logs), 2),
        "steps": logs,
    }

    return result


def run_all_tasks():
    """Run all 3 difficulty modes and return a comparison."""
    results = {}
    for diff in ["easy", "medium", "hard"]:
        try:
            results[diff] = run_simulation(diff)
        except Exception as e:
            results[diff] = {"error": str(e)}
    return results


def launch_gradio():
    with gr.Blocks(title="Cashflow RL Simulator") as demo:
        gr.Markdown("# 💰 Cashflow RL Simulator")
        gr.Markdown(
            "Simulates invoice payment decisions using an RL environment "
            "with a Groq LLM (LLama 3.1) as the policy agent."
        )

        with gr.Row():
            difficulty_input = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="medium",
                label="Difficulty",
                info="Easy: lenient scoring, more time, fewer invoices. "
                     "Hard: strict scoring, less time, more invoices.",
            )
            run_btn = gr.Button("▶ Run Simulation", variant="primary")
            run_all_btn = gr.Button("🔄 Run All 3 Tasks")

        output = gr.JSON(label="Simulation Results")

        run_btn.click(fn=run_simulation, inputs=[difficulty_input], outputs=[output])
        run_all_btn.click(fn=run_all_tasks, inputs=[], outputs=[output])

    demo.launch(server_name="0.0.0.0", server_port=7860)


# ---- Run both servers ----
def main():
    import uvicorn

    # Run Gradio in background thread
    threading.Thread(target=launch_gradio, daemon=True).start()

    # Run FastAPI (OpenEnv backend)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()