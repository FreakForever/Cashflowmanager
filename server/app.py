from fastapi import FastAPI
import gradio as gr

from openenv.core.env_server.http_server import create_app

try:
    from ..models import CashflowmanagerAction, CashflowmanagerObservation
    from .cashflowmanager_environment import CashflowmanagerEnvironment
except ImportError:
    from models import CashflowmanagerAction, CashflowmanagerObservation
    from server.cashflowmanager_environment import CashflowmanagerEnvironment


app: FastAPI = create_app(
    CashflowmanagerEnvironment,
    CashflowmanagerAction,
    CashflowmanagerObservation,
    env_name="cashflowmanager",
    max_concurrent_envs=1,
)

from server.client import groq_policy
from server.tasks import run_task
import pandas as pd
import time


def run_simulation(difficulty="medium"):
    env = CashflowmanagerEnvironment(difficulty=difficulty)

    logs, cash_hist, score, history, episode_days = run_task(
        difficulty=difficulty,
        env=env,
        policy_fn=groq_policy,
        seed=int(time.time()),
    )

    full_result = {
        "difficulty": difficulty,
        "score": round(score, 4),
        "total_reward": round(sum(l["reward"] for l in logs), 2),
        "final_cash": round(logs[-1]["cash"], 2) if logs else 0,
        "total_credit_used": round(logs[-1]["credit_used"], 2) if logs else 0,
        "total_late_fees": round(sum(l["late_fee"] for l in logs), 2),
        "total_interest": round(sum(l["interest"] for l in logs), 4),
        "episode": episode_days,
    }

    rows = []
    for day in episode_days:
        for action in day["actions"]:
            rows.append({
                "Day": day["day"],
                "Type": "action",
                "Invoice ID": action["invoice_id"],
                "Action": action["action"],
                "Cash": round(action["cash_after"], 2),
                "Credit": round(action.get("credit_used", 0), 2),
                "Late Fee": round(action["late_fee"], 2),
                "Interest": round(action["interest"], 4),
                "Reward": round(action["reward"], 4),
            })

        rows.append({
            "Day": day["day"],
            "Type": "── day end",
            "Invoice ID": "",
            "Action":     "",
            "Cash":       round(day["end_cash"] or 0, 2),
            "Credit":     "",
            "Late Fee":   round(day["day_late_fee"] or 0, 2),
            "Interest":   round(day["day_interest"] or 0, 4),
            "Reward":     "",
        })

    df = pd.DataFrame(rows, columns=["Day","Type","Invoice ID","Action","Cash","Credit","Late Fee","Interest","Reward"])
    return full_result, df


def build_ui():
    with gr.Blocks(title="Cashflow RL Simulator") as demo:
        gr.Markdown("# 💸 Cashflow RL Simulator")
        gr.Markdown(
            "Simulates invoice payment decisions using an RL environment "
            "with a Groq LLM (LLaMA 3.1) as the policy agent."
        )

        with gr.Row():
            difficulty = gr.Dropdown(
                ["easy", "medium", "hard"],
                value="medium",
                label="Select Difficulty",
                scale=3
            )
            run_btn = gr.Button("Run Simulation", variant="primary", scale=1)

        with gr.Row():
            with gr.Column(scale=1):
                summary = gr.JSON(label="Results Summary")

            with gr.Column(scale=3):
                table = gr.Dataframe(
                    label="Step-by-Step History",
                    headers=["Day","Type","Invoice ID","Action","Cash","Credit","Late Fee","Interest","Reward"],
                    wrap=False,
                )

        run_btn.click(
            run_simulation,
            inputs=[difficulty],
            outputs=[summary, table],
        )

    return demo


# Mount Gradio at ROOT
gradio_app = build_ui()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")


# Default port for uvicorn CLI is 8000
USE_PORT = 8000

@app.on_event("startup")
async def _print_ui_url():
    import logging
    logger = logging.getLogger("uvicorn.error")
    # logger.info(f"\033[1mCashflow UI running on http://127.0.0.1:{USE_PORT}/ui\033[0m")


def main():
    import uvicorn
    # Use 7860 as the default port for local/Space deployment
    uvicorn.run(app, host="0.0.0.0", port=7860)
if __name__ == "__main__":
    main()