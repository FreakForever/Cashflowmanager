# =========================
# Cashflowmanager Environment Client (UPDATED)
# =========================

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CashflowmanagerAction, CashflowmanagerObservation


class CashflowmanagerEnv(
    EnvClient[CashflowmanagerAction, CashflowmanagerObservation, State]
):
    """
    Client for RL-based Cashflowmanager Environment.

    Supports:
    - reset()
    - step()
    - state()

    Designed for invoice payment scheduling simulation.
    """

    # -------------------------
    # Step Payload
    # -------------------------
    def _step_payload(self, action: CashflowmanagerAction) -> Dict:
        return {
            "type": action.type,
            "invoice_id": action.invoice_id,
        }

    # -------------------------
    # Parse Step Result
    # -------------------------
    def _parse_result(self, payload: Dict) -> StepResult[CashflowmanagerObservation]:
        obs_data = payload.get("observation", {})

        observation = CashflowmanagerObservation(
            day=obs_data.get("day", 0),
            cash=obs_data.get("cash", 0.0),
            credit_used=obs_data.get("credit_used", 0.0),
            invoices=obs_data.get("invoices", []),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    # -------------------------
    # Parse State
    # -------------------------
    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )