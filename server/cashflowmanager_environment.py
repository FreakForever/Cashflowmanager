# =========================
# Cashflowmanager Environment (REAL VERSION)
# =========================

import random
import numpy as np
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CashflowmanagerAction, CashflowmanagerObservation, Invoice
except ImportError:
    from models import CashflowmanagerAction, CashflowmanagerObservation, Invoice

from server.reward import compute_reward


# -------------------------
# Difficulty Presets
# -------------------------
DIFFICULTY_PRESETS = {
    "easy": {
        "max_days": 10,
        "num_invoices": 2,
        "amount_range": (50, 200),
        "late_fee": 20,
        "interest": 0.01,
        "starting_cash": 2000,
        "min_payment": 30,
    },
    "medium": {
        "max_days": 5,
        "num_invoices": 3,
        "amount_range": (100, 500),
        "late_fee": 50,
        "interest": 0.02,
        "starting_cash": 1000,
        "min_payment": 50,
    },
    "hard": {
        "max_days": 3,
        "num_invoices": 5,
        "amount_range": (200, 800),
        "late_fee": 100,
        "interest": 0.05,
        "starting_cash": 500,
        "min_payment": 75,
    },
}


class CashflowmanagerEnvironment(Environment):
    """
    RL-based Invoice Payment Environment.

    Agent must decide:
    - Skip payment
    - Pay minimum
    - Pay full

    Goal:
    Minimize penalties + interest while maintaining liquidity.

    Supports difficulty modes: easy, medium, hard
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, difficulty="medium", seed=None):
        self.difficulty = difficulty
        self.seed = seed
        self.params = DIFFICULTY_PRESETS[difficulty]

        self.max_days = self.params["max_days"]
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.cash = self.params["starting_cash"]
        self.credit_used = 0
        self.day = 0
        self.invoices = []

    # -------------------------
    # Generate invoices
    # -------------------------
    def _generate_invoices(self):
        p = self.params
        return [
            Invoice(
                amount=random.randint(p["amount_range"][0], p["amount_range"][1]),
                due_in=random.randint(3, 15),
                late_fee=p["late_fee"],
                min_payment=p["min_payment"],
                interest=p["interest"],
            )
            for _ in range(p["num_invoices"])
        ]

    # -------------------------
    # Reset
    # -------------------------
    def reset(self, difficulty=None, seed=None) -> CashflowmanagerObservation:
        # Allow overriding difficulty/seed on reset
        if difficulty is not None:
            self.difficulty = difficulty
            self.params = DIFFICULTY_PRESETS[difficulty]

        if seed is not None:
            self.seed = seed

        # Seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)

        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.max_days = self.params["max_days"]
        self.cash = self.params["starting_cash"]
        self.credit_used = 0
        self.day = 0
        self.invoices = self._generate_invoices()

        return self._build_obs(reward=0.0, done=False)

    # -------------------------
    # Step
    # -------------------------
    def step(self, action: CashflowmanagerAction) -> CashflowmanagerObservation:
        self._state.step_count += 1

        late_fee_total = 0
        interest_total = 0
        paid = 0

        # --- Apply Action ---
        if action.type != 0:
            # Clamp invoice_id to valid range
            inv_id = min(action.invoice_id, len(self.invoices) - 1)
            inv = self.invoices[inv_id]

            pay_amount = (
                inv.min_payment if action.type == 1 else inv.amount
            )

            pay_amount = min(pay_amount, self.cash)

            self.cash -= pay_amount
            inv.amount -= pay_amount

            if inv.amount <= 0:
                inv.amount = 0
                paid += 1

        # --- Time Progress ---
        for inv in self.invoices:
            inv.due_in -= 1

            # Late fee
            if inv.due_in < 0 and inv.amount > 0:
                late_fee_total += inv.late_fee
                inv.amount += inv.late_fee

            # Interest
            if inv.amount > 0:
                interest = inv.amount * inv.interest
                interest_total += interest
                inv.amount += interest

        # Deduct penalties from cash
        self.cash -= late_fee_total

        # --- Reward ---
        reward = compute_reward(
            cash=self.cash,
            late_fee=late_fee_total,
            interest=interest_total,
            credit_used=self.credit_used,
            paid=paid,
        )

        # --- Update Time ---
        self.day += 1
        done = self.day >= self.max_days

        return self._build_obs(
            reward=reward,
            done=done,
            late_fee=late_fee_total,
            interest=interest_total,
        )

    # -------------------------
    # Build Observation
    # -------------------------
    def _build_obs(self, reward, done, late_fee=0, interest=0):
        return CashflowmanagerObservation(
            day=self.day,
            cash=self.cash,
            credit_used=self.credit_used,
            invoices=self.invoices,
            reward=reward,
            done=done,
            metadata={
                "late_fee": late_fee,
                "interest": interest,
                "step": self._state.step_count,
                "difficulty": self.difficulty,
            },
        )

    # -------------------------
    # State property (required)
    # -------------------------
    @property
    def state(self) -> State:
        return self._state