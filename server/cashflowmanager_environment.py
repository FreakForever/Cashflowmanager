import random
import numpy as np
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from cashflowmanager.models import CashflowmanagerAction, CashflowmanagerObservation, Invoice
    from cashflowmanager.server.reward import compute_reward
except ImportError:
    try:
        from models import CashflowmanagerAction, CashflowmanagerObservation, Invoice
        from server.reward import compute_reward
    except ImportError:
        from ..models import CashflowmanagerAction, CashflowmanagerObservation, Invoice
        from .reward import compute_reward

DIFFICULTY_PRESETS = {
    "easy": {
        "max_days": 3,
        "num_invoices": 3,
        "amount_range": (50, 200),
        "credit_limit": 500,
        "due_day": (7, 10),
        "late_fee": 20,
        "interest": 0.01,
        "starting_cash": 1000,
        "min_payment": 30,
    },
    "medium": {
        "max_days": 5,
        "num_invoices": 3,
        "amount_range": (300, 450),
        "credit_limit": 700,
        "due_day": (5, 7),
        "late_fee": 50,
        "interest": 0.02,
        "starting_cash": 1200,
        "min_payment": 50,
    },
    "hard": {
        "max_days": 7,
        "num_invoices": 3,
        "amount_range": (400, 700),
        "credit_limit": 800,
        "due_day": (2, 5),
        "late_fee": 100,
        "interest": 0.05,
        "starting_cash": 2500,
        "min_payment": 100,
    },
}


class CashflowmanagerEnvironment(Environment):
    """
    RL-based Invoice Payment Environment.

    Each day, num_invoices fresh invoices are generated.
    Agent handles them one by one (one action per invoice).
    Unpaid invoices carry over with compounding interest + late fees.

    Total steps = max_days × num_invoices + extra_buff
      easy:   3 days × 3 invoices = 9 steps + buff
      medium:  5 days × 3 invoices = 15 steps + buff
      hard:    7 days × 3 invoices = 21 steps + buff
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
        self.day = 0            # reset() sets it to 1 before episode starts
        self.invoices = []
        self.daily_queue = []
        self.daily_index = 0

    def _generate_daily_invoices(self):
        p = self.params
        return [
            Invoice(
                amount=random.randint(p["amount_range"][0], p["amount_range"][1]),
                due_in=random.randint(p["due_day"][0], p["due_day"][1]),
                late_fee=p["late_fee"],
                min_payment=p["min_payment"],
                interest=p["interest"],
            )
            for _ in range(p["num_invoices"])
        ]

    def reset(self, difficulty=None, seed=None) -> CashflowmanagerObservation:
        if difficulty is not None:
            self.difficulty = difficulty
            self.params = DIFFICULTY_PRESETS[difficulty]

        if seed is not None:
            self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.max_days = self.params["max_days"]
        self.cash = self.params["starting_cash"]
        self.credit_used = 0
        self.day = 1

        self.daily_queue = self._generate_daily_invoices()
        self.daily_index = 0
        self.invoices = list(self.daily_queue)

        return self._build_obs(reward=0.0, done=False)

    def step(self, action: CashflowmanagerAction) -> CashflowmanagerObservation:
        self._state.step_count += 1

        late_fee_total = 0.0
        interest_total = 0.0
        paid = 0

        if self.daily_index >= len(self.daily_queue):
            return self._build_obs(reward=0.0, done=True)

        inv = self.daily_queue[self.daily_index]

        if action.type != 0:
            pay_amount = inv.min_payment if action.type == 1 else inv.amount
            credit_limit = self.params.get("credit_limit", 1000)
            available = self.cash + (credit_limit - self.credit_used)
            pay_amount = min(pay_amount, available)

            if pay_amount > self.cash:
                credit_used_now = pay_amount - self.cash
                self.credit_used += credit_used_now
                self.cash = 0
            else:
                self.cash -= pay_amount

            inv.amount -= pay_amount

            if inv.amount <= 0:
                inv.amount = 0
                paid += 1

        # remove fully paid invoices
        self.invoices = [i for i in self.invoices if i.amount > 0]

        # advance invoice index
        self.daily_index += 1
        done = False
        day_boundary = False

        # if all invoices for today handled → advance to next day
        if self.daily_index >= len(self.daily_queue):
            day_boundary = True
            self.day += 1

            # --- Age ALL active invoices ONCE per day ---
            for active_inv in self.invoices:
                active_inv.due_in -= 1

                if active_inv.due_in < 0 and active_inv.amount > 0:
                    late_fee_total += active_inv.late_fee
                    active_inv.amount += active_inv.late_fee

                if active_inv.amount > 0:
                    interest = active_inv.amount * active_inv.interest
                    interest_total += interest
                    active_inv.amount += interest

            # Remove invoices that became zero after aging
            self.invoices = [i for i in self.invoices if i.amount > 0]

            if self.day > self.max_days:
                done = True
                #penalize invoices that are actually overdue
                for inv in self.invoices:
                    if inv.amount > 0 and inv.due_in < 0:
                        late_fee_total += inv.late_fee
                        self.cash -= inv.late_fee
            else:
                #generate fresh invoices for next day
                new_invoices = self._generate_daily_invoices()
                carryovers = [i for i in self.daily_queue if i.amount > 0]
                self.daily_queue = new_invoices + carryovers
                self.daily_index = 0
                self.invoices = list(self.daily_queue)
        
        if self.cash < 0:
            self.credit_used += abs(self.cash)
            self.cash = 0.0

        # Projected penalties for reward signal on intra-day steps
        reward_late = late_fee_total
        reward_interest = interest_total
        if not day_boundary and not done:
            queue_len = max(len(self.daily_queue), 1)
            for active_inv in self.invoices:
                if active_inv.due_in <= 0 and active_inv.amount > 0:
                    reward_late += active_inv.late_fee / queue_len
                if active_inv.amount > 0:
                    reward_interest += (active_inv.amount * active_inv.interest) / queue_len

        # reward (uses projected penalties; metadata keeps actuals for grading)
        reward = compute_reward(
            cash=self.cash,
            late_fee=reward_late,
            interest=reward_interest,
            credit_used=self.credit_used,
            paid=paid,
        )

        return self._build_obs(
            reward=reward,
            done=done,
            late_fee=late_fee_total,
            interest=interest_total,
        )

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
                "daily_index": self.daily_index,
                "invoices_today": len(self.daily_queue),
            },
        )

    @property
    def state(self) -> State:
        return self._state