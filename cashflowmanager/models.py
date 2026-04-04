# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Cashflowmanager Environment.

The cashflowmanager environment is a simple test environment that echoes back messages.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from openenv.core.env_server.types import Action, Observation

class Invoice(BaseModel):
    amount: float
    due_in: int
    late_fee: float
    min_payment: float
    interest: float

class CashflowmanagerAction(Action):
    type: int = Field(..., description="0 skip, 1 min, 2 full")
    invoice_id: int = Field(..., description="Invoice ID to pay")

class CashflowmanagerObservation(Observation):
    day: int
    cash: float
    credit_used: float
    invoices: List[Invoice]
    reward: float = 0.0
    done: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)