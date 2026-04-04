import os
import json
from dotenv import load_dotenv
from models import CashflowmanagerAction

try:
    from groq import Groq
except ImportError:
    pass

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def groq_policy(s, history=None):
    """
    Intelligent Groq Decision Policy with History.
    Decides whether to pay invoices (0: Skip, 1: Min, 2: Full).
    """
    # 1. Format the current state for the LLM
    invoices_str = ""
    for i, inv in enumerate(s.invoices):
        invoices_str += f"ID {i}: Amount {inv.amount:.2f}, Due in {inv.due_in} days, Late Fee {inv.late_fee}, Interest {inv.interest*100}%\n"

    history_str = "No previous history."
    if history:
        history_str = ""
        for h in history:
            history_str += f"Day {h['day']}: Action {h['action']} on Invoice {h['invoice_id']}, Reward: {h['reward']:.2f}, Late Fee: {h['late_fee']:.2f}, Interest: {h['interest']:.2f}\n"

    prompt = f"""
    You are a Cashflow Management AI. Your goal is to maximize reward by minimizing penalties (late fees/interest) and maintaining liquidity.
    
    Current State:
    Day: {s.day}
    Cash: {s.cash:.2f}
    Credit Used: {s.credit_used:.2f}
    
    Invoices:
    {invoices_str}

    Previous Action History & Feedback:
    {history_str}
    
    Action Definition:
    - type=0: No action (Skip)
    - type=1: Pay Minimum on an invoice
    - type=2: Pay Full on an invoice
    - invoice_id: The ID of the invoice (0-2)
    
    Strategic Note: Compare your previous actions and the resulting penalties/rewards. If an action led to a large negative reward (e.g. high late fees), adjust your strategy for the current state.

    You MUST output a valid JSON object with {{"type": <int>, "invoice_id": <int>}}.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
        )
        data = json.loads(chat_completion.choices[0].message.content)
        return CashflowmanagerAction(type=data["type"], invoice_id=data["invoice_id"])
    except Exception as e:
        print(f"Policy Error: {e}")
        return CashflowmanagerAction(type=0, invoice_id=0)
