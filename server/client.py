import os
import json
from dotenv import load_dotenv
from models import CashflowmanagerAction

try:
    from openai import OpenAI
except ImportError:
    pass

load_dotenv()

# Mandatory OpenAI Client Configuration (Disqualification Compliance)
API_BASE_URL = os.environ.get("API_BASE_URL") or "https://api.groq.com/openai/v1"
MODEL_NAME = os.environ.get("MODEL_NAME") or "llama-3.1-8b-instant"
# Prioritize HF_TOKEN as per mandatory requirements
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def groq_policy(s, history=None):
    """
    Intelligent Decision Policy with History using the Modern OpenAI Responses API.
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

    instructions = """
    You are a Cashflow Management AI. Your goal is to maximize reward by minimizing penalties (late fees/interest) and maintaining liquidity.
    
    Action Definition:
    - type=0: No action (Skip)
    - type=1: Pay Minimum on an invoice
    - type=2: Pay Full on an invoice
    - invoice_id: The ID of the invoice (0-4)
    
    Strategic Note: Compare your previous actions and the resulting penalties/rewards. If an action led to a large negative reward (e.g. high late fees), adjust your strategy for the current state.

    You MUST output a valid JSON object with {"type": <int>, "invoice_id": <int>}.
    Only return the JSON object, nothing else.
    """

    state_input = f"""
    Current State:
    Day: {s.day}
    Cash: {s.cash:.2f}
    Credit Used: {s.credit_used:.2f}
    
    Invoices:
    {invoices_str}

    Previous Action History & Feedback:
    {history_str}
    """

    try:
        # Using the newer, efficient Responses API as requested
        response = client.responses.create(
            model=MODEL_NAME,
            instructions=instructions,
            input=state_input
        )
        
        # Access the text output and parse JSON
        content = response.output_text.strip()
        # Handle cases where the model might wrap JSON in backticks
        if content.startswith("```json"):
            content = content.replace("```json", "", 1).replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "", 1).replace("```", "").strip()
            
        data = json.loads(content)
        return CashflowmanagerAction(type=data["type"], invoice_id=data["invoice_id"])
    except Exception as e:
        print(f"Policy Error (Responses API): {e}")
        # Fallback to a safe default action
        return CashflowmanagerAction(type=0, invoice_id=0)
