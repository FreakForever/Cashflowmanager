
import os
import json
from dotenv import load_dotenv
from models import CashflowmanagerAction

try:
    from openai import OpenAI
except ImportError:
    pass

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL") or "https://api.groq.com/openai/v1"
MODEL_NAME = os.environ.get("MODEL_NAME") or "llama-3.1-8b-instant"
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# Cache: stores {day: {invoice_index: action_type}}
_action_cache = {}

def groq_policy(obs, history=None):
    """
    Batched policy — one API call per day for ALL invoices.
    Cache is keyed by day so Groq is only called once per day,
    not once per invoice.
    """
    global _action_cache

    if not obs.invoices:
        return CashflowmanagerAction(type=0, invoice_id=0)

    daily_index = obs.metadata.get("daily_index", 0)
    invoices_today = obs.metadata.get("invoices_today", 1)
    current_idx = min(daily_index, len(obs.invoices) - 1)
    day_key = obs.day

    if daily_index == 0 or day_key not in _action_cache:
        _action_cache[day_key] = _fetch_all_actions(obs, history, invoices_today)

    cached = _action_cache[day_key].get(daily_index)
    if cached is not None:
        action_type = max(0, min(2, int(cached)))
    else:
        action_type = 0 

    inv = obs.invoices[current_idx]

    if action_type == 2 and obs.cash < inv.amount:
        action_type = 1 if obs.cash >= inv.min_payment else 0
    if action_type == 1 and obs.cash < inv.min_payment:
        action_type = 0

    return CashflowmanagerAction(type=action_type, invoice_id=current_idx)


def _fetch_all_actions(obs, history, invoices_today):
    """
    Single Groq call — asks for actions on ALL invoices for today.
    Returns dict: {invoice_index: action_type}
    """
    today_invoices = obs.invoices[:invoices_today]

    invoices_str = ""
    for i, inv in enumerate(today_invoices):
        urgency = "OVERDUE" if inv.due_in <= 0 else \
                  "URGENT"  if inv.due_in <= 2 else \
                  "SOON"    if inv.due_in <= 4 else "OK"
        invoices_str += (
            f"  [{i}] {urgency} | Amount: {inv.amount:.2f} | "
            f"Due in: {inv.due_in}d | Late Fee: {inv.late_fee} | "
            f"Min: {inv.min_payment} | Interest: {inv.interest*100:.1f}%/day\n"
        )

    history_str = "None yet."
    if history:
        history_str = "\n".join(
            f"  Day {h['day']}: {h['action']} → "
            f"Reward {h['reward']:.2f} | Late: {h['late_fee']:.0f} | Interest: {h['interest']:.2f}"
            for h in history[-3:]
        )

    total_full = sum(inv.amount for inv in today_invoices)
    total_min  = sum(inv.min_payment for inv in today_invoices)

    prompt = f"""You are a cashflow management AI. Decide actions for ALL {len(today_invoices)} invoices for today.

Day: {obs.day}
Cash available: {obs.cash:.2f}
Credit used so far: {obs.credit_used:.2f}

Today's invoices:
{invoices_str}

Cost if paying all full: {total_full:.2f} | Cost if paying all min: {total_min:.2f}
Available cash: {obs.cash:.2f}

Recent history:
{history_str}

RULES:
- Total payments must NOT exceed available cash ({obs.cash:.2f})
- Prioritize OVERDUE invoices first (they accumulate late fees every step)
- URGENT invoices (due in 1-2 days) should be paid if cash allows
- Skip only if cash is insufficient or invoice is not urgent
- Minimize late fees and interest, maximize reward

Respond ONLY with a JSON object in this exact format:
{{"actions": [{{"invoice_id": 0, "type": 0}}, {{"invoice_id": 1, "type": 2}}, ...]}}

Where type: 0=Skip, 1=Pay Min, 2=Pay Full
Include one entry per invoice [{', '.join(str(i) for i in range(len(today_invoices)))}]"""

    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=200,
            timeout=15,
        )
        raw = json.loads(resp.choices[0].message.content)

        if "actions" in raw:
            actions = raw["actions"]
            return {a["invoice_id"]: a["type"] for a in actions}
        else:
            return {int(k): v for k, v in raw.items() if str(k).isdigit()}

    except Exception as e:
        print(f"[Policy] Groq error: {e} — defaulting all to skip")
        return {}