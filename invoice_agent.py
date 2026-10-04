"""
invoice_agent.py
An autonomous invoice triage agent built on Claude's tool-use (function-calling)
capability. Unlike a single-pass "AI-augmented app," this agent:

  1. Reasons over extracted invoice data
  2. Autonomously decides which tools to call, in what order, and how many times
  3. Loops until it reaches a final decision
  4. Takes one of several possible actions based on what it discovers

This is the key difference from a traditional pipeline: the control flow is
decided by the model at run time, not hardcoded by the developer.

Setup:
    pip install anthropic python-dotenv

Environment variables required (.env file):
    ANTHROPIC_API_KEY=<your-key>
"""

import os
import json
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Mock "systems" the agent's tools query. In a real deployment these would be
# calls to a database, ERP system, or audit log.
# ---------------------------------------------------------------------------

PROCESSED_INVOICES_DB = {
    "INV-100": {"vendor": "Contoso Ltd.", "amount": 610.00, "processed_on": "2024-03-01"},
    "INV-205": {"vendor": "Acme Corp", "amount": 1200.00, "processed_on": "2024-02-15"},
}

VENDOR_HISTORY_DB = {
    "Contoso Ltd.": {"avg_invoice_amount": 340.00, "invoice_count": 12},
    "Acme Corp": {"avg_invoice_amount": 1150.00, "invoice_count": 8},
}

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def check_duplicate_invoice(invoice_number: str) -> dict:
    if invoice_number in PROCESSED_INVOICES_DB:
        return {"duplicate": True, "details": PROCESSED_INVOICES_DB[invoice_number]}
    return {"duplicate": False}


def lookup_vendor_history(vendor_name: str) -> dict:
    if vendor_name in VENDOR_HISTORY_DB:
        return VENDOR_HISTORY_DB[vendor_name]
    return {"avg_invoice_amount": None, "invoice_count": 0, "note": "No prior history for this vendor."}


def flag_for_human_review(reason: str) -> dict:
    return {"action_taken": "FLAGGED_FOR_REVIEW", "reason": reason}


def auto_approve_invoice(invoice_id: str, justification: str) -> dict:
    return {"action_taken": "AUTO_APPROVED", "invoice_id": invoice_id, "justification": justification}


def request_reextraction(reason: str) -> dict:
    return {"action_taken": "REEXTRACTION_REQUESTED", "reason": reason}


TOOL_FUNCTIONS = {
    "check_duplicate_invoice": check_duplicate_invoice,
    "lookup_vendor_history": lookup_vendor_history,
    "flag_for_human_review": flag_for_human_review,
    "auto_approve_invoice": auto_approve_invoice,
    "request_reextraction": request_reextraction,
}

# ---------------------------------------------------------------------------
# Tool schema (what Claude sees and chooses from)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "check_duplicate_invoice",
        "description": "Check whether this invoice number has already been processed, to catch duplicate payments.",
        "input_schema": {
            "type": "object",
            "properties": {"invoice_number": {"type": "string"}},
            "required": ["invoice_number"],
        },
    },
    {
        "name": "lookup_vendor_history",
        "description": "Look up this vendor's typical invoice amount and history, to judge whether the current invoice amount is unusual.",
        "input_schema": {
            "type": "object",
            "properties": {"vendor_name": {"type": "string"}},
            "required": ["vendor_name"],
        },
    },
    {
        "name": "flag_for_human_review",
        "description": "Escalate this invoice to a human reviewer because something is uncertain, risky, or anomalous enough that an autonomous decision isn't appropriate.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Clear explanation of why this needs human review."}},
            "required": ["reason"],
        },
    },
    {
        "name": "auto_approve_invoice",
        "description": "Approve the invoice automatically because it is low-risk: no duplicates, amount consistent with vendor history, high extraction confidence, no missing fields.",
        "input_schema": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string"},
                "justification": {"type": "string", "description": "Why this was safe to auto-approve."},
            },
            "required": ["invoice_id", "justification"],
        },
    },
    {
        "name": "request_reextraction",
        "description": "Request the document be re-extracted because the data quality is too poor (very low confidence scores, critical missing fields) to make any reliable decision.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]

SYSTEM_PROMPT = """You are an autonomous invoice triage agent for an accounts payable team.

You will be given structured data extracted from an invoice (fields, values, and
confidence scores from an OCR/extraction system).

Your job is to investigate and decide what should happen to this invoice. You have
tools available to check for duplicates, look up vendor history, and take a final
action. You decide which tools to use, in what order, and how many times — think
like an investigator, not a checklist.

Guidelines:
- Always check for duplicate invoices before approving anything.
- If the vendor has prior history, compare the current amount against it. A
  significant deviation (e.g., more than ~50% above average) is worth flagging.
- Low confidence scores (below 0.75) on critical fields (vendor, total, invoice
  number) should push you toward caution — flag for review or request
  re-extraction rather than guessing.
- Only auto-approve when you have positive evidence of low risk: no duplicate,
  amount consistent with history, high confidence on key fields.
- You MUST end by calling exactly one final action tool: flag_for_human_review,
  auto_approve_invoice, or request_reextraction. Investigation tools
  (check_duplicate_invoice, lookup_vendor_history) do not count as a final decision.
- After taking your final action, write one short paragraph summarizing your
  reasoning and the full chain of what you checked and why.
"""


def run_agent(extracted_invoice: dict, verbose: bool = True) -> dict:
    """
    Runs the full agentic loop: Claude reasons, calls tools, receives results,
    and continues until it takes a final action. Returns the full trace plus
    the final decision.
    """
    messages = [
        {
            "role": "user",
            "content": f"Here is the extracted invoice data:\n\n{json.dumps(extracted_invoice, indent=2, default=str)}\n\nInvestigate and decide what should happen to this invoice.",
        }
    ]

    trace = []
    final_action = None
    max_turns = 8  # safety limit so a misbehaving loop can't run forever

    for turn in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Log any reasoning text the model produced this turn
        for block in response.content:
            if block.type == "text" and block.text.strip():
                trace.append({"type": "reasoning", "text": block.text.strip()})
                if verbose:
                    print(f"\n[Turn {turn+1}] Agent reasoning:\n{block.text.strip()}")

        if response.stop_reason != "tool_use":
            # Agent finished without calling a tool this turn — done.
            break

        # Handle every tool call in this turn
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                trace.append({"type": "tool_call", "tool": tool_name, "input": tool_input})
                if verbose:
                    print(f"[Turn {turn+1}] Agent calls tool: {tool_name}({tool_input})")

                func = TOOL_FUNCTIONS.get(tool_name)
                result = func(**tool_input) if func else {"error": f"Unknown tool {tool_name}"}
                trace.append({"type": "tool_result", "tool": tool_name, "result": result})
                if verbose:
                    print(f"[Turn {turn+1}] Tool result: {result}")

                if tool_name in ("flag_for_human_review", "auto_approve_invoice", "request_reextraction"):
                    final_action = {"tool": tool_name, "input": tool_input, "result": result}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        if final_action:
            break

    return {"trace": trace, "final_action": final_action}


if __name__ == "__main__":
    # Example: reuse the extracted fields from your Azure Document Intelligence project
    sample_invoice = {
        "file_name": "sample-invoice.pdf",
        "fields": {
            "InvoiceId": {"value": "INV-100", "confidence": 0.98},
            "VendorName": {"value": "Contoso Ltd.", "confidence": 0.94},
            "InvoiceTotal": {"value": 610.00, "confidence": 0.91},
            "InvoiceDate": {"value": "2019-06-18", "confidence": 0.85},
            "Items": {"value": None, "confidence": None},
            "TaxDetails": {"value": None, "confidence": None},
        },
    }

    result = run_agent(sample_invoice)

    print("\n" + "=" * 50)
    print("FINAL DECISION")
    print("=" * 50)
    print(json.dumps(result["final_action"], indent=2, default=str))
