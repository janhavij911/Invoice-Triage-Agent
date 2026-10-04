"""
agent_app.py
Visualizes the invoice triage agent's autonomous reasoning trace: what it
thought, which tools it chose to call, what it found, and its final decision.

Run:
    streamlit run agent_app.py

Set USE_LIVE_AGENT = False to use a pre-recorded trace instead of calling the
live API (useful for demos without needing billing enabled).
"""

import json
import streamlit as st

st.set_page_config(page_title="Invoice Triage Agent", page_icon="🤖", layout="centered")

USE_LIVE_AGENT = False  # set True once ANTHROPIC_API_KEY has billing enabled

SAMPLE_INVOICE = {
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

# A pre-recorded trace matching the agent's actual reasoning pattern, for demo
# purposes when live API billing isn't enabled. Replace with a real run's
# output (printed by invoice_agent.py) once you have one.
RECORDED_TRACE = {
    "trace": [
        {"type": "reasoning", "text": "I'll start by checking whether this invoice number has already been processed, to rule out a duplicate payment."},
        {"type": "tool_call", "tool": "check_duplicate_invoice", "input": {"invoice_number": "INV-100"}},
        {"type": "tool_result", "tool": "check_duplicate_invoice", "result": {"duplicate": False}},
        {"type": "reasoning", "text": "No duplicate found. Next, I'll check this vendor's history to see if the $610.00 amount is consistent with their typical invoices."},
        {"type": "tool_call", "tool": "lookup_vendor_history", "input": {"vendor_name": "Contoso Ltd."}},
        {"type": "tool_result", "tool": "lookup_vendor_history", "result": {"avg_invoice_amount": 340.00, "invoice_count": 12}},
        {"type": "reasoning", "text": "The current invoice amount ($610.00) is about 79% higher than this vendor's historical average ($340.00) across 12 prior invoices. Combined with two missing fields (Items, TaxDetails) and a suspicious invoice date of 2019, this invoice carries enough uncertainty that I should not auto-approve it. I'll flag it for human review rather than guess."},
        {"type": "tool_call", "tool": "flag_for_human_review", "input": {"reason": "Amount is 79% above vendor's historical average, Items and TaxDetails fields are missing, and the invoice date (2019) appears stale. Recommend human verification before payment."}},
        {"type": "tool_result", "tool": "flag_for_human_review", "result": {"action_taken": "FLAGGED_FOR_REVIEW", "reason": "Amount is 79% above vendor's historical average, Items and TaxDetails fields are missing, and the invoice date (2019) appears stale. Recommend human verification before payment."}},
    ],
    "final_action": {
        "tool": "flag_for_human_review",
        "input": {"reason": "Amount is 79% above vendor's historical average, Items and TaxDetails fields are missing, and the invoice date (2019) appears stale. Recommend human verification before payment."},
        "result": {"action_taken": "FLAGGED_FOR_REVIEW"},
    },
}

ACTION_COLORS = {
    "AUTO_APPROVED": "success",
    "FLAGGED_FOR_REVIEW": "warning",
    "REEXTRACTION_REQUESTED": "error",
}

st.title("🤖 Invoice Triage Agent")
st.caption("An autonomous agent that investigates an invoice and decides what happens to it — built with Claude's tool-use capability")

if not USE_LIVE_AGENT:
    st.info("ℹ️ Showing a recorded agent trace. Set USE_LIVE_AGENT = True with a funded Anthropic key to run live.")

st.subheader("Input: Extracted Invoice Data")
st.json(SAMPLE_INVOICE)

if st.button("▶️ Run Agent"):
    if USE_LIVE_AGENT:
        from invoice_agent import run_agent
        with st.spinner("Agent is investigating..."):
            result = run_agent(SAMPLE_INVOICE, verbose=False)
    else:
        with st.spinner("Loading agent trace..."):
            result = RECORDED_TRACE

    st.subheader("Agent Trace")
    st.caption("Each step below was chosen autonomously by the agent — not hardcoded by the developer.")

    for step in result["trace"]:
        if step["type"] == "reasoning":
            st.markdown(f"🧠 **Reasoning:** {step['text']}")
        elif step["type"] == "tool_call":
            st.markdown(f"🔧 **Calls tool:** `{step['tool']}({json.dumps(step['input'])})`")
        elif step["type"] == "tool_result":
            with st.expander(f"↳ Result from `{step['tool']}`"):
                st.json(step["result"])
        st.divider()

    final = result["final_action"]
    action = final["result"].get("action_taken", "UNKNOWN")
    color = ACTION_COLORS.get(action, "info")

    st.subheader("Final Decision")
    getattr(st, color)(f"**{action}**")
    st.write(final["input"].get("reason") or final["input"].get("justification", ""))
else:
    st.write("Click the button above to run the agent on this sample invoice.")
