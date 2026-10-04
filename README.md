# Invoice Triage Agent

An autonomous AI agent that investigates an invoice and decides what happens to it — built using Claude's tool-use (function-calling) capability.

## What makes this an agent, not just an app

Unlike a single-pass AI pipeline, this agent:
- Decides which tools to call and in what order (not hardcoded)
- Loops between reasoning and tool calls until it has enough evidence
- Chooses one of three final actions autonomously: auto-approve, flag for review, or request re-extraction

## Tools available to the agent
- `check_duplicate_invoice` — catches duplicate payments
- `lookup_vendor_history` — flags unusual amounts vs. vendor's typical spend
- `flag_for_human_review` / `auto_approve_invoice` / `request_reextraction` — final actions

## Example run
On a test invoice with a missing field, a stale date, and an amount 79% above the vendor's average, the agent investigated both checks and autonomously chose to flag it for human review — explaining its reasoning in plain English.

## Running it
pip install -r requirements.txt
streamlit run agent_app.py

## Tech Stack
Claude (tool-use/function-calling), Streamlit, Python

## Part of a connected portfolio
Extends the AI-103 Document Analyzer project — closing the loop from "AI-augmented app" to true autonomous agent.
