"""
Unified LangGraph orchestrator, replacing both the old workflow.py
(macro_sentinel -> quant_builder -> tax_architect -> compliance_critic)
and backend/app/agents/agent_workflow.py's single LangChain tool-agent.

Flow (see PLAN.md's Orchestration flow diagram):

    START --> diagnostics \\
    START --> market_regime > (parallel, fan-in) --> stock_research
                                                            |
                                              (fan-out, parallel)
                                                    /              \\
                                            suitability        tax_awareness
                                                    \\              /
                                                  (fan-in) --> guardrail_gate
                                                                    |
                                        [retry -> stock_research] <-+-> [proceed -> finance_report]
                                                                          |
                                                                    approval_gate
                                                                          |
                                                                         END

guardrail_gate and approval_gate are lightweight LangGraph nodes: the
former is the fan-in point for the retry-loop decision, the latter is
where the human-in-the-loop interrupt lives.

DB persistence of the audit trail / final report (the agent_runs and
reports tables) is NOT done here -- this module owns graph control flow
only. The caller (server.py) is responsible for persisting the returned
AgentState after a run completes.
"""

import uuid
from typing import Dict, List, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.diagnostics import diagnostics_node
from agents.finance_report import finance_report_node
from agents.market_regime import market_regime_node
from agents.stock_research import stock_research_node
from agents.suitability import suitability_node
from agents.tax_awareness import tax_awareness_node
from checkpointer import get_checkpointer
from db import ClientProfile as ClientProfileModel
from db import Holding, SessionLocal
from logging_setup import get_logger
from services.market_data import get_current_prices
from state import AgentState
from state import ClientProfile as ClientProfileState

logger = get_logger(__name__)

MAX_RESEARCH_ATTEMPTS = 3
LOW_CONFIDENCE_THRESHOLD = 0.3


def guardrail_gate_node(state: AgentState) -> dict:
    """
    Fan-in point for suitability + tax_awareness, and the single place where
    the two guardrails are reconciled into one decision.

    Three jobs:

    1.  ENFORCE THE TAX ASSESSMENT. The Tax-Awareness agent runs in parallel
        with Suitability, so Suitability cannot see its wash-sale flags. Any
        recommendation that cleared suitability but is wash-sale flagged is
        stripped here. Previously nothing in the system consumed
        `wash_sale_flags` at all: the agent computed them, wrote them to
        state, the report printed them -- and the flagged ticker was still
        recommended with a dollar figure next to it. That is precisely the
        violation the check exists to prevent.

    2.  ACCUMULATE RETRY FEEDBACK. Every ticker rejected by either guardrail
        is added to `excluded_tickers`, and the human-readable reason to
        `guardrail_feedback`. Stock Research consumes both on its next pass.
        Without this the retry edge is decorative -- Stock Research would
        screen the same universe with the same inputs and return the same
        rejected names, three times over.

    3.  ROUTE. Retry while we still have budget and nothing has been
        approved; otherwise proceed to the report.

    Note on sizing after a tax block: the removed candidate's dollars are NOT
    silently reassigned to the survivors. Re-sizing is Suitability's job and
    only it knows the position caps, so a blocked ticker either gets replaced
    on a retry pass (which re-runs sizing properly) or the run simply deploys
    less cash. Under-deploying is the safe direction to fail.
    """
    suitability = dict(state.get("suitability_result") or {})
    tax = state.get("tax_assessment") or {}
    attempts = state.get("research_attempts", 0)

    excluded: List[str] = list(state.get("excluded_tickers") or [])
    feedback: List[str] = list(state.get("guardrail_feedback") or [])

    recommendations = list(suitability.get("adjusted_recommendations") or [])
    wash_sale_flags = set(tax.get("wash_sale_flags") or [])

    # --- 1. Tax enforcement ------------------------------------------------
    surviving = []
    tax_blocked: List[str] = []
    for rec in recommendations:
        ticker = rec.get("ticker")
        if ticker in wash_sale_flags:
            tax_blocked.append(ticker)
            reason = (
                f"{ticker}: blocked by the Tax-Awareness agent -- repurchasing it now "
                f"would be a wash-sale violation, disallowing the loss on the recent sale."
            )
            logger.info("[Guardrail Gate] TAX-BLOCKED %s", ticker)
            suitability.setdefault("violations", [])
            suitability["violations"] = list(suitability["violations"]) + [reason]
            feedback.append(reason)
        else:
            surviving.append(rec)

    suitability["adjusted_recommendations"] = surviving
    suitability["approved"] = len(surviving) > 0

    # --- 2. Accumulate exclusions for the retry pass -----------------------
    for ticker in tax_blocked:
        if ticker not in excluded:
            excluded.append(ticker)
    for violation in state.get("suitability_result", {}).get("violations", []) or []:
        # Violation strings are formatted "TICKER: rejected -- ..." by the
        # suitability agent; the leading token is the ticker to exclude.
        ticker = violation.split(":", 1)[0].strip()
        if ticker and " " not in ticker and ticker not in excluded:
            excluded.append(ticker)
        if violation not in feedback:
            feedback.append(violation)

    approved = suitability["approved"]

    # --- 3. Route ----------------------------------------------------------
    update = {
        "suitability_result": suitability,
        "tax_blocked_recommendations": tax_blocked,
        "excluded_tickers": excluded,
        "guardrail_feedback": feedback,
    }

    if approved or attempts >= MAX_RESEARCH_ATTEMPTS:
        if not approved:
            logger.warning(
                "[Guardrail Gate] No candidate survived after %d research attempt(s); "
                "proceeding to report with zero recommendations.", attempts,
            )
        update["needs_research_retry"] = False
        return update

    logger.info(
        "[Guardrail Gate] Nothing approved; sending Stock Research back for attempt %d/%d, "
        "excluding %s",
        attempts + 1, MAX_RESEARCH_ATTEMPTS, excluded,
    )
    update["research_attempts"] = attempts + 1
    update["needs_research_retry"] = True
    return update


def route_after_guardrails(state: AgentState) -> Literal["retry", "proceed"]:
    return "retry" if state.get("needs_research_retry") else "proceed"


def approval_gate_node(state: AgentState) -> dict:
    """
    Human-in-the-loop gate. Triggers a real LangGraph interrupt() when:
      - Market Regime's confidence is below LOW_CONFIDENCE_THRESHOLD, or
      - No candidate survived the guardrails
    Otherwise the run is auto-approved (informational report, no trade
    execution in this phase) and passes straight through.
    """
    regime = state.get("market_regime") or {}
    suitability = state.get("suitability_result") or {}
    attempts = state.get("research_attempts", 0)

    low_confidence = regime.get("confidence", 0.0) < LOW_CONFIDENCE_THRESHOLD
    nothing_approved = not suitability.get("approved", False)

    if not (low_confidence or nothing_approved):
        return {"requires_human_approval": False, "human_approved": True}

    # Report the reason accurately: "exhausted retries" is only true if we
    # actually spent them. Reporting a first-pass rejection as an exhausted
    # retry budget misleads whoever is reviewing the paused run.
    if low_confidence:
        reason = "low_confidence_regime"
    elif attempts >= MAX_RESEARCH_ATTEMPTS:
        reason = "suitability_exhausted_retries"
    else:
        reason = "no_suitable_candidates"

    logger.info("[Approval Gate] Pausing run for human approval: %s", reason)
    decision = interrupt(
        {
            "reason": reason,
            "market_regime": regime,
            "suitability_result": suitability,
            "candidate_stocks": state.get("candidate_stocks"),
            "tax_assessment": state.get("tax_assessment"),
            "research_attempts": attempts,
        }
    )
    return {"requires_human_approval": True, "human_approved": bool(decision)}


def _build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("diagnostics", diagnostics_node)
    workflow.add_node("market_regime", market_regime_node)
    workflow.add_node("stock_research", stock_research_node)
    workflow.add_node("suitability", suitability_node)
    workflow.add_node("tax_awareness", tax_awareness_node)
    workflow.add_node("guardrail_gate", guardrail_gate_node)
    workflow.add_node("finance_report", finance_report_node)
    workflow.add_node("approval_gate", approval_gate_node)

    # Fan-out: diagnostics and market_regime have no data dependency on
    # each other, both run off the client profile / market data alone.
    workflow.add_edge(START, "diagnostics")
    workflow.add_edge(START, "market_regime")

    # Fan-in: stock_research needs both.
    workflow.add_edge("diagnostics", "stock_research")
    workflow.add_edge("market_regime", "stock_research")

    # Fan-out: suitability and tax_awareness both only consume candidate_stocks.
    workflow.add_edge("stock_research", "suitability")
    workflow.add_edge("stock_research", "tax_awareness")

    # Fan-in at the guardrail gate.
    workflow.add_edge("suitability", "guardrail_gate")
    workflow.add_edge("tax_awareness", "guardrail_gate")

    workflow.add_conditional_edges(
        "guardrail_gate",
        route_after_guardrails,
        {"retry": "stock_research", "proceed": "finance_report"},
    )

    workflow.add_edge("finance_report", "approval_gate")
    workflow.add_edge("approval_gate", END)

    return workflow


_compiled_workflow = None


def get_workflow():
    """Compile the graph lazily against the durable checkpointer.

    Lazy rather than at import time so that tests (and the demo notebook) can
    point CHECKPOINT_DB_PATH at a temp location before the first run, and so
    that merely importing the orchestrator doesn't create files on disk.
    """
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = _build_graph().compile(checkpointer=get_checkpointer())
    return _compiled_workflow


def reset_workflow() -> None:
    """Drop the compiled graph so the next call rebinds to a fresh checkpointer."""
    global _compiled_workflow
    _compiled_workflow = None


def load_client_state(client_id: int) -> AgentState:
    """
    Client Profile Intake -> initial AgentState. Resolves each Holding's
    quantity into a dollar market value (CASH is already dollar-valued;
    everything else is quantity * current price via services.market_data).

    Symbols with no resolvable price are dropped from the holdings map and
    logged, rather than being silently valued at $0 -- a $0 valuation would
    quietly distort every concentration percentage and flaw downstream.
    """
    db = SessionLocal()
    try:
        client = db.query(ClientProfileModel).filter(ClientProfileModel.id == client_id).first()
        if not client:
            raise ValueError(f"No client profile with id {client_id}")

        holding_rows = db.query(Holding).filter(Holding.client_id == client_id).all()
        priced_symbols = [h.symbol for h in holding_rows if h.symbol != "CASH"]
        prices = get_current_prices(priced_symbols) if priced_symbols else {}

        holdings: Dict[str, float] = {}
        unpriced: List[str] = []
        for h in holding_rows:
            if h.symbol == "CASH":
                holdings["CASH"] = holdings.get("CASH", 0.0) + h.quantity
            else:
                price = prices.get(h.symbol)
                if price is None or price <= 0:
                    unpriced.append(h.symbol)
                    continue
                holdings[h.symbol] = holdings.get(h.symbol, 0.0) + h.quantity * price

        if unpriced:
            logger.warning(
                "Client %s holds %s but no current price could be resolved; these positions "
                "are excluded from this run's analysis rather than valued at $0.",
                client_id, unpriced,
            )

        client_profile: ClientProfileState = {
            "client_id": client.id,
            "age": client.age,
            "risk_tolerance": client.risk_tolerance,
            "time_horizon_years": client.time_horizon_years,
            "goals": list(client.goals or []),
            "holdings": holdings,
            "net_worth": client.net_worth,
        }

        initial_state: AgentState = {
            "run_id": str(uuid.uuid4()),
            "client_profile": client_profile,
            "portfolio_diagnostics": None,
            "market_regime": None,
            "candidate_stocks": None,
            "suitability_result": None,
            "tax_assessment": None,
            "final_report": None,
            "audit_trail": [],
            "requires_human_approval": False,
            "human_approved": None,
            "research_attempts": 0,
            "needs_research_retry": False,
            "excluded_tickers": [],
            "guardrail_feedback": [],
            "tax_blocked_recommendations": [],
        }
        return initial_state
    finally:
        db.close()


def run_client_graph(client_id: int) -> dict:
    """
    Starts a fresh graph run for a client. Returns the LangGraph invoke()
    result -- either the completed final state, or a dict containing an
    "__interrupt__" key if the approval gate paused the run.
    """
    initial_state = load_client_state(client_id)
    config = {"configurable": {"thread_id": initial_state["run_id"]}}
    logger.info("Starting graph run %s for client %s", initial_state["run_id"], client_id)
    result = get_workflow().invoke(initial_state, config=config)
    return {"run_id": initial_state["run_id"], "result": result}


def resume_client_graph(run_id: str, approved: bool) -> dict:
    """
    Resumes a run paused at the approval gate.
    """
    config = {"configurable": {"thread_id": run_id}}
    logger.info("Resuming run %s with approved=%s", run_id, approved)
    result = get_workflow().invoke(Command(resume=approved), config=config)
    return {"run_id": run_id, "result": result}


if __name__ == "__main__":
    import json

    from logging_setup import configure_logging

    configure_logging()

    print("=== Running graph for seeded client_id=1 ===")
    run = run_client_graph(1)
    print("run_id:", run["run_id"])
    result = run["result"]

    if "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        print("\n=== Graph paused at approval_gate (interrupt) ===")
        print(json.dumps(interrupt_obj.value, indent=2, default=str))
        print("\n=== Resuming with human_approved=True ===")
        resumed = resume_client_graph(run["run_id"], approved=True)
        result = resumed["result"]

    print("\n=== Final state keys ===")
    print(list(result.keys()))
    print("\n=== market_regime ===")
    print(json.dumps(result.get("market_regime"), indent=2, default=str))
    print("\n=== portfolio_diagnostics flaws ===")
    print(json.dumps((result.get("portfolio_diagnostics") or {}).get("flaws"), indent=2))
    print("\n=== suitability_result ===")
    print(json.dumps(result.get("suitability_result"), indent=2, default=str))
    print("\n=== tax_assessment ===")
    print(json.dumps(result.get("tax_assessment"), indent=2, default=str))
    print("\n=== research_attempts ===", result.get("research_attempts"))
    print("\n=== requires_human_approval / human_approved ===",
          result.get("requires_human_approval"), result.get("human_approved"))
    print("\n=== audit_trail node sequence ===")
    print([r["node_name"] for r in result.get("audit_trail", [])])
    print("\n=== final_report (first 800 chars) ===")
    print((result.get("final_report") or "")[:800])
