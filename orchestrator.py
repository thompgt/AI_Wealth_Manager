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
from typing import Dict, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.diagnostics import diagnostics_node
from agents.finance_report import finance_report_node
from agents.market_regime import market_regime_node
from agents.stock_research import stock_research_node
from agents.suitability import suitability_node
from agents.tax_awareness import tax_awareness_node
from db import ClientProfile as ClientProfileModel
from db import Holding, SessionLocal
from services.market_data import get_current_prices
from state import AgentState
from state import ClientProfile as ClientProfileState

MAX_RESEARCH_ATTEMPTS = 3
LOW_CONFIDENCE_THRESHOLD = 0.3


def guardrail_gate_node(state: AgentState) -> dict:
    """
    Fan-in point for suitability + tax_awareness. Decides whether Stock
    Research needs another pass (suitability rejected everything and we
    haven't hit the retry cap) or whether the run can proceed to the
    Finance Report agent.
    """
    suitability = state.get("suitability_result") or {}
    approved = suitability.get("approved", False)
    attempts = state.get("research_attempts", 0)

    if approved or attempts >= MAX_RESEARCH_ATTEMPTS:
        return {"needs_research_retry": False}
    return {"research_attempts": attempts + 1, "needs_research_retry": True}


def route_after_guardrails(state: AgentState) -> Literal["retry", "proceed"]:
    return "retry" if state.get("needs_research_retry") else "proceed"


def approval_gate_node(state: AgentState) -> dict:
    """
    Human-in-the-loop gate. Triggers a real LangGraph interrupt() when:
      - Market Regime's confidence is below LOW_CONFIDENCE_THRESHOLD, or
      - Suitability never reached approval even after MAX_RESEARCH_ATTEMPTS
    Otherwise the run is auto-approved (informational report, no trade
    execution in this phase) and passes straight through.
    """
    regime = state.get("market_regime") or {}
    suitability = state.get("suitability_result") or {}

    low_confidence = regime.get("confidence", 0.0) < LOW_CONFIDENCE_THRESHOLD
    guardrail_exhausted = not suitability.get("approved", False)

    if not (low_confidence or guardrail_exhausted):
        return {"requires_human_approval": False, "human_approved": True}

    reason = "low_confidence_regime" if low_confidence else "suitability_exhausted_retries"
    decision = interrupt(
        {
            "reason": reason,
            "market_regime": regime,
            "suitability_result": suitability,
            "candidate_stocks": state.get("candidate_stocks"),
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


_checkpointer = MemorySaver()
app_workflow = _build_graph().compile(checkpointer=_checkpointer)


def load_client_state(client_id: int) -> AgentState:
    """
    Client Profile Intake -> initial AgentState. Resolves each Holding's
    quantity into a dollar market value (CASH is already dollar-valued;
    everything else is quantity * current price via services.market_data).
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
        for h in holding_rows:
            if h.symbol == "CASH":
                holdings["CASH"] = holdings.get("CASH", 0.0) + h.quantity
            else:
                price = prices.get(h.symbol, 0.0)
                holdings[h.symbol] = holdings.get(h.symbol, 0.0) + h.quantity * price

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
    result = app_workflow.invoke(initial_state, config=config)
    return {"run_id": initial_state["run_id"], "result": result}


def resume_client_graph(run_id: str, approved: bool) -> dict:
    """
    Resumes a run paused at the approval gate.
    """
    config = {"configurable": {"thread_id": run_id}}
    result = app_workflow.invoke(Command(resume=approved), config=config)
    return {"run_id": run_id, "result": result}


if __name__ == "__main__":
    import json

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
