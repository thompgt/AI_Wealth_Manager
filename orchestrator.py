"""The multi-agent graph and its control flow.

    START ─┬─> diagnostics ────┐
           └─> market_regime ──┴─> stock_research ─┬─> suitability ──┐
                                                   └─> tax_awareness ┴─> guardrail_gate
    guardrail_gate ─(retry)──> stock_research
    guardrail_gate ─(proceed)─> rebalance ─> finance_report ─> approval_gate ─> END

Two structural fixes over the previous version.

**The approval gate was inert.** It sat *after* the report, so the full report
was written -- and persisted -- before any human saw the run. Its only outgoing
edge was to END, so `human_approved=False` changed nothing: a rejected run
still produced and stored a client-facing report identical to an approved one.
The gate now sits before the report, and rejection is a real branch that
produces a report marked as rejected, carrying the reviewer's reason, and
suppresses the trade plan.

**The tax guardrail could be disabled by a database error.** The tax node used
to swallow every exception and return empty wash-sale flags, so a transient
failure silently turned the only hard tax control off while the run reported
success. The gate now treats an unverified tax assessment as blocking rather
than clean.

DB persistence is not done here -- this module owns control flow only. The
caller (the job worker) persists the returned state.
"""

import uuid
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.diagnostics import diagnostics_node
from agents.finance_report import finance_report_node
from agents.market_regime import market_regime_node
from agents.rebalance import rebalance_node
from agents.stock_research import stock_research_node
from agents.suitability import suitability_node
from agents.tax_awareness import tax_awareness_node
from checkpointer import get_checkpointer
from config import settings
from db import ClientProfile as ClientProfileModel
from db import SessionLocal, utcnow
from logging_setup import get_logger, log_context
from metrics import approvals_requested, guardrail_blocks, runs_started
from services.llm import run_budget
from services.market_data import quote_provenance
from services.policy import resolve as resolve_policy
from services.portfolio import load_portfolio
from state import AgentState, PolicySnapshot
from state import ClientProfile as ClientProfileState

logger = get_logger(__name__)

MAX_RESEARCH_ATTEMPTS = 3
# Below this, the regime call is not informative enough to justify acting on
# without a human looking at it.
LOW_CONFIDENCE_THRESHOLD = 0.3
# Any plan moving more than this share of the portfolio gets human review
# regardless of how confident everything else is. A large trade is a large
# trade.
LARGE_TRADE_FRACTION = 0.20


# --- Guardrail gate ----------------------------------------------------------


def guardrail_gate_node(state: AgentState) -> dict:
    """Reconcile the two guardrails into one decision, and route.

    Three jobs:

    1. **Enforce the tax assessment.** Tax-awareness runs in parallel with
       suitability, so suitability cannot see its wash-sale flags. Anything
       that cleared suitability but is flagged is stripped here. Previously
       nothing in the system consumed those flags: the agent computed them,
       the report printed them, and the flagged ticker was still recommended
       with a dollar figure beside it -- precisely the violation the check
       exists to prevent.

    2. **Accumulate retry feedback.** Every rejected ticker goes into
       `excluded_tickers` and its reason into `guardrail_feedback`, both read
       by research on the next pass. Without them the retry edge is
       decorative: research would screen the same universe with the same
       inputs and return the same rejects, three times.

    3. **Route.** Retry while budget remains and nothing is approved;
       otherwise proceed.

    Blocked dollars are deliberately *not* reassigned to the survivors.
    Re-sizing is suitability's job and only it knows the caps, so a blocked
    ticker is either replaced on a retry pass -- which re-runs sizing
    properly -- or the run simply deploys less. Under-deploying is the safe
    direction to fail.
    """
    suitability = dict(state.get("suitability_result") or {})
    tax = state.get("tax_assessment") or {}
    attempts = state.get("research_attempts", 0)

    excluded: List[str] = list(state.get("excluded_tickers") or [])
    feedback: List[str] = list(state.get("guardrail_feedback") or [])
    recommendations = list(suitability.get("adjusted_recommendations") or [])
    wash_sale_flags = {t.upper() for t in (tax.get("wash_sale_flags") or [])}
    reasons_by_ticker = {
        str(d.get("symbol", "")).upper(): d.get("reason", "")
        for d in (tax.get("wash_sale_detail") or [])
    }

    # --- 1. Tax enforcement --------------------------------------------------
    surviving = []
    tax_blocked: List[str] = []

    if tax.get("verification_failed"):
        # The tax check could not run. Treat that as blocking rather than
        # clean: an unverified position is not a verified-safe one, and the
        # previous fail-open behaviour meant a database blip silently disabled
        # this control while the run reported success.
        for recommendation in recommendations:
            ticker = str(recommendation.get("ticker", "")).upper()
            tax_blocked.append(ticker)
            reason = (
                f"{ticker}: withheld -- the tax assessment could not be completed, so this "
                f"purchase could not be checked against the wash-sale rule. Recommendations "
                f"are withheld rather than issued unverified."
            )
            feedback.append(reason)
            guardrail_blocks.labels("tax_unverified").inc()
        suitability["violations"] = list(suitability.get("violations") or []) + [
            "The tax assessment failed, so every recommendation in this run was withheld "
            "pending verification."
        ]
    else:
        for recommendation in recommendations:
            ticker = str(recommendation.get("ticker", "")).upper()
            if ticker in wash_sale_flags:
                tax_blocked.append(ticker)
                reason = reasons_by_ticker.get(ticker) or (
                    f"{ticker}: blocked -- repurchasing it now would trigger the wash-sale "
                    f"rule and disallow the loss on the recent sale."
                )
                logger.info("[Guardrail] tax-blocked %s", ticker)
                guardrail_blocks.labels("wash_sale").inc()
                suitability["violations"] = list(suitability.get("violations") or []) + [reason]
                feedback.append(reason)
            else:
                surviving.append(recommendation)

    suitability["adjusted_recommendations"] = surviving
    suitability["approved"] = len(surviving) > 0

    # --- 2. Accumulate exclusions for the retry pass -------------------------
    for ticker in tax_blocked:
        if ticker and ticker not in excluded:
            excluded.append(ticker)

    for violation in (state.get("suitability_result") or {}).get("violations", []) or []:
        # Violations are formatted "TICKER: rejected -- ...". Parsing the
        # leading token is what builds the exclusion list; it is guarded so a
        # message that does not match simply contributes no exclusion instead
        # of adding a garbage one, which is how this silently became a no-op
        # before.
        head = violation.split(":", 1)[0].strip()
        looks_like_ticker = (
            head
            and " " not in head
            and 1 <= len(head) <= 12
            and head.replace("-", "").replace(".", "").isalnum()
            and head.upper() == head
        )
        if looks_like_ticker and head not in excluded:
            excluded.append(head)
        if violation not in feedback:
            feedback.append(violation)

    approved = suitability["approved"]
    update: Dict[str, Any] = {
        "suitability_result": suitability,
        "tax_blocked_recommendations": tax_blocked,
        "excluded_tickers": excluded,
        "guardrail_feedback": feedback,
    }

    # --- 3. Route ------------------------------------------------------------
    if approved or attempts >= MAX_RESEARCH_ATTEMPTS:
        if not approved:
            logger.warning(
                "[Guardrail] nothing survived after %d research attempt(s); proceeding "
                "with zero recommendations.", attempts,
            )
        update["needs_research_retry"] = False
        return update

    logger.info(
        "[Guardrail] nothing approved; returning to research for attempt %d/%d, excluding %s",
        attempts + 1, MAX_RESEARCH_ATTEMPTS, excluded,
    )
    update["research_attempts"] = attempts + 1
    update["needs_research_retry"] = True
    return update


def route_after_guardrails(state: AgentState) -> Literal["retry", "proceed"]:
    return "retry" if state.get("needs_research_retry") else "proceed"


# --- Approval gate -----------------------------------------------------------


def _approval_reason(state: AgentState) -> Optional[str]:
    """Why this run needs a human, or None if it does not.

    Ordered by severity so the reason reported is the most serious one, not
    merely the first checked.
    """
    regime = state.get("market_regime") or {}
    suitability = state.get("suitability_result") or {}
    tax = state.get("tax_assessment") or {}
    plan = state.get("rebalance_plan") or {}
    diagnostics = state.get("portfolio_diagnostics") or {}
    degradations = state.get("degradations") or []
    attempts = state.get("research_attempts", 0)

    if tax.get("verification_failed"):
        return "tax_verification_failed"

    proposals = plan.get("proposals") or []
    total_value = diagnostics.get("total_value") or 0
    gross = plan.get("gross_notional") or 0
    if total_value > 0 and gross > LARGE_TRADE_FRACTION * total_value:
        return "large_trade"

    if any(p.get("side") == "SELL" for p in proposals):
        # A sale is irreversible and realizes tax. Buying with idle cash is a
        # different kind of decision from liquidating something the client
        # already owns, and only one of them should ever be automatic.
        return "sell_proposed"

    if (plan.get("estimated_tax_cost") or 0) > 0:
        return "tax_cost"

    if regime.get("confidence", 0.0) < LOW_CONFIDENCE_THRESHOLD:
        return "low_confidence_regime"

    if not suitability.get("approved", False):
        return "suitability_exhausted_retries" if attempts >= MAX_RESEARCH_ATTEMPTS else "no_suitable_candidates"

    # A run where several steps degraded may be perfectly reasonable, but a
    # human should decide that rather than the system deciding for them.
    if len({d.get("node") for d in degradations}) >= 2:
        return "multiple_degraded_steps"

    return None


APPROVAL_EXPLANATIONS = {
    "tax_verification_failed": "The tax assessment could not complete, so no recommendation "
                               "has been checked against the wash-sale rule.",
    "large_trade": "The proposed trades move a large share of the portfolio.",
    "sell_proposed": "The plan proposes selling existing positions, which is irreversible "
                     "and realizes tax.",
    "tax_cost": "The proposed trades would realize a taxable gain.",
    "low_confidence_regime": "The market regime assessment has low confidence, so the "
                             "recommendations rest on portfolio construction alone.",
    "no_suitable_candidates": "No candidate passed the guardrails.",
    "suitability_exhausted_retries": "No candidate passed the guardrails after every "
                                     "research attempt was exhausted.",
    "multiple_degraded_steps": "Several analysis steps ran with reduced information.",
}


def approval_gate_node(state: AgentState) -> dict:
    """Pause for a human when the run warrants it.

    Placed *before* the report, which is the fix that makes it mean anything.
    Previously it ran after, so the client-facing report was fully written and
    persisted before any human saw the run, and the decision had no outgoing
    edge other than END -- rejecting a run changed nothing at all.
    """
    reason = _approval_reason(state)
    if reason is None:
        return {"requires_human_approval": False, "human_approved": True, "approval_reason": None}

    plan = state.get("rebalance_plan") or {}
    approvals_requested.labels(reason).inc()

    # Logged at debug, not info. LangGraph re-executes this node from its start
    # when a run resumes, so an info-level "pausing for approval" here prints
    # again on every successful resume -- which made the log of the one
    # operation you most want to audit read as though the run had paused twice
    # and never been approved. The decision is logged below, after it resolves.
    logger.debug("[Approval] interrupting for human review: %s", reason)

    decision = interrupt(
        {
            "reason": reason,
            "explanation": APPROVAL_EXPLANATIONS.get(reason, reason),
            "market_regime": state.get("market_regime"),
            "suitability_result": state.get("suitability_result"),
            "tax_assessment": state.get("tax_assessment"),
            "rebalance_plan": plan,
            "tax_blocked_recommendations": state.get("tax_blocked_recommendations"),
            "degradations": state.get("degradations"),
            "research_attempts": state.get("research_attempts", 0),
        }
    )

    # The decision may arrive as a bare bool or as {"approved": ..., "note": ...}.
    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
        note = decision.get("note")
    else:
        approved = bool(decision)
        note = None

    logger.info(
        "[Approval] reviewer %s the run (requested because: %s)",
        "approved" if approved else "rejected", reason,
    )

    if not approved:
        # A rejected plan is emptied rather than left in state. Otherwise the
        # report would describe trades that were explicitly declined, and
        # anything reading the state afterwards would see an actionable plan.
        return {
            "requires_human_approval": True,
            "human_approved": False,
            "approval_reason": reason,
            "rejection_note": note,
            "rebalance_plan": {
                **plan,
                "proposals": [],
                "notes": list(plan.get("notes") or [])
                + [
                    "This plan was reviewed by a person and rejected"
                    + (f": {note}" if note else ".")
                    + " No trades are proposed."
                ],
            },
        }

    return {
        "requires_human_approval": True,
        "human_approved": True,
        "approval_reason": reason,
        "rejection_note": None,
    }


# --- Graph -------------------------------------------------------------------


def _build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("diagnostics", diagnostics_node)
    workflow.add_node("market_regime", market_regime_node)
    workflow.add_node("stock_research", stock_research_node)
    workflow.add_node("suitability", suitability_node)
    workflow.add_node("tax_awareness", tax_awareness_node)
    workflow.add_node("guardrail_gate", guardrail_gate_node)
    workflow.add_node("rebalance", rebalance_node)
    workflow.add_node("approval_gate", approval_gate_node)
    workflow.add_node("finance_report", finance_report_node)

    # Fan out: diagnostics and market_regime share no data dependency.
    workflow.add_edge(START, "diagnostics")
    workflow.add_edge(START, "market_regime")

    # Fan in: research needs both.
    workflow.add_edge("diagnostics", "stock_research")
    workflow.add_edge("market_regime", "stock_research")

    # Fan out: the two guardrails both consume only the candidates.
    workflow.add_edge("stock_research", "suitability")
    workflow.add_edge("stock_research", "tax_awareness")

    # Fan in at the gate, which reconciles them.
    workflow.add_edge("suitability", "guardrail_gate")
    workflow.add_edge("tax_awareness", "guardrail_gate")

    workflow.add_conditional_edges(
        "guardrail_gate",
        route_after_guardrails,
        {"retry": "stock_research", "proceed": "rebalance"},
    )

    # Approval before the report, so a rejected run never produces a
    # client-facing document describing trades the reviewer declined.
    workflow.add_edge("rebalance", "approval_gate")
    workflow.add_edge("approval_gate", "finance_report")
    workflow.add_edge("finance_report", END)

    return workflow


_compiled_workflow = None


def get_workflow():
    """Compile the graph lazily against the durable checkpointer.

    Lazy rather than at import so tests and the demo notebook can point
    CHECKPOINT_DB_PATH at a temporary location first, and so importing this
    module does not create files on disk.
    """
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = _build_graph().compile(checkpointer=get_checkpointer())
    return _compiled_workflow


def reset_workflow() -> None:
    """Drop the compiled graph so the next call rebinds to a fresh checkpointer."""
    global _compiled_workflow
    _compiled_workflow = None


# --- Entry points ------------------------------------------------------------


def load_client_state(client_id: int, run_id: Optional[str] = None) -> AgentState:
    """Build the initial state: profile, live positions and the policy snapshot.

    The policy is snapshotted once here rather than resolved per node, so a
    policy activated mid-run cannot leave some nodes enforcing the old limits
    and others the new ones.
    """
    db = SessionLocal()
    try:
        client = db.query(ClientProfileModel).filter(ClientProfileModel.id == client_id).first()
        if client is None:
            raise ValueError(f"No client profile with id {client_id}")

        policy = resolve_policy(db, client)
        view = load_portfolio(db, client)

        holdings: Dict[str, float] = {}
        for holding in view.holdings:
            holdings[holding.symbol] = holdings.get(holding.symbol, 0.0) + holding.market_value
        if view.cash > 0:
            holdings["CASH"] = view.cash

        profile: ClientProfileState = {
            "client_id": client.id,
            "org_id": client.org_id,
            "name": client.name,
            "age": client.age,
            "risk_tolerance": client.risk_tolerance,
            "time_horizon_years": client.time_horizon_years,
            "goals": list(client.goals or []),
            "net_worth": float(client.net_worth or 0),
            "notes": client.notes,
            "holdings": holdings,
            "accounts": [
                {
                    "id": account.id,
                    "name": account.name,
                    "type": account.account_type,
                    "tax_treatment": account.tax_treatment,
                    "cash": float(account.cash_balance or 0),
                }
                for account in client.accounts
            ],
            "unpriced_symbols": view.unpriced_symbols,
        }

        provenance = quote_provenance(view.quotes)

        return AgentState(
            run_id=run_id or str(uuid.uuid4()),
            client_profile=profile,
            policy=PolicySnapshot(
                version=policy.version,
                source=policy.source,
                risk_tier=policy.risk_tier,
                max_position_pct=policy.max_position_pct,
                max_sector_pct=policy.max_sector_pct,
                min_cash_pct=policy.min_cash_pct,
                max_cash_pct=policy.max_cash_pct,
                min_market_cap=policy.min_market_cap,
                target_allocation=policy.target_allocation,
                benchmark_ticker=policy.benchmark_ticker,
                is_approved=policy.is_approved,
                summary=policy.summary(),
            ),
            portfolio_diagnostics=None,
            market_regime=None,
            candidate_stocks=None,
            suitability_result=None,
            tax_assessment=None,
            rebalance_plan=None,
            final_report=None,
            audit_trail=[],
            degradations=[],
            requires_human_approval=False,
            approval_reason=None,
            human_approved=None,
            rejection_note=None,
            research_attempts=0,
            needs_research_retry=False,
            excluded_tickers=[],
            guardrail_feedback=[],
            tax_blocked_recommendations=[],
            data_as_of=provenance.get("as_of"),
            llm_enabled=settings.llm_configured,
        )
    finally:
        db.close()


def run_client_graph(client_id: int, run_id: Optional[str] = None) -> dict:
    """Start a run. Returns the final state, or one containing `__interrupt__`."""
    initial_state = load_client_state(client_id, run_id=run_id)
    resolved_run_id = initial_state["run_id"]
    config = {"configurable": {"thread_id": resolved_run_id}}
    runs_started.inc()

    # The spend ceiling is bound to the thread rather than carried on state:
    # LangGraph checkpoints its state and a budget holds a lock, which is not
    # serializable. A resumed run gets a fresh budget, which is correct --
    # the earlier spend already happened and is already recorded.
    with log_context(run_id=resolved_run_id, client_id=client_id), run_budget():
        logger.info("Starting analysis run for client %s", client_id)
        result = get_workflow().invoke(initial_state, config=config)

    return {"run_id": resolved_run_id, "result": result}


def resume_client_graph(run_id: str, approved: bool, note: Optional[str] = None) -> dict:
    """Resume a run paused at the approval gate."""
    config = {"configurable": {"thread_id": run_id}}
    with log_context(run_id=run_id), run_budget():
        logger.info("Resuming run with approved=%s", approved)
        result = get_workflow().invoke(
            Command(resume={"approved": approved, "note": note}), config=config
        )
    return {"run_id": run_id, "result": result}


def run_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Condense a finished run for an API response or a job result."""
    plan = result.get("rebalance_plan") or {}
    suitability = result.get("suitability_result") or {}
    diagnostics = result.get("portfolio_diagnostics") or {}
    regime = result.get("market_regime") or {}
    return {
        "flaws": len(diagnostics.get("flaws") or []),
        "regime": regime.get("regime_label"),
        "regime_confidence": regime.get("confidence"),
        "recommendations": len(suitability.get("adjusted_recommendations") or []),
        "proposals": len(plan.get("proposals") or []),
        "gross_notional": plan.get("gross_notional", 0.0),
        "blocked": result.get("tax_blocked_recommendations") or [],
        "degraded_nodes": sorted({d.get("node") for d in (result.get("degradations") or [])}),
        "human_approved": result.get("human_approved"),
        "approval_reason": result.get("approval_reason"),
        "completed_at": utcnow().isoformat(),
    }
