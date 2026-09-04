"""Running an analysis and persisting everything it produced.

This is the seam between the graph (which owns control flow and owns no
database) and the database (which owns durability and knows nothing about
LangGraph). It exists as its own module because the persistence rules are
subtle enough to be worth stating in one place:

* **The audit trail is written per node, deduplicated on (run_id, node_name,
  started_at).** A resumed run replays its accumulated `audit_trail`, so a
  naive insert would duplicate every record from before the interrupt.

* **A rejected run still produces a stored report, marked rejected.** Deleting
  it would be worse: books-and-records rules expect the artifact to exist, and
  an advisor needs to see what was declined. What must never happen is a
  rejected report that is indistinguishable from an approved one, which is
  exactly what the previous version stored.

* **Nothing executes trades here.** The graph proposes; execution is a
  separate, explicitly authorized step. Keeping them apart means an analysis
  run can never move money as a side effect.
"""

from decimal import Decimal
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from db import AgentRun, Approval, ClientProfile, Job, Report, TradeProposal, utcnow
from logging_setup import get_logger
from metrics import approvals_decided, run_duration, runs_finished
from orchestrator import load_client_state, resume_client_graph, run_client_graph, run_summary
from services import jobs
from services.audit import Action, record as record_audit
from services.performance import capture_snapshot, record_recommendations

logger = get_logger(__name__)

JOB_TYPE = "analysis_run"

# Node order, used only to turn "which nodes have finished" into a progress
# percentage. Approximate on purpose: the retry loop means the true count
# varies per run, and a progress bar that occasionally sits still is better
# than one that lies precisely.
_NODE_SEQUENCE = [
    "diagnostics", "market_regime", "stock_research",
    "suitability", "tax_awareness", "rebalance", "finance_report",
]


def _parse_iso(value: Optional[str]):
    """ISO string to naive UTC datetime, matching the column convention.

    Agents stamp timestamps as ISO strings; the columns are timezone-naive.
    Comparing a naive column against an aware value raises at runtime, so the
    tzinfo is dropped here rather than at each call site.
    """
    if not value:
        return None
    from datetime import datetime

    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed


def persist_audit_trail(
    db: Session, run_id: str, org_id: int, client_id: int, state: Dict[str, Any]
) -> int:
    """Write each node's record. Idempotent across a resume.

    Deduplicated on (run_id, node_name, started_at) because LangGraph returns
    the whole accumulated trail after a resume, including everything that ran
    before the interrupt.
    """
    records = state.get("audit_trail") or []
    if not records:
        return 0

    existing = {
        (row.node_name, row.started_at)
        for row in db.query(AgentRun.node_name, AgentRun.started_at)
        .filter(AgentRun.run_id == run_id)
        .all()
    }

    written = 0
    for record in records:
        started_at = _parse_iso(record.get("started_at"))
        if (record.get("node_name"), started_at) in existing:
            continue
        cost = record.get("cost_usd")
        db.add(
            AgentRun(
                org_id=org_id,
                client_id=client_id,
                run_id=run_id,
                node_name=record.get("node_name", "unknown"),
                started_at=started_at or utcnow(),
                completed_at=_parse_iso(record.get("completed_at")),
                duration_ms=record.get("duration_ms"),
                input_snapshot=record.get("input_snapshot") or {},
                output_snapshot=record.get("output_snapshot") or {},
                model_used=record.get("model_used"),
                prompt_version=record.get("prompt_version"),
                temperature=record.get("temperature"),
                prompt_tokens=record.get("prompt_tokens"),
                completion_tokens=record.get("completion_tokens"),
                cost_usd=Decimal(str(cost)) if cost is not None else None,
                policy_version=record.get("policy_version"),
                data_provenance=record.get("data_provenance"),
                status=record.get("status", "success"),
                degraded=bool(record.get("degraded")),
                error_detail=record.get("error_detail"),
            )
        )
        written += 1

    db.flush()
    return written


def persist_proposals(
    db: Session, run_id: str, client: ClientProfile, state: Dict[str, Any]
) -> int:
    """Store the trade plan, including what was blocked.

    Blocked and deferred items are stored as proposals with a status, not
    discarded. An orders table alone records only what happened; the more
    interesting half of this system is what the guardrails stopped, and that
    has to be queryable afterwards.
    """
    if db.query(TradeProposal).filter(TradeProposal.run_id == run_id).first() is not None:
        return 0  # already persisted, e.g. on a resume

    plan = state.get("rebalance_plan") or {}
    blocked = {t.upper() for t in (state.get("tax_blocked_recommendations") or [])}
    written = 0

    for proposal in plan.get("proposals") or []:
        db.add(
            TradeProposal(
                org_id=client.org_id,
                client_id=client.id,
                account_id=proposal.get("account_id"),
                run_id=run_id,
                symbol=str(proposal.get("symbol", "")).upper(),
                side=str(proposal.get("side", "BUY")).upper(),
                quantity=Decimal(str(proposal["quantity"])) if proposal.get("quantity") else None,
                notional=Decimal(str(proposal.get("notional") or 0)),
                target_weight=proposal.get("target_weight"),
                current_weight=proposal.get("current_weight"),
                rationale=proposal.get("rationale"),
                addresses_flaw=(proposal.get("addresses_flaw") or "")[:255] or None,
                confidence=proposal.get("confidence"),
                estimated_tax_cost=(
                    Decimal(str(proposal["estimated_tax_cost"]))
                    if proposal.get("estimated_tax_cost") is not None
                    else None
                ),
                status="proposed",
                sequence=proposal.get("sequence", 0),
            )
        )
        written += 1

    suitability = state.get("suitability_result") or {}
    for recommendation in suitability.get("adjusted_recommendations") or []:
        ticker = str(recommendation.get("ticker", "")).upper()
        if ticker in blocked:
            db.add(
                TradeProposal(
                    org_id=client.org_id,
                    client_id=client.id,
                    run_id=run_id,
                    symbol=ticker,
                    side="BUY",
                    notional=Decimal(str(recommendation.get("allocation_amount") or 0)),
                    rationale=recommendation.get("regime_fit_rationale"),
                    addresses_flaw=(recommendation.get("addresses_flaw") or "")[:255] or None,
                    confidence=recommendation.get("confidence"),
                    status="blocked",
                    blocked_reason="Withheld by the tax guardrail (wash-sale rule).",
                )
            )
            written += 1

    for ticker in blocked:
        if not any(
            str(r.get("ticker", "")).upper() == ticker
            for r in (suitability.get("adjusted_recommendations") or [])
        ):
            db.add(
                TradeProposal(
                    org_id=client.org_id,
                    client_id=client.id,
                    run_id=run_id,
                    symbol=ticker,
                    side="BUY",
                    notional=Decimal("0"),
                    status="blocked",
                    blocked_reason="Withheld by the tax guardrail (wash-sale rule).",
                )
            )
            written += 1

    db.flush()
    return written


def persist_report(
    db: Session, run_id: str, client: ClientProfile, state: Dict[str, Any]
) -> Optional[Report]:
    """Store the report, marked with whether it cleared human review."""
    text = state.get("final_report")
    if not text:
        return None

    existing = db.query(Report).filter(Report.run_id == run_id).first()
    approval_state = (
        "rejected"
        if state.get("human_approved") is False
        else "approved" if state.get("human_approved") else "pending"
    )
    degraded = bool(state.get("degradations"))

    payload = {
        "summary": run_summary(state),
        "portfolio_diagnostics": state.get("portfolio_diagnostics"),
        "market_regime": state.get("market_regime"),
        "suitability_result": state.get("suitability_result"),
        "tax_assessment": state.get("tax_assessment"),
        # What the guardrail gate withheld, and why. This was computed on
        # every run and then dropped at the persistence boundary, so the
        # stored report -- the artifact a client reads, and the one an
        # examiner asks for -- recorded the recommendations that survived and
        # no trace of the ones a control removed. A report that shows only the
        # survivors is indistinguishable from a run where nothing was blocked.
        "tax_blocked_recommendations": state.get("tax_blocked_recommendations") or [],
        "rebalance_plan": state.get("rebalance_plan"),
        "degradations": state.get("degradations"),
        "policy": state.get("policy"),
    }

    if existing is not None:
        existing.report_text = text
        existing.structured_payload = payload
        existing.approval_state = approval_state
        existing.degraded = degraded
        db.flush()
        return existing

    report = Report(
        org_id=client.org_id,
        client_id=client.id,
        run_id=run_id,
        report_text=text,
        structured_payload=payload,
        approval_state=approval_state,
        policy_version=(state.get("policy") or {}).get("version"),
        llm_enabled=bool(state.get("llm_enabled")),
        degraded=degraded,
        data_as_of=_parse_iso(state.get("data_as_of")),
        disclosure_version="2026-07",
    )
    db.add(report)
    db.flush()
    return report


def persist_run(
    db: Session, run_id: str, client: ClientProfile, state: Dict[str, Any]
) -> Dict[str, Any]:
    """Write everything one finished (or paused) run produced."""
    persist_audit_trail(db, run_id, client.org_id, client.id, state)

    interrupted = "__interrupt__" in state
    if interrupted:
        interrupt_value = state["__interrupt__"][0].value
        pending = (
            db.query(Approval)
            .filter(Approval.run_id == run_id, Approval.decision.is_(None))
            .first()
        )
        if pending is None:
            db.add(
                Approval(
                    org_id=client.org_id,
                    client_id=client.id,
                    run_id=run_id,
                    reason=interrupt_value.get("reason"),
                    context=interrupt_value,
                )
            )
            record_audit(
                db,
                org_id=client.org_id,
                action=Action.APPROVAL_REQUESTED,
                entity_type="run",
                entity_id=run_id,
                detail={"reason": interrupt_value.get("reason"), "client_id": client.id},
            )
        db.flush()
        return {
            "run_id": run_id,
            "status": "pending_approval",
            "interrupt": interrupt_value,
        }

    persist_proposals(db, run_id, client, state)
    report = persist_report(db, run_id, client, state)

    # Outcome tracking only for recommendations that actually stood. Scoring a
    # withheld or rejected pick would measure advice the system declined to
    # give.
    if state.get("human_approved") is not False:
        suitability = state.get("suitability_result") or {}
        record_recommendations(
            db, client, run_id, suitability.get("adjusted_recommendations") or []
        )

    try:
        capture_snapshot(db, client)
    except Exception:  # noqa: BLE001 -- a snapshot failure must not fail the run
        logger.warning("Could not capture a portfolio snapshot for client %s.", client.id)

    record_audit(
        db,
        org_id=client.org_id,
        action=Action.RUN_COMPLETED,
        entity_type="run",
        entity_id=run_id,
        detail=run_summary(state),
    )

    summary = run_summary(state)
    return {
        "run_id": run_id,
        "status": "completed",
        "report_id": report.id if report else None,
        "summary": summary,
        "degradations": state.get("degradations") or [],
        "llm_enabled": bool(state.get("llm_enabled")),
    }


@jobs.register(JOB_TYPE)
def run_analysis_job(db: Session, job: Job) -> Dict[str, Any]:
    """Job handler: execute the graph for one client and persist the result."""
    client = db.query(ClientProfile).filter(ClientProfile.id == job.client_id).first()
    if client is None:
        raise ValueError(f"Client {job.client_id} no longer exists.")

    if not jobs.heartbeat(db, job, step="starting", progress=1.0):
        raise jobs.JobCancelled()

    state = load_client_state(client.id)
    run_id = state["run_id"]
    job.run_id = run_id
    db.commit()

    record_audit(
        db,
        org_id=client.org_id,
        action=Action.RUN_REQUESTED,
        user_id=job.requested_by_user_id,
        entity_type="run",
        entity_id=run_id,
        detail={"client_id": client.id, "job_id": job.job_id},
    )
    db.commit()

    jobs.heartbeat(db, job, step="analysing portfolio", progress=5.0)

    started = utcnow()
    result = run_client_graph(client.id, run_id=run_id)["result"]
    duration = (utcnow() - started).total_seconds()
    run_duration.observe(duration)

    # Progress is derived after the fact rather than streamed: LangGraph's
    # invoke is synchronous, so there is no mid-run callback to hook. The
    # honest alternative to a fake progress bar is to report what actually
    # completed.
    completed_nodes = {r.get("node_name") for r in (result.get("audit_trail") or [])}
    progress = 100.0 * len(completed_nodes & set(_NODE_SEQUENCE)) / len(_NODE_SEQUENCE)
    jobs.heartbeat(db, job, step="persisting results", progress=min(95.0, progress))

    outcome = persist_run(db, run_id, client, result)
    db.commit()

    runs_finished.labels(outcome["status"]).inc()
    return outcome


def decide_approval(
    db: Session,
    approval: Approval,
    *,
    approved: bool,
    user_id: int,
    note: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> Dict[str, Any]:
    """Record a decision and resume the paused run.

    Ordering matters and the previous version got it wrong: it committed the
    decision *before* resuming, so a failed resume left a run that was
    permanently wedged -- decided, unresumed, and rejected on retry by the
    "already decided" branch. Here the decision is flushed but committed
    together with the resumed result, so a resume failure rolls the decision
    back and the operation can simply be retried.
    """
    client = (
        db.query(ClientProfile).filter(ClientProfile.id == approval.client_id).first()
        if approval.client_id
        else None
    )

    approval.decision = "approved" if approved else "rejected"
    approval.decided_at = utcnow()
    approval.decided_by_user_id = user_id
    approval.notes = note
    approval.ip_address = ip_address
    db.flush()

    record_audit(
        db,
        org_id=approval.org_id,
        action=Action.APPROVAL_GRANTED if approved else Action.APPROVAL_REJECTED,
        user_id=user_id,
        entity_type="run",
        entity_id=approval.run_id,
        detail={"reason": approval.reason, "note": note},
        ip_address=ip_address,
    )
    approvals_decided.labels(approval.decision).inc()

    result = resume_client_graph(approval.run_id, approved=approved, note=note)["result"]

    if client is not None:
        outcome = persist_run(db, approval.run_id, client, result)
    else:
        outcome = {"run_id": approval.run_id, "status": "completed"}

    db.commit()
    runs_finished.labels(outcome.get("status", "completed")).inc()
    return outcome
