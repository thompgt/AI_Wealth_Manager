"""
Unified FastAPI app for the new multi-agent orchestrator (replaces the old
workflow.py-driven server described in PLAN.md step 9).

Endpoints:
  Client profile CRUD  -- POST/GET/PUT /api/v1/clients
  Trigger a graph run  -- POST /api/v1/clients/{client_id}/run
  Fetch a report       -- GET  /api/v1/clients/{client_id}/reports (list)
                           GET  /api/v1/reports/{report_id}
  Approvals            -- POST /api/v1/runs/{run_id}/approve

All endpoints except /health require an `X-API-Key` header matching
settings.API_AUTH_KEY.
"""

import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import settings
from db import AgentRun, Approval, ClientProfile, Holding, Report, SessionLocal, init_db, utcnow
from logging_setup import configure_logging, get_logger
from orchestrator import resume_client_graph, run_client_graph

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup checks.

    Two things had to move here. First, nothing ever called `init_db()` in
    the serving path, so a fresh deployment answered every request with a
    'no such table' 500 until somebody remembered to run `python db.py` by
    hand. Second, the app booted happily with the built-in development API
    key and a placeholder Gemini key -- i.e. it would serve an unauthenticated,
    AI-less system to production traffic without a word of complaint.
    """
    init_db()

    problems = settings.validate_for_environment()
    if problems:
        for problem in problems:
            logger.critical("FATAL CONFIG: %s", problem)
        raise RuntimeError(
            f"Refusing to start in ENVIRONMENT={settings.ENVIRONMENT} with "
            f"{len(problems)} unsafe setting(s); see the log above."
        )

    if settings.is_development:
        if settings.API_AUTH_KEY == "DEV_ONLY_CHANGE_ME":
            logger.warning(
                "Running with the built-in development API key. Set API_AUTH_KEY "
                "before deploying anywhere reachable."
            )
        if not settings.llm_configured:
            logger.warning(
                "No GEMINI_API_KEY configured -- the market regime, stock ranking and "
                "report agents will run in deterministic-fallback mode. Output will be "
                "produced, but none of it is LLM-generated."
            )
    logger.info(
        "%s ready (env=%s, model=%s, llm_configured=%s)",
        settings.PROJECT_NAME, settings.ENVIRONMENT, settings.GEMINI_MODEL,
        settings.llm_configured,
    )
    yield


app = FastAPI(title="AI Wealth Manager Engine", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    # secrets.compare_digest rather than `!=`: `!=` on strings short-circuits
    # at the first differing byte, which leaks the key one character at a
    # time to anyone who can measure response latency.
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.API_AUTH_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


# --------------------------------------------------------------------------
# Pydantic schemas
# --------------------------------------------------------------------------

# The risk tolerance string drives position caps, concentration limits and
# the beta guardrail. Accepting a free-form string meant a typo like
# "moderate" or "Balanced" silently fell through to every agent's default
# tier instead of being rejected at the door -- a client could be sized
# against limits they never agreed to.
RiskTolerance = Literal["Conservative", "Moderate", "Aggressive"]


class HoldingIn(BaseModel):
    symbol: str = Field(min_length=1, max_length=12)
    # Negative quantities would flow straight into portfolio valuation and
    # produce nonsensical negative concentrations.
    quantity: float = Field(ge=0)
    cost_basis: float = Field(default=0.0, ge=0)


class ClientCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: Optional[str] = None
    age: int = Field(ge=18, le=120)
    risk_tolerance: RiskTolerance
    time_horizon_years: int = Field(ge=0, le=100)
    goals: List[str] = []
    net_worth: float = Field(default=0.0, ge=0)
    holdings: List[HoldingIn] = []


class ClientUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    email: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=18, le=120)
    risk_tolerance: Optional[RiskTolerance] = None
    time_horizon_years: Optional[int] = Field(default=None, ge=0, le=100)
    goals: Optional[List[str]] = None
    net_worth: Optional[float] = Field(default=None, ge=0)


class HoldingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    quantity: float
    cost_basis: float


class ClientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: Optional[str]
    age: int
    risk_tolerance: str
    time_horizon_years: int
    goals: List[str]
    net_worth: float
    holdings: List[HoldingOut]


class ApprovalDecision(BaseModel):
    approved: bool
    decided_by: Optional[str] = None
    notes: Optional[str] = None


# --------------------------------------------------------------------------
# Operational endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health(db: Session = Depends(get_db)):
    """Unauthenticated liveness/readiness probe.

    Deliberately outside the API-key dependency so a load balancer or
    container orchestrator can reach it. Reports whether the LLM layer is
    actually enabled, because "up but running entirely on deterministic
    fallbacks" is a state that otherwise looks identical to healthy from the
    outside -- which is exactly how a dead AI layer goes unnoticed.
    """
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        logger.exception("Health check database probe failed")
        db_ok = False

    status = "ok" if db_ok else "degraded"
    return {
        "status": status,
        "environment": settings.ENVIRONMENT,
        "database": "ok" if db_ok else "unreachable",
        "llm_enabled": settings.llm_configured,
        "llm_model": settings.GEMINI_MODEL if settings.llm_configured else None,
        "agents": [
            "portfolio_diagnostics",
            "market_regime",
            "stock_research",
            "suitability",
            "tax_awareness",
            "finance_report",
        ],
    }


# --------------------------------------------------------------------------
# Client profile CRUD
# --------------------------------------------------------------------------

def _client_to_out(client: ClientProfile, db: Session) -> ClientOut:
    holdings = db.query(Holding).filter(Holding.client_id == client.id).all()
    return ClientOut(
        id=client.id,
        name=client.name,
        email=client.email,
        age=client.age,
        risk_tolerance=client.risk_tolerance,
        time_horizon_years=client.time_horizon_years,
        goals=list(client.goals or []),
        net_worth=client.net_worth,
        holdings=[HoldingOut.model_validate(h) for h in holdings],
    )


@app.post("/api/v1/clients", response_model=ClientOut, dependencies=[Depends(require_api_key)])
def create_client(req: ClientCreate, db: Session = Depends(get_db)):
    client = ClientProfile(
        name=req.name,
        email=req.email,
        age=req.age,
        risk_tolerance=req.risk_tolerance,
        time_horizon_years=req.time_horizon_years,
        goals=req.goals,
        net_worth=req.net_worth,
    )
    db.add(client)
    db.commit()
    db.refresh(client)

    for h in req.holdings:
        db.add(Holding(client_id=client.id, symbol=h.symbol, quantity=h.quantity, cost_basis=h.cost_basis))
    db.commit()

    return _client_to_out(client, db)


@app.get("/api/v1/clients", response_model=List[ClientOut], dependencies=[Depends(require_api_key)])
def list_clients(db: Session = Depends(get_db)):
    clients = db.query(ClientProfile).all()
    return [_client_to_out(c, db) for c in clients]


@app.get("/api/v1/clients/{client_id}", response_model=ClientOut, dependencies=[Depends(require_api_key)])
def get_client(client_id: int, db: Session = Depends(get_db)):
    client = db.query(ClientProfile).filter(ClientProfile.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    return _client_to_out(client, db)


@app.put("/api/v1/clients/{client_id}", response_model=ClientOut, dependencies=[Depends(require_api_key)])
def update_client(client_id: int, req: ClientUpdate, db: Session = Depends(get_db)):
    client = db.query(ClientProfile).filter(ClientProfile.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    for field, value in req.model_dump(exclude_unset=True).items():
        setattr(client, field, value)
    client.updated_at = utcnow()
    db.commit()
    db.refresh(client)
    return _client_to_out(client, db)


# --------------------------------------------------------------------------
# Graph runs
# --------------------------------------------------------------------------

def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse an agent's ISO timestamp into a NAIVE UTC datetime.

    Agents emit timezone-aware ISO strings, but the DateTime columns are
    naive and round-trip as naive. Keeping the aware value here would make
    every comparison against a stored row false -- which silently defeats the
    audit-trail dedupe above.
    """
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _persist_audit_trail(db: Session, client_id: Optional[int], run_id: str, audit_trail: List[dict]) -> None:
    """Write this run's audit records, skipping any already stored.

    LangGraph's resumed result carries the FULL accumulated audit_trail,
    including every record from before the approval pause. Persisting it
    verbatim on resume duplicated every pre-interrupt node in agent_runs --
    which is a real problem for an audit table whose whole purpose is being a
    faithful record of what ran. Dedupe on (node_name, started_at), which
    uniquely identifies a node execution within a run.
    """
    existing = {
        (row.node_name, row.started_at)
        for row in db.query(AgentRun.node_name, AgentRun.started_at).filter(AgentRun.run_id == run_id)
    }

    added = 0
    for record in audit_trail:
        key = (record.get("node_name"), _parse_iso(record.get("started_at")))
        if key in existing:
            continue
        existing.add(key)
        added += 1
        db.add(
            AgentRun(
                client_id=client_id,
                run_id=run_id,
                node_name=record.get("node_name"),
                started_at=key[1],
                completed_at=_parse_iso(record.get("completed_at")),
                output_snapshot={"summary": record.get("summary")},
                status=record.get("status", "success"),
                error_detail=record.get("error_detail"),
            )
        )
    db.commit()
    logger.info("Persisted %d new audit record(s) for run %s", added, run_id)


def _persist_report(db: Session, client_id: Optional[int], run_id: str, result: dict) -> Optional[Report]:
    final_report = result.get("final_report")
    if not final_report:
        return None

    # A run produces exactly one report. Approving the same run twice (a
    # double-clicked button, a retried request) must not create a second
    # copy -- return the existing one instead.
    existing = db.query(Report).filter(Report.run_id == run_id).first()
    if existing is not None:
        return existing

    report = Report(
        client_id=client_id,
        run_id=run_id,
        report_text=final_report,
        structured_payload={
            "portfolio_diagnostics": result.get("portfolio_diagnostics"),
            "market_regime": result.get("market_regime"),
            "candidate_stocks": result.get("candidate_stocks"),
            "suitability_result": result.get("suitability_result"),
            "tax_assessment": result.get("tax_assessment"),
            "tax_blocked_recommendations": result.get("tax_blocked_recommendations"),
            "research_attempts": result.get("research_attempts"),
        },
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def _run_response(client_id: Optional[int], run_id: str, result: dict, db: Session) -> dict:
    _persist_audit_trail(db, client_id, run_id, result.get("audit_trail", []))

    if "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        # Only open a new approval request if one isn't already outstanding,
        # so a retried trigger doesn't pile up duplicate pending approvals
        # against the same run.
        pending = (
            db.query(Approval)
            .filter(Approval.run_id == run_id, Approval.decision.is_(None))
            .first()
        )
        if pending is None:
            db.add(Approval(run_id=run_id, notes=str(interrupt_obj.value)))
            db.commit()
        return {
            "run_id": run_id,
            "status": "pending_approval",
            "interrupt": interrupt_obj.value,
        }

    report = _persist_report(db, client_id, run_id, result)
    return {
        "run_id": run_id,
        "status": "completed",
        "portfolio_diagnostics": result.get("portfolio_diagnostics"),
        "market_regime": result.get("market_regime"),
        "candidate_stocks": result.get("candidate_stocks"),
        "suitability_result": result.get("suitability_result"),
        "tax_assessment": result.get("tax_assessment"),
        "tax_blocked_recommendations": result.get("tax_blocked_recommendations") or [],
        "research_attempts": result.get("research_attempts", 0),
        "requires_human_approval": result.get("requires_human_approval", False),
        "human_approved": result.get("human_approved"),
        "audit_trail": result.get("audit_trail", []),
        "final_report": result.get("final_report"),
        "llm_enabled": settings.llm_configured,
        "report_id": report.id if report else None,
    }


@app.post("/api/v1/clients/{client_id}/run", dependencies=[Depends(require_api_key)])
def trigger_run(client_id: int, db: Session = Depends(get_db)):
    client = db.query(ClientProfile).filter(ClientProfile.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    try:
        run = run_client_graph(client_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # Don't leak stack traces / connection strings to the caller; the
        # detail is in the log. A failed graph run is a server-side fault,
        # not a malformed request, so 500 is the honest status code -- the
        # previous blanket 400 told clients to fix a request that was fine.
        logger.exception("Graph run failed for client %s", client_id)
        raise HTTPException(status_code=500, detail="Analysis run failed; see server logs.")

    return _run_response(client_id, run["run_id"], run["result"], db)


@app.post("/api/v1/runs/{run_id}/approve", dependencies=[Depends(require_api_key)])
def approve_run(run_id: str, req: ApprovalDecision, db: Session = Depends(get_db)):
    approval = (
        db.query(Approval)
        .filter(Approval.run_id == run_id, Approval.decision.is_(None))
        .order_by(Approval.id.desc())
        .first()
    )
    if not approval:
        # Distinguish "never existed" from "already decided" -- resubmitting a
        # decision for a settled run is a different mistake than typing the
        # wrong run id, and the previous code silently re-decided settled runs.
        settled = db.query(Approval).filter(Approval.run_id == run_id).first()
        if settled is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Run {run_id} was already {settled.decision} at {settled.decided_at}.",
            )
        raise HTTPException(status_code=404, detail="No pending approval for this run_id")

    approval.decided_at = utcnow()
    approval.decided_by = req.decided_by
    approval.decision = "approved" if req.approved else "rejected"
    approval.notes = req.notes or approval.notes
    db.commit()

    try:
        resumed = resume_client_graph(run_id, approved=req.approved)
    except Exception:
        logger.exception("Failed to resume run %s", run_id)
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not resume this run. Its checkpoint may be missing -- see "
                "server logs."
            ),
        )

    agent_run = db.query(AgentRun).filter(AgentRun.run_id == run_id).first()
    client_id = agent_run.client_id if agent_run else None
    return _run_response(client_id, run_id, resumed["result"], db)


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

@app.get("/api/v1/clients/{client_id}/reports", dependencies=[Depends(require_api_key)])
def list_reports(client_id: int, db: Session = Depends(get_db)):
    reports = (
        db.query(Report)
        .filter(Report.client_id == client_id)
        .order_by(Report.generated_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "run_id": r.run_id,
            "generated_at": r.generated_at,
            "report_text": r.report_text,
            "structured_payload": r.structured_payload,
        }
        for r in reports
    ]


@app.get("/api/v1/reports/{report_id}", dependencies=[Depends(require_api_key)])
def get_report(report_id: int, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return {
        "id": report.id,
        "client_id": report.client_id,
        "run_id": report.run_id,
        "generated_at": report.generated_at,
        "report_text": report.report_text,
        "structured_payload": report.structured_payload,
    }
