"""
Unified FastAPI app for the new multi-agent orchestrator (replaces the old
workflow.py-driven server described in PLAN.md step 9).

Endpoints:
  Client profile CRUD  -- POST/GET/PUT /api/v1/clients
  Trigger a graph run  -- POST /api/v1/clients/{client_id}/run
  Fetch a report       -- GET  /api/v1/clients/{client_id}/reports (list)
                           GET  /api/v1/reports/{report_id}
  Approvals            -- POST /api/v1/runs/{run_id}/approve

All endpoints require an `X-API-Key` header matching settings.API_AUTH_KEY.
"""

from datetime import datetime
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import settings
from db import AgentRun, Approval, ClientProfile, Holding, Report, SessionLocal
from orchestrator import resume_client_graph, run_client_graph

app = FastAPI(title="AI Wealth Manager Engine")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if x_api_key != settings.API_AUTH_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


# --------------------------------------------------------------------------
# Pydantic schemas
# --------------------------------------------------------------------------

class HoldingIn(BaseModel):
    symbol: str
    quantity: float
    cost_basis: float = 0.0


class ClientCreate(BaseModel):
    name: str
    email: Optional[str] = None
    age: int
    risk_tolerance: str  # "Conservative" | "Moderate" | "Aggressive"
    time_horizon_years: int
    goals: List[str] = []
    net_worth: float = 0.0
    holdings: List[HoldingIn] = []


class ClientUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    age: Optional[int] = None
    risk_tolerance: Optional[str] = None
    time_horizon_years: Optional[int] = None
    goals: Optional[List[str]] = None
    net_worth: Optional[float] = None


class HoldingOut(BaseModel):
    symbol: str
    quantity: float
    cost_basis: float

    class Config:
        from_attributes = True


class ClientOut(BaseModel):
    id: int
    name: str
    email: Optional[str]
    age: int
    risk_tolerance: str
    time_horizon_years: int
    goals: List[str]
    net_worth: float
    holdings: List[HoldingOut]

    class Config:
        from_attributes = True


class ApprovalDecision(BaseModel):
    approved: bool
    decided_by: Optional[str] = None
    notes: Optional[str] = None


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
    client.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(client)
    return _client_to_out(client, db)


# --------------------------------------------------------------------------
# Graph runs
# --------------------------------------------------------------------------

def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _persist_audit_trail(db: Session, client_id: int, run_id: str, audit_trail: List[dict]) -> None:
    for record in audit_trail:
        db.add(
            AgentRun(
                client_id=client_id,
                run_id=run_id,
                node_name=record.get("node_name"),
                started_at=_parse_iso(record.get("started_at")),
                completed_at=_parse_iso(record.get("completed_at")),
                output_snapshot={"summary": record.get("summary")},
                status=record.get("status", "success"),
                error_detail=record.get("error_detail"),
            )
        )
    db.commit()


def _persist_report(db: Session, client_id: int, run_id: str, result: dict) -> Optional[Report]:
    final_report = result.get("final_report")
    if not final_report:
        return None
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
        },
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def _run_response(client_id: int, run_id: str, result: dict, db: Session) -> dict:
    _persist_audit_trail(db, client_id, run_id, result.get("audit_trail", []))

    if "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
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
        "final_report": result.get("final_report"),
        "report_id": report.id if report else None,
    }


@app.post("/api/v1/clients/{client_id}/run", dependencies=[Depends(require_api_key)])
def trigger_run(client_id: int, db: Session = Depends(get_db)):
    client = db.query(ClientProfile).filter(ClientProfile.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    try:
        run = run_client_graph(client_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _run_response(client_id, run["run_id"], run["result"], db)


@app.post("/api/v1/runs/{run_id}/approve", dependencies=[Depends(require_api_key)])
def approve_run(run_id: str, req: ApprovalDecision, db: Session = Depends(get_db)):
    approval = db.query(Approval).filter(Approval.run_id == run_id).order_by(Approval.id.desc()).first()
    if not approval:
        raise HTTPException(status_code=404, detail="No pending approval for this run_id")

    approval.decided_at = datetime.utcnow()
    approval.decided_by = req.decided_by
    approval.decision = "approved" if req.approved else "rejected"
    approval.notes = req.notes or approval.notes
    db.commit()

    try:
        resumed = resume_client_graph(run_id, approved=req.approved)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
