"""API-layer tests.

The server had no test coverage at all, which is how the duplicate
audit-trail writes, the unvalidated risk-tolerance field, and the missing
table creation all survived. These exercise the HTTP surface directly with
the graph itself stubbed out -- graph behaviour is covered by
test_integration.py.
"""

import pytest
from fastapi.testclient import TestClient

import server
from config import settings
from db import AgentRun, Approval, Report, SessionLocal, init_db

HEADERS = {"X-API-Key": settings.API_AUTH_KEY}


@pytest.fixture
def client():
    init_db()
    with TestClient(server.app) as test_client:
        yield test_client


def _make_client(client, **overrides):
    payload = {
        "name": "Test Client",
        "email": "t@example.com",
        "age": 45,
        "risk_tolerance": "Moderate",
        "time_horizon_years": 20,
        "goals": ["retirement"],
        "net_worth": 100000.0,
        "holdings": [{"symbol": "CASH", "quantity": 100000.0, "cost_basis": 1.0}],
    }
    payload.update(overrides)
    return client.post("/api/v1/clients", json=payload, headers=HEADERS)


# --- auth -------------------------------------------------------------------

def test_health_needs_no_api_key(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # The probe must surface whether the AI layer is live, not just that the
    # process is up.
    assert body["llm_enabled"] is False
    assert "stock_research" in body["agents"]


def test_endpoints_reject_a_missing_api_key(client):
    assert client.get("/api/v1/clients").status_code == 401


def test_endpoints_reject_a_wrong_api_key(client):
    response = client.get("/api/v1/clients", headers={"X-API-Key": "nope"})
    assert response.status_code == 401


# --- validation -------------------------------------------------------------

def test_rejects_an_unrecognized_risk_tolerance(client):
    """A free-form risk tolerance used to be accepted and then silently fall
    through to the Moderate defaults in every downstream agent."""
    response = _make_client(client, risk_tolerance="Balanced")
    assert response.status_code == 422


def test_rejects_a_negative_holding_quantity(client):
    response = _make_client(
        client, holdings=[{"symbol": "AAPL", "quantity": -5.0, "cost_basis": 10.0}]
    )
    assert response.status_code == 422


def test_rejects_a_negative_net_worth(client):
    assert _make_client(client, net_worth=-1.0).status_code == 422


def test_creates_and_reads_back_a_client(client):
    created = _make_client(client)
    assert created.status_code == 200
    client_id = created.json()["id"]

    fetched = client.get(f"/api/v1/clients/{client_id}", headers=HEADERS)
    assert fetched.status_code == 200
    assert fetched.json()["risk_tolerance"] == "Moderate"
    assert fetched.json()["holdings"][0]["symbol"] == "CASH"


def test_unknown_client_is_404(client):
    assert client.get("/api/v1/clients/999999", headers=HEADERS).status_code == 404


# --- audit trail dedupe -----------------------------------------------------

def test_audit_trail_is_not_duplicated_when_a_run_is_resumed():
    """Regression guard.

    LangGraph's resumed result carries the whole accumulated audit_trail,
    including every pre-interrupt record. Writing it verbatim on resume
    duplicated each node in agent_runs -- corrupting the one table whose
    entire job is being a faithful record of what ran.
    """
    db = SessionLocal()
    try:
        init_db()
        run_id = "dedupe-test-run"
        db.query(AgentRun).filter(AgentRun.run_id == run_id).delete()
        db.commit()

        first_pass = [
            {
                "node_name": "market_regime",
                "started_at": "2026-01-01T00:00:00+00:00",
                "completed_at": "2026-01-01T00:00:05+00:00",
                "status": "success",
                "summary": "first",
                "error_detail": None,
            }
        ]
        # What a resume actually returns: the original record plus new ones.
        resumed = first_pass + [
            {
                "node_name": "approval_gate",
                "started_at": "2026-01-01T00:01:00+00:00",
                "completed_at": "2026-01-01T00:01:01+00:00",
                "status": "success",
                "summary": "approved",
                "error_detail": None,
            }
        ]

        server._persist_audit_trail(db, 1, run_id, first_pass)
        server._persist_audit_trail(db, 1, run_id, resumed)

        rows = db.query(AgentRun).filter(AgentRun.run_id == run_id).all()
        assert len(rows) == 2, f"expected 2 audit rows, got {len(rows)}"
        assert sorted(r.node_name for r in rows) == ["approval_gate", "market_regime"]
    finally:
        db.close()


def test_report_is_written_once_per_run():
    db = SessionLocal()
    try:
        init_db()
        run_id = "report-once-run"
        db.query(Report).filter(Report.run_id == run_id).delete()
        db.commit()

        result = {"final_report": "hello", "suitability_result": {"approved": True}}
        first = server._persist_report(db, 1, run_id, result)
        second = server._persist_report(db, 1, run_id, result)

        assert first.id == second.id
        assert db.query(Report).filter(Report.run_id == run_id).count() == 1
    finally:
        db.close()


# --- approvals --------------------------------------------------------------

def test_approving_an_unknown_run_is_404(client):
    response = client.post(
        "/api/v1/runs/no-such-run/approve", json={"approved": True}, headers=HEADERS
    )
    assert response.status_code == 404


def test_approving_an_already_decided_run_is_409(client):
    db = SessionLocal()
    try:
        run_id = "already-decided-run"
        db.query(Approval).filter(Approval.run_id == run_id).delete()
        db.add(Approval(run_id=run_id, decision="approved", decided_by="someone"))
        db.commit()
    finally:
        db.close()

    response = client.post(
        f"/api/v1/runs/{run_id}/approve", json={"approved": True}, headers=HEADERS
    )
    assert response.status_code == 409
    assert "already approved" in response.json()["detail"]
