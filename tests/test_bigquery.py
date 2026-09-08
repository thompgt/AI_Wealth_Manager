"""Tests for the BigQuery analytical lakehouse integration.

Verifies:
1. Safe degradation: When BIGQUERY_ENABLED=False (the default), all calls are clean
   no-ops that do not raise exceptions, require GCP credentials, or touch the network.
2. Serialization and format helpers: Correct transformation of ORM objects and dicts
   into JSON-safe BigQuery schemas with date partitioning and clustering keys.
3. Schema setup: Idempotent dataset and table provisioning with day partitioning.
4. Streaming row ingestion: Proper call patterns and error resilience with a mock client.
5. Synchronization / backfill: Syncing historical records from SQLAlchemy models to BigQuery.
6. API routes: RBAC enforcement and status reporting on /api/v1/analytics/bigquery/status
   and /api/v1/maintenance/sync-bigquery, as well as /system/status diagnostics.
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import sys
from types import ModuleType
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from config import settings
from db import AuditEvent, Organization, PortfolioSnapshot, RecommendationOutcome, SessionLocal, User, init_db
from security import hash_password
import server
from services import bigquery_service

PASSWORD = "test-password-12345"
FIRM_SLUG = f"bq-test-firm-{uuid4().hex[:8]}"


@pytest.fixture
def mock_google_bigquery(monkeypatch):
    """Provide a mock google.cloud.bigquery module if not installed in the local environment."""
    mock_bq = ModuleType("google.cloud.bigquery")
    mock_bq.Dataset = MagicMock(side_effect=lambda dataset_id: MagicMock(table_id=dataset_id, dataset_id=dataset_id))
    mock_bq.Table = MagicMock(side_effect=lambda table_id, schema=None: MagicMock(table_id=table_id, schema=schema))
    mock_bq.SchemaField = MagicMock(side_effect=lambda name, type, mode="NULLABLE": MagicMock(name=name, type=type, mode=mode))
    mock_tp = MagicMock()
    mock_tp.DAY = "DAY"
    mock_bq.TimePartitioningType = mock_tp
    mock_bq.TimePartitioning = MagicMock()

    try:
        import google.cloud
        monkeypatch.setattr(google.cloud, "bigquery", mock_bq, raising=False)
    except ImportError:
        pass
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", mock_bq)
    return mock_bq


@pytest.fixture(scope="module")
def bq_test_identities():
    """Seed test identities for testing authenticated BigQuery endpoints."""
    init_db()
    db = SessionLocal()
    try:
        org = Organization(name=FIRM_SLUG, slug=FIRM_SLUG)
        db.add(org)
        db.flush()

        admin_user = User(
            org_id=org.id,
            email=f"admin@{FIRM_SLUG}.example",
            full_name="Admin User",
            password_hash=hash_password(PASSWORD),
            role="admin",
        )
        advisor_user = User(
            org_id=org.id,
            email=f"advisor@{FIRM_SLUG}.example",
            full_name="Advisor User",
            password_hash=hash_password(PASSWORD),
            role="advisor",
        )
        db.add_all([admin_user, advisor_user])
        db.commit()
        return {"org_id": org.id, "admin_email": admin_user.email, "advisor_email": advisor_user.email}
    finally:
        db.close()


@pytest.fixture(scope="module")
def api():
    init_db()
    with TestClient(server.app) as test_client:
        yield test_client


def _login(api_client, org_slug, email):
    resp = api_client.post(
        "/api/v1/auth/login",
        json={"org_slug": org_slug, "email": email, "password": PASSWORD},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


# --- 1. Disabled & Graceful Degradation Tests ---------------------------------


def test_bigquery_disabled_by_default():
    """BIGQUERY_ENABLED is False by default so local dev & CI run completely offline."""
    assert not settings.BIGQUERY_ENABLED
    assert not bigquery_service.is_available()
    assert bigquery_service.get_bigquery_client() is None

    st = bigquery_service.status()
    assert st["enabled"] is False
    assert st["available"] is False
    assert st["connected"] is False
    assert st["dataset"] == settings.BIGQUERY_DATASET
    assert st["location"] == settings.BIGQUERY_LOCATION


def test_streaming_degrades_gracefully_when_disabled():
    """Streaming methods return False without throwing when BigQuery is disabled."""
    assert bigquery_service.stream_portfolio_snapshot({"client_id": 1, "market_value": 1000}) is False
    assert bigquery_service.stream_recommendation_outcomes([{"symbol": "AAPL", "entry_price": 150}]) is False
    assert bigquery_service.stream_audit_log({"action": "test", "entity_type": "user"}) is False

    init_db()
    db = SessionLocal()
    try:
        counts = bigquery_service.sync_all(db)
        assert counts == {"snapshots": 0, "outcomes": 0, "audit_logs": 0}
    finally:
        db.close()


# --- 2. Serialization and Format Helpers Tests -------------------------------


def test_to_json_value():
    """Verify JSON normalization handles primitives, dates, Decimals, dicts, and lists."""
    assert bigquery_service._to_json_value(None) is None
    assert bigquery_service._to_json_value(123) == 123
    assert bigquery_service._to_json_value(45.67) == 45.67
    assert bigquery_service._to_json_value("hello") == "hello"
    assert bigquery_service._to_json_value(Decimal("99.95")) == "99.95"

    d = date(2026, 9, 8)
    assert bigquery_service._to_json_value(d) == "2026-09-08"

    dt = datetime(2026, 9, 8, 12, 0, 0)
    assert bigquery_service._to_json_value(dt) == "2026-09-08T12:00:00"

    nested = {"date": d, "amount": Decimal("10.5"), "items": [dt, 1]}
    assert bigquery_service._to_json_value(nested) == {
        "date": "2026-09-08",
        "amount": "10.5",
        "items": ["2026-09-08T12:00:00", 1],
    }


def test_format_snapshot_from_dict_and_object():
    """Ensure _format_snapshot handles both dictionaries and ORM-like objects."""
    today = date(2026, 9, 8)
    dict_snap = {
        "client_id": 42,
        "org_id": 1,
        "as_of": today,
        "market_value": Decimal("150000.00"),
        "cash_balance": Decimal("25000.00"),
        "external_flow_today": Decimal("5000.00"),
        "gross_return_today": 0.0125,
        "is_reconstructed": False,
    }
    formatted = bigquery_service._format_snapshot(dict_snap)
    assert formatted["snapshot_id"] == f"snap-42-{today.isoformat()}"
    assert formatted["client_id"] == 42
    assert formatted["org_id"] == 1
    assert formatted["as_of_date"] == "2026-09-08"
    assert formatted["market_value"] == "150000.00"
    assert formatted["cash_balance"] == "25000.00"
    assert formatted["external_flow_today"] == "5000.00"
    assert formatted["gross_return_today"] == 0.0125
    assert formatted["is_reconstructed"] is False

    # ORM-like object with cash_value and net_flow aliases
    class FakeORM:
        client_id = 99
        org_id = 2
        as_of = date(2026, 9, 7)
        market_value = Decimal("200000.00")
        cash_value = Decimal("10000.00")
        net_flow = Decimal("0.00")
        gross_return_today = None
        is_reconstructed = True
        created_at = datetime(2026, 9, 7, 18, 0, 0, tzinfo=timezone.utc)

    orm_formatted = bigquery_service._format_snapshot(FakeORM())
    assert orm_formatted["snapshot_id"] == "snap-99-2026-09-07"
    assert orm_formatted["cash_balance"] == "10000.00"
    assert orm_formatted["external_flow_today"] == "0.00"
    assert orm_formatted["gross_return_today"] is None
    assert orm_formatted["is_reconstructed"] is True


def test_format_outcome_from_dict_and_object():
    """Ensure _format_outcome maps recommendation outcomes and computes eval_due_date."""
    rec_time = datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    dict_out = {
        "id": 101,
        "org_id": 1,
        "client_id": 42,
        "run_id": "run-abc",
        "symbol": "msft",
        "recommended_at": rec_time,
        "horizon_days": 90,
        "entry_price": Decimal("400.00"),
        "exit_price": Decimal("440.00"),
        "return_pct": 0.10,
        "benchmark_return_pct": 0.04,
        "excess_return_pct": 0.06,
        "beat_benchmark": True,
        "evaluated_at": datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc),
    }
    formatted = bigquery_service._format_outcome(dict_out)
    assert formatted["outcome_id"] == "101"
    assert formatted["ticker"] == "MSFT"
    assert formatted["horizon_days"] == 90
    assert formatted["eval_due_date"] == (rec_time + timedelta(days=90)).date().isoformat()
    assert formatted["base_price"] == "400.00"
    assert formatted["horizon_price"] == "440.00"
    assert formatted["asset_return"] == 0.10
    assert formatted["excess_return"] == 0.06
    assert formatted["hit"] is True


def test_format_audit_log():
    """Ensure _format_audit_log serializes detail to JSON string and maps hash fields."""
    now = datetime(2026, 9, 8, 14, 0, 0, tzinfo=timezone.utc)
    event_dict = {
        "id": 55,
        "org_id": 1,
        "user_id": 12,
        "action": "rebalance.approved",
        "entity_type": "rebalance_plan",
        "entity_id": "plan-789",
        "occurred_at": now,
        "detail": {"orders_count": 3, "drift": 0.05},
        "prev_hash": "abc123hash",
        "hash": "def456hash",
    }
    formatted = bigquery_service._format_audit_log(event_dict)
    assert formatted["event_id"] == "55"
    assert formatted["action"] == "rebalance.approved"
    assert formatted["previous_hash"] == "abc123hash"
    assert formatted["current_hash"] == "def456hash"
    assert '"orders_count": 3' in formatted["detail_json"]


# --- 3. Mock Client Ingestion & Schema Tests ----------------------------------


def test_ensure_dataset_and_tables_with_mock_client(mock_google_bigquery):
    """Verify that ensure_dataset_and_tables sets up dataset and tables with partitioning."""
    mock_client = MagicMock()
    mock_client.project = "test-project-123"

    success = bigquery_service.ensure_dataset_and_tables(client=mock_client)
    assert success is True
    assert mock_client.create_dataset.call_count == 1
    assert mock_client.create_table.call_count == 3

    # Verify table configurations passed to create_table
    created_tables = [call[0][0] for call in mock_client.create_table.call_args_list]
    table_ids = [t.table_id for t in created_tables]
    assert f"test-project-123.{settings.BIGQUERY_DATASET}.portfolio_snapshots" in table_ids
    assert f"test-project-123.{settings.BIGQUERY_DATASET}.recommendation_outcomes" in table_ids
    assert f"test-project-123.{settings.BIGQUERY_DATASET}.audit_logs" in table_ids


def test_stream_rows_with_mock_client():
    """Verify stream_rows behavior on success, batch empty, and insert errors."""
    mock_client = MagicMock()
    mock_client.project = "test-project-123"

    # 1. Empty rows -> True without calling client
    assert bigquery_service.stream_rows("some_table", [], client=mock_client) is True
    assert mock_client.insert_rows_json.call_count == 0

    # 2. Successful batch
    mock_client.insert_rows_json.return_value = []
    rows = [{"col1": "val1"}]
    assert bigquery_service.stream_rows("portfolio_snapshots", rows, client=mock_client) is True
    mock_client.insert_rows_json.assert_called_once_with(
        f"test-project-123.{settings.BIGQUERY_DATASET}.portfolio_snapshots",
        rows,
    )

    # 3. Insert errors from BigQuery API
    mock_client.insert_rows_json.return_value = [{"index": 0, "errors": [{"reason": "invalid"}]}]
    assert bigquery_service.stream_rows("portfolio_snapshots", rows, client=mock_client) is False

    # 4. Exception handling
    mock_client.insert_rows_json.side_effect = RuntimeError("network disconnected")
    assert bigquery_service.stream_rows("portfolio_snapshots", rows, client=mock_client) is False


def test_sync_all_with_mock_client(bq_test_identities, mock_google_bigquery, monkeypatch):
    """Verify sync_all extracts records and streams them to the respective tables."""
    from db import ClientProfile

    mock_client = MagicMock()
    mock_client.project = "test-project-123"
    mock_client.insert_rows_json.return_value = []

    # Monkeypatch is_available to return True so sync_all proceeds with the provided client
    monkeypatch.setattr(bigquery_service, "is_available", lambda: True)

    db = SessionLocal()
    try:
        # Create a test organization and client
        org = Organization(name="Sync Test Org", slug=f"sync-org-{uuid4().hex[:6]}")
        db.add(org)
        db.flush()

        client = ClientProfile(
            org_id=org.id,
            name="Sync Test Client",
            email="sync@example.com",
            risk_tolerance="moderate",
        )
        db.add(client)
        db.flush()

        # Add a test snapshot
        snap = PortfolioSnapshot(
            org_id=org.id,
            client_id=client.id,
            as_of=datetime.now(timezone.utc),
            market_value=Decimal("100000.00"),
            cash_value=Decimal("10000.00"),
            net_flow=Decimal("0.00"),
            is_reconstructed=False,
        )
        # Add a test recommendation outcome
        outcome = RecommendationOutcome(
            org_id=org.id,
            client_id=client.id,
            run_id="sync-run-1",
            symbol="NVDA",
            horizon_days=90,
            recommended_at=datetime.now(timezone.utc),
            entry_price=Decimal("120.00"),
            status="pending",
        )
        # Add a test audit event
        audit = AuditEvent(
            org_id=org.id,
            user_id=None,
            action="system.test_sync",
            entity_type="system",
            entity_id="1",
            occurred_at=datetime.now(timezone.utc),
            detail={"source": "test"},
            prev_hash="0" * 64,
            hash="1" * 64,
        )
        db.add_all([snap, outcome, audit])
        db.commit()

        counts = bigquery_service.sync_all(db, org_id=org.id, client=mock_client)
        assert counts["snapshots"] >= 1
        assert counts["outcomes"] >= 1
        assert counts["audit_logs"] >= 1

        # insert_rows_json should have been called for all 3 tables
        assert mock_client.insert_rows_json.call_count >= 3
    finally:
        db.close()


# --- 4. API Endpoints & Role-Based Access Tests ------------------------------


def test_system_status_includes_bigquery(api, bq_test_identities):
    """The administrative /api/v1/system/status endpoint reports BigQuery status."""
    token = _login(api, FIRM_SLUG, bq_test_identities["admin_email"])
    resp = api.get("/api/v1/system/status", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "bigquery" in data
    assert data["bigquery"]["enabled"] is False
    assert data["bigquery"]["dataset"] == settings.BIGQUERY_DATASET


def test_bigquery_analytics_status_rbac(api, bq_test_identities):
    """GET /api/v1/analytics/bigquery/status requires analytics:sync (admin only)."""
    # 1. Unauthenticated -> 401
    unauth = api.get("/api/v1/analytics/bigquery/status")
    assert unauth.status_code == 401

    # 2. Advisor -> 403 Forbidden
    advisor_token = _login(api, FIRM_SLUG, bq_test_identities["advisor_email"])
    forbidden = api.get(
        "/api/v1/analytics/bigquery/status",
        headers={"Authorization": f"Bearer {advisor_token}"},
    )
    assert forbidden.status_code == 403

    # 3. Admin -> 200 OK
    admin_token = _login(api, FIRM_SLUG, bq_test_identities["admin_email"])
    ok = api.get(
        "/api/v1/analytics/bigquery/status",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert ok.status_code == 200
    res = ok.json()
    assert res["enabled"] is False
    assert res["dataset"] == settings.BIGQUERY_DATASET


def test_sync_bigquery_endpoint_rbac_and_behavior(api, bq_test_identities, monkeypatch):
    """POST /api/v1/maintenance/sync-bigquery enforces admin access and validates enable flag."""
    advisor_token = _login(api, FIRM_SLUG, bq_test_identities["advisor_email"])
    admin_token = _login(api, FIRM_SLUG, bq_test_identities["admin_email"])

    # 1. Advisor -> 403
    resp_adv = api.post(
        "/api/v1/maintenance/sync-bigquery",
        headers={"Authorization": f"Bearer {advisor_token}"},
    )
    assert resp_adv.status_code == 403

    # 2. Admin when BIGQUERY_ENABLED is False -> 400 Bad Request
    resp_dis = api.post(
        "/api/v1/maintenance/sync-bigquery",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp_dis.status_code == 400
    assert "BIGQUERY_ENABLED=false" in resp_dis.json()["detail"]

    # 3. Admin when BIGQUERY_ENABLED is True -> calls sync_all and returns 200
    monkeypatch.setattr(settings, "BIGQUERY_ENABLED", True)
    monkeypatch.setattr(
        bigquery_service,
        "sync_all",
        lambda db, org_id: {"snapshots": 12, "outcomes": 5, "audit_logs": 30},
    )
    resp_en = api.post(
        "/api/v1/maintenance/sync-bigquery",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp_en.status_code == 200
    body = resp_en.json()
    assert body["status"] == "synced"
    assert body["counts"] == {"snapshots": 12, "outcomes": 5, "audit_logs": 30}
