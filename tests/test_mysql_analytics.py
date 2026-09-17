"""Unit tests for MySQL analytical store and replication service."""

from datetime import date, datetime
import pytest
from services import mysql_analytics
from config import settings


def test_mysql_analytics_availability():
    """Verify is_available returns boolean without error."""
    avail = mysql_analytics.is_available()
    assert isinstance(avail, bool)


def test_mysql_analytics_status():
    """Verify status probe returns structured diagnostic dictionary."""
    st = mysql_analytics.status()
    assert "enabled" in st
    assert "connected" in st
    assert "driver" in st
    assert "tables" in st


def test_mysql_analytics_disabled(monkeypatch):
    """When disabled, streaming and sync degrade to no-ops."""
    monkeypatch.setattr(settings, "MYSQL_ANALYTICS_ENABLED", False)

    assert mysql_analytics.is_available() is False
    assert mysql_analytics.get_engine() is None

    # These should not raise
    mysql_analytics.stream_portfolio_snapshot(None)
    mysql_analytics.stream_recommendation_outcome(None)
    mysql_analytics.stream_audit_log(None)

    res = mysql_analytics.sync_all(None)
    assert res == {"snapshots": 0, "outcomes": 0, "audit_logs": 0}

    assert mysql_analytics.query_client_performance_history(1) == []
    assert mysql_analytics.query_audit_trail(1) == []


def test_mysql_analytics_sqlite_compatible_test_store(tmp_path, monkeypatch):
    """Verify schema and serialization against a local SQLite-backed test analytical store."""
    test_db_url = f"sqlite:///{tmp_path}/test_analytics.db"
    monkeypatch.setattr(settings, "MYSQL_ANALYTICS_ENABLED", True)
    monkeypatch.setattr(settings, "MYSQL_ANALYTICS_URL", test_db_url)

    # Force re-init with test URL
    mysql_analytics._engine = None
    mysql_analytics._SessionFactory = None
    mysql_analytics._tables_initialized = False

    engine = mysql_analytics.get_engine()
    assert engine is not None

    class MockSnapshot:
        snapshot_date = date.today()
        client_id = 42
        org_id = 1
        total_value = 150000.0
        cash_value = 25000.0
        invested_value = 125000.0
        positions_count = 4
        holdings = {"AAPL": 50000.0, "MSFT": 75000.0}

    class MockOutcome:
        org_id = 1
        client_id = 42
        run_id = "test-run-1"
        ticker = "JNJ"
        action = "BUY"
        allocated_usd = 20000.0
        price_at_recommendation = 150.0
        price_current = 155.0
        return_pct = 0.0333
        benchmark_return_pct = 0.012
        excess_return_pct = 0.0213
        evaluated_at = datetime.utcnow()

    class MockAuditEvent:
        org_id = 1
        user_id = 10
        actor_label = "test-analyst"
        action = "portfolio.rebalanced"
        entity_type = "client"
        entity_id = "42"
        detail = {"reason": "risk_alignment"}
        occurred_at = datetime.utcnow()
        hash = "a" * 64
        prev_hash = "0" * 64

    # Test streaming
    mysql_analytics.stream_portfolio_snapshot(MockSnapshot())
    mysql_analytics.stream_recommendation_outcome(MockOutcome())
    mysql_analytics.stream_audit_log(MockAuditEvent())

    # Test query
    history = mysql_analytics.query_client_performance_history(42)
    assert len(history) == 1
    assert history[0]["total_value"] == 150000.0
    assert history[0]["positions_count"] == 4

    audit_records = mysql_analytics.query_audit_trail(1)
    assert len(audit_records) == 1
    assert audit_records[0]["action"] == "portfolio.rebalanced"
    assert audit_records[0]["entity_id"] == "42"

    # Reset globals
    mysql_analytics._engine = None
    mysql_analytics._SessionFactory = None
    mysql_analytics._tables_initialized = False
