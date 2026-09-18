"""MySQL analytical lakehouse service for snapshots, outcomes, and audit records.

Replaces BigQuery with a containerized MySQL analytical store for:
- Historical portfolio snapshots and time series performance.
- Recommendation outcome scorecards and alpha tracking.
- Long-term regulatory audit event replication.
- Aggregated multi-tenant analytics and reporting.

Fails open: if MySQL analytics is disabled (MYSQL_ANALYTICS_ENABLED=false),
all calls degrade cleanly to no-ops without interrupting transactional workflows.
"""

import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from config import settings
from logging_setup import get_logger

logger = get_logger(__name__)

Base = declarative_base()

# --- MySQL Analytical Tables -------------------------------------------------


class AnalyticalPortfolioSnapshot(Base):
    __tablename__ = "analytical_portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date = Column(Date, nullable=False, index=True)
    client_id = Column(Integer, nullable=False, index=True)
    org_id = Column(Integer, nullable=False, index=True)
    total_value = Column(Numeric(18, 2), nullable=False)
    cash_value = Column(Numeric(18, 2), nullable=False)
    invested_value = Column(Numeric(18, 2), nullable=False)
    positions_count = Column(Integer, nullable=False)
    holdings_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_aps_org_client_date", "org_id", "client_id", "snapshot_date"),
    )


class AnalyticalRecommendationOutcome(Base):
    __tablename__ = "analytical_recommendation_outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    org_id = Column(Integer, nullable=False, index=True)
    client_id = Column(Integer, nullable=False, index=True)
    run_id = Column(String(64), nullable=False, index=True)
    ticker = Column(String(16), nullable=False, index=True)
    action = Column(String(16), nullable=False)
    allocated_usd = Column(Numeric(18, 2), nullable=False)
    price_at_recommendation = Column(Numeric(18, 4), nullable=True)
    price_current = Column(Numeric(18, 4), nullable=True)
    return_pct = Column(Float, nullable=True)
    benchmark_return_pct = Column(Float, nullable=True)
    excess_return_pct = Column(Float, nullable=True)
    evaluated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_aro_org_ticker", "org_id", "ticker"),
    )


class AnalyticalAuditLog(Base):
    __tablename__ = "analytical_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    org_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=True)
    actor_label = Column(String(128), nullable=True)
    action = Column(String(64), nullable=False, index=True)
    entity_type = Column(String(64), nullable=True)
    entity_id = Column(String(64), nullable=True)
    detail_json = Column(Text, nullable=True)
    occurred_at = Column(DateTime, nullable=False, index=True)
    hash = Column(String(64), nullable=False)
    prev_hash = Column(String(64), nullable=False)

    __table_args__ = (
        Index("ix_aal_org_occurred", "org_id", "occurred_at"),
    )


class AnalyticalMarketData(Base):
    __tablename__ = "analytical_market_data_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(16), nullable=False, index=True)
    price_date = Column(Date, nullable=False, index=True)
    close_price = Column(Numeric(18, 4), nullable=False)
    volume = Column(Numeric(20, 2), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_amd_ticker_date", "ticker", "price_date", unique=True),
    )


_engine = None
_SessionFactory = None
_tables_initialized = False


def is_available() -> bool:
    """True if MySQL analytical store is configured and enabled."""
    return getattr(settings, "MYSQL_ANALYTICS_ENABLED", False)


def get_engine():
    """Lazily construct SQLAlchemy engine for MySQL analytical database."""
    global _engine, _SessionFactory
    if not is_available():
        return None
    if _engine is None:
        url = getattr(settings, "MYSQL_ANALYTICS_URL", "")
        if not url:
            return None
        try:
            _engine = create_engine(
                url,
                pool_size=5,
                max_overflow=10,
                pool_recycle=3600,
                pool_pre_ping=True,
            )
            _SessionFactory = sessionmaker(bind=_engine)
            ensure_tables()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialize MySQL analytics engine: %s", exc)
            _engine = None
    return _engine


def get_session():
    engine = get_engine()
    if engine is None or _SessionFactory is None:
        return None
    return _SessionFactory()


def ensure_tables():
    """Ensure analytical tables exist in the target MySQL database."""
    global _tables_initialized
    if _tables_initialized or _engine is None:
        return
    try:
        Base.metadata.create_all(_engine)
        _tables_initialized = True
        logger.info("Ensured MySQL analytical tables exist.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not create MySQL analytical tables: %s", exc)


def status() -> Dict[str, Any]:
    """Diagnostic status for /health and status probes."""
    available = is_available()
    engine = get_engine() if available else None
    connected = False
    counts = {}
    if engine:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                connected = True
                for tbl in (
                    AnalyticalPortfolioSnapshot.__tablename__,
                    AnalyticalRecommendationOutcome.__tablename__,
                    AnalyticalAuditLog.__tablename__,
                ):
                    try:
                        res = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
                        counts[tbl] = res or 0
                    except Exception:
                        counts[tbl] = 0
        except Exception as exc:  # noqa: BLE001
            connected = False
            logger.debug("MySQL analytics connection check failed: %s", exc)

    return {
        "enabled": available,
        "connected": connected,
        "driver": "MySQL / SQLAlchemy",
        "url": getattr(settings, "MYSQL_ANALYTICS_URL", "").split("@")[-1] if available else None,
        "tables": counts,
    }


# --- Streaming Ingestion APIs ------------------------------------------------


def stream_portfolio_snapshot(snapshot: Any) -> None:
    """Stream a single portfolio snapshot into the analytical store."""
    if not is_available():
        return
    session = get_session()
    if not session:
        return
    try:
        holdings_data = getattr(snapshot, "holdings_json", None) or getattr(snapshot, "holdings", None)
        record = AnalyticalPortfolioSnapshot(
            snapshot_date=getattr(snapshot, "as_of_date", None) or getattr(snapshot, "snapshot_date", date.today()),
            client_id=getattr(snapshot, "client_id", 0),
            org_id=getattr(snapshot, "org_id", 1),
            total_value=getattr(snapshot, "total_value", 0.0),
            cash_value=getattr(snapshot, "cash_value", 0.0),
            invested_value=getattr(snapshot, "invested_value", 0.0),
            positions_count=getattr(snapshot, "positions_count", 0),
            holdings_json=json.dumps(holdings_data, default=str) if isinstance(holdings_data, (dict, list)) else str(holdings_data or "{}"),
        )
        session.add(record)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.debug("Failed streaming portfolio snapshot to MySQL analytics: %s", exc)
    finally:
        session.close()


def stream_recommendation_outcome(outcome: Any) -> None:
    """Stream a recommendation outcome into the analytical store."""
    if not is_available():
        return
    session = get_session()
    if not session:
        return
    try:
        record = AnalyticalRecommendationOutcome(
            org_id=getattr(outcome, "org_id", 1),
            client_id=getattr(outcome, "client_id", 0),
            run_id=str(getattr(outcome, "run_id", "")),
            ticker=str(getattr(outcome, "ticker", "")),
            action=str(getattr(outcome, "action", "BUY")),
            allocated_usd=getattr(outcome, "allocated_usd", 0.0),
            price_at_recommendation=getattr(outcome, "price_at_recommendation", None),
            price_current=getattr(outcome, "price_current", None),
            return_pct=getattr(outcome, "return_pct", None),
            benchmark_return_pct=getattr(outcome, "benchmark_return_pct", None),
            excess_return_pct=getattr(outcome, "excess_return_pct", None),
            evaluated_at=getattr(outcome, "evaluated_at", None),
        )
        session.add(record)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.debug("Failed streaming outcome to MySQL analytics: %s", exc)
    finally:
        session.close()


def stream_audit_log(event: Any) -> None:
    """Stream an audit event into the analytical store."""
    if not is_available():
        return
    session = get_session()
    if not session:
        return
    try:
        detail = getattr(event, "detail", {})
        record = AnalyticalAuditLog(
            org_id=getattr(event, "org_id", 1),
            user_id=getattr(event, "user_id", None),
            actor_label=getattr(event, "actor_label", None),
            action=str(getattr(event, "action", "")),
            entity_type=getattr(event, "entity_type", None),
            entity_id=str(getattr(event, "entity_id", "") or "") or None,
            detail_json=json.dumps(detail, default=str) if isinstance(detail, (dict, list)) else str(detail or "{}"),
            occurred_at=getattr(event, "occurred_at", datetime.utcnow()),
            hash=str(getattr(event, "hash", "")),
            prev_hash=str(getattr(event, "prev_hash", "")),
        )
        session.add(record)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.debug("Failed streaming audit log to MySQL analytics: %s", exc)
    finally:
        session.close()


# --- Batch Synchronization ---------------------------------------------------


def sync_all(db: Any, org_id: Optional[int] = None) -> Dict[str, int]:
    """Sync existing snapshots, outcomes, and audit logs into MySQL analytics."""
    if not is_available():
        return {"snapshots": 0, "outcomes": 0, "audit_logs": 0}
    session = get_session()
    if not session:
        return {"snapshots": 0, "outcomes": 0, "audit_logs": 0}

    counts = {"snapshots": 0, "outcomes": 0, "audit_logs": 0}
    try:
        from db import AuditEvent, PortfolioSnapshot, RecommendationOutcome

        # 1. Snapshots
        snap_query = db.query(PortfolioSnapshot)
        if org_id is not None:
            snap_query = snap_query.filter(PortfolioSnapshot.org_id == org_id)
        for s in snap_query.all():
            stream_portfolio_snapshot(s)
            counts["snapshots"] += 1

        # 2. Outcomes
        out_query = db.query(RecommendationOutcome)
        if org_id is not None:
            out_query = out_query.filter(RecommendationOutcome.org_id == org_id)
        for o in out_query.all():
            stream_recommendation_outcome(o)
            counts["outcomes"] += 1

        # 3. Audit
        audit_query = db.query(AuditEvent)
        if org_id is not None:
            audit_query = audit_query.filter(AuditEvent.org_id == org_id)
        for a in audit_query.all():
            stream_audit_log(a)
            counts["audit_logs"] += 1

    except Exception as exc:  # noqa: BLE001
        logger.warning("Sync to MySQL analytics encountered an error: %s", exc)
    finally:
        session.close()

    return counts


# --- Analytics Query APIs ----------------------------------------------------


def query_client_performance_history(client_id: int, days: int = 90) -> List[Dict[str, Any]]:
    """Retrieve historical performance snapshots for a client."""
    if not is_available():
        return []
    session = get_session()
    if not session:
        return []
    try:
        rows = (
            session.query(AnalyticalPortfolioSnapshot)
            .filter(AnalyticalPortfolioSnapshot.client_id == client_id)
            .order_by(AnalyticalPortfolioSnapshot.snapshot_date.asc())
            .limit(days)
            .all()
        )
        return [
            {
                "snapshot_date": r.snapshot_date.isoformat(),
                "total_value": float(r.total_value),
                "cash_value": float(r.cash_value),
                "invested_value": float(r.invested_value),
                "positions_count": r.positions_count,
            }
            for r in rows
        ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed querying client performance from MySQL analytics: %s", exc)
        return []
    finally:
        session.close()


def query_audit_trail(org_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve immutable audit records for regulatory oversight."""
    if not is_available():
        return []
    session = get_session()
    if not session:
        return []
    try:
        rows = (
            session.query(AnalyticalAuditLog)
            .filter(AnalyticalAuditLog.org_id == org_id)
            .order_by(AnalyticalAuditLog.occurred_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "action": r.action,
                "actor_label": r.actor_label,
                "entity_type": r.entity_type,
                "entity_id": r.entity_id,
                "occurred_at": r.occurred_at.isoformat(),
                "hash": r.hash,
                "prev_hash": r.prev_hash,
            }
            for r in rows
        ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed querying audit trail from MySQL analytics: %s", exc)
        return []
    finally:
        session.close()
