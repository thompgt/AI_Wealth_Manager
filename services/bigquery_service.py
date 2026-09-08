"""BigQuery service: analytical lakehouse for snapshots, outcomes, and audit records.

Designed to mirror the dual-engine architecture in `docs/GCP_AND_BIGQUERY_MIGRATION.md`:
Cloud SQL / PostgreSQL owns the transactional core (ACID, low-latency, row-level locks),
while BigQuery stores analytical time-series and regulatory audit records.

Rules:
1. When BIGQUERY_ENABLED is false (the default for local development and CI), every
   call degrades cleanly to a no-op without network round trips or exceptions.
2. Missing libraries (`google-cloud-bigquery`) or missing GCP credentials never crash
   the application or fail a run.
3. Partitioning (by DATE) and clustering (by ticker / org_id) are enforced by
   construction during schema creation.
"""

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from config import settings
from logging_setup import get_logger

logger = get_logger(__name__)

TABLE_SNAPSHOTS = "portfolio_snapshots"
TABLE_OUTCOMES = "recommendation_outcomes"
TABLE_AUDIT_LOGS = "audit_logs"
TABLE_MARKET_DATA = "market_data_history"

_client = None


def is_available() -> bool:
    """True if BigQuery is enabled and the client library is importable."""
    if not settings.BIGQUERY_ENABLED:
        return False
    try:
        import google.cloud.bigquery  # noqa: F401
        return True
    except ImportError:
        return False


def get_bigquery_client():
    """Lazily construct a BigQuery Client using application default credentials."""
    global _client
    if _client is not None:
        return _client
    if not is_available():
        return None

    try:
        from google.cloud import bigquery
        kwargs = {}
        if settings.BIGQUERY_PROJECT_ID:
            kwargs["project"] = settings.BIGQUERY_PROJECT_ID
        if settings.BIGQUERY_LOCATION:
            kwargs["location"] = settings.BIGQUERY_LOCATION
        _client = bigquery.Client(**kwargs)
        return _client
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not initialize BigQuery client: %s", exc)
        return None


def reset_client() -> None:
    """Reset cached client (useful for tests)."""
    global _client
    _client = None


# --- Serialization helpers ---------------------------------------------------


def _to_json_value(val: Any) -> Any:
    """Normalize types into JSON-compatible values for BigQuery insert_rows_json."""
    if val is None:
        return None
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return str(val)
    if isinstance(val, (int, float, bool, str)):
        return val
    if isinstance(val, dict):
        return {k: _to_json_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_to_json_value(x) for x in val]
    return str(val)


def _get_val(obj, *keys, default=None):
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        if hasattr(obj, key):
            val = getattr(obj, key)
            if val is not None:
                return val
    return default


def _format_snapshot(s) -> Dict[str, Any]:
    client_id = _get_val(s, "client_id", default=0)
    as_of = _get_val(s, "as_of", "as_of_date", default=date.today())
    as_of_str = as_of.date().isoformat() if isinstance(as_of, datetime) else _to_json_value(as_of)
    return {
        "snapshot_id": f"snap-{client_id}-{as_of_str}",
        "org_id": _get_val(s, "org_id", default=0),
        "client_id": client_id,
        "as_of_date": as_of_str,
        "market_value": _to_json_value(_get_val(s, "market_value", default=0)),
        "cash_balance": _to_json_value(_get_val(s, "cash_value", "cash_balance", default=0)),
        "external_flow_today": _to_json_value(_get_val(s, "net_flow", "external_flow_today", default=0)),
        "gross_return_today": (
            float(_get_val(s, "gross_return_today"))
            if _get_val(s, "gross_return_today") is not None
            else None
        ),
        "is_reconstructed": bool(_get_val(s, "is_reconstructed", default=False)),
        "captured_at": _to_json_value(_get_val(s, "created_at", "captured_at", default=datetime.now(timezone.utc))),
    }


def _format_outcome(o) -> Dict[str, Any]:
    rec_id = _get_val(o, "id")
    client_id = _get_val(o, "client_id", default=0)
    ticker = str(_get_val(o, "symbol", "ticker", default="")).upper()
    horizon = int(_get_val(o, "horizon_days", default=90))
    rec_at = _get_val(o, "recommended_at", default=datetime.now(timezone.utc))
    eval_due = _get_val(o, "eval_due_date")
    if not eval_due and isinstance(rec_at, (datetime, date)):
        from datetime import timedelta
        eval_due = (rec_at + timedelta(days=horizon))
        if isinstance(eval_due, datetime):
            eval_due = eval_due.date()

    return {
        "outcome_id": str(rec_id) if rec_id else f"out-{client_id}-{ticker}-{horizon}",
        "org_id": _get_val(o, "org_id", default=0),
        "client_id": client_id,
        "run_id": str(_get_val(o, "run_id", default="")),
        "ticker": ticker,
        "recommended_at": _to_json_value(rec_at),
        "horizon_days": horizon,
        "eval_due_date": _to_json_value(eval_due),
        "base_price": _to_json_value(_get_val(o, "entry_price", "base_price", default=0)),
        "horizon_price": _to_json_value(_get_val(o, "exit_price", "horizon_price")),
        "asset_return": (
            float(_get_val(o, "return_pct", "asset_return"))
            if _get_val(o, "return_pct", "asset_return") is not None
            else None
        ),
        "benchmark_return": (
            float(_get_val(o, "benchmark_return_pct", "benchmark_return"))
            if _get_val(o, "benchmark_return_pct", "benchmark_return") is not None
            else None
        ),
        "excess_return": (
            float(_get_val(o, "excess_return_pct", "excess_return"))
            if _get_val(o, "excess_return_pct", "excess_return") is not None
            else None
        ),
        "hit": (
            bool(_get_val(o, "beat_benchmark", "hit"))
            if _get_val(o, "beat_benchmark", "hit") is not None
            else None
        ),
        "evaluated_at": _to_json_value(_get_val(o, "evaluated_at")),
    }


def _format_audit_log(a) -> Dict[str, Any]:
    import json
    detail = _get_val(a, "detail", default={})
    detail_str = json.dumps(detail) if isinstance(detail, dict) else (str(detail) if detail else None)
    event_id = _get_val(a, "id")
    org_id = _get_val(a, "org_id", default=0)
    occurred_at = _get_val(a, "occurred_at", default=datetime.now(timezone.utc))
    return {
        "event_id": str(event_id) if event_id else f"audit-{org_id}-{occurred_at}",
        "org_id": org_id,
        "user_id": _get_val(a, "user_id"),
        "action": str(_get_val(a, "action", default="")),
        "entity_type": str(_get_val(a, "entity_type", default="")),
        "entity_id": str(_get_val(a, "entity_id", default="")),
        "occurred_at": _to_json_value(occurred_at),
        "detail_json": detail_str,
        "previous_hash": str(_get_val(a, "prev_hash", "previous_hash", default="")),
        "current_hash": str(_get_val(a, "hash", "current_hash", default="")),
    }


# --- Schema setup -------------------------------------------------------------


def ensure_dataset_and_tables(client=None) -> bool:
    """Create the analytical dataset and partitioned/clustered tables if missing."""
    client = client or get_bigquery_client()
    if client is None:
        return False

    try:
        from google.cloud import bigquery

        dataset_id = f"{client.project}.{settings.BIGQUERY_DATASET}"
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = settings.BIGQUERY_LOCATION
        client.create_dataset(dataset, exists_ok=True)

        # 1. Snapshots: partitioned by as_of_date, clustered by org_id, client_id
        t_snap = bigquery.Table(
            f"{dataset_id}.{TABLE_SNAPSHOTS}",
            schema=[
                bigquery.SchemaField("snapshot_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("org_id", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("client_id", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("as_of_date", "DATE", mode="REQUIRED"),
                bigquery.SchemaField("market_value", "NUMERIC", mode="REQUIRED"),
                bigquery.SchemaField("cash_balance", "NUMERIC", mode="REQUIRED"),
                bigquery.SchemaField("external_flow_today", "NUMERIC", mode="REQUIRED"),
                bigquery.SchemaField("gross_return_today", "FLOAT"),
                bigquery.SchemaField("is_reconstructed", "BOOLEAN", mode="REQUIRED"),
                bigquery.SchemaField("captured_at", "TIMESTAMP", mode="REQUIRED"),
            ],
        )
        t_snap.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="as_of_date"
        )
        t_snap.clustering_fields = ["org_id", "client_id"]
        client.create_table(t_snap, exists_ok=True)

        # 2. Recommendation Outcomes: partitioned by eval_due_date, clustered by ticker, horizon_days
        t_out = bigquery.Table(
            f"{dataset_id}.{TABLE_OUTCOMES}",
            schema=[
                bigquery.SchemaField("outcome_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("org_id", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("client_id", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("recommended_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("horizon_days", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("eval_due_date", "DATE", mode="REQUIRED"),
                bigquery.SchemaField("base_price", "NUMERIC", mode="REQUIRED"),
                bigquery.SchemaField("horizon_price", "NUMERIC"),
                bigquery.SchemaField("asset_return", "FLOAT"),
                bigquery.SchemaField("benchmark_return", "FLOAT"),
                bigquery.SchemaField("excess_return", "FLOAT"),
                bigquery.SchemaField("hit", "BOOLEAN"),
                bigquery.SchemaField("evaluated_at", "TIMESTAMP"),
            ],
        )
        t_out.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="eval_due_date"
        )
        t_out.clustering_fields = ["ticker", "horizon_days"]
        client.create_table(t_out, exists_ok=True)

        # 3. Audit Logs: partitioned by occurred_at, clustered by org_id, action
        t_audit = bigquery.Table(
            f"{dataset_id}.{TABLE_AUDIT_LOGS}",
            schema=[
                bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("org_id", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("user_id", "INTEGER"),
                bigquery.SchemaField("action", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("entity_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("entity_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("occurred_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("detail_json", "STRING"),
                bigquery.SchemaField("previous_hash", "STRING"),
                bigquery.SchemaField("current_hash", "STRING", mode="REQUIRED"),
            ],
        )
        t_audit.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="occurred_at"
        )
        t_audit.clustering_fields = ["org_id", "action"]
        client.create_table(t_audit, exists_ok=True)

        logger.info("Ensured BigQuery dataset and tables in %s", dataset_id)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed ensuring BigQuery schema: %s", exc)
        return False


# --- Streaming ingest methods -------------------------------------------------


def stream_rows(table_name: str, rows: Sequence[Dict[str, Any]], client=None) -> bool:
    """Stream a batch of dict records into a BigQuery table using insert_rows_json."""
    if not rows:
        return True
    client = client or get_bigquery_client()
    if client is None:
        return False

    try:
        table_ref = f"{client.project}.{settings.BIGQUERY_DATASET}.{table_name}"
        errors = client.insert_rows_json(table_ref, rows)
        if errors:
            logger.error("BigQuery insert_rows_json errors for %s: %s", table_name, errors)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("BigQuery stream failure on %s: %s", table_name, exc)
        return False


def stream_portfolio_snapshot(snapshot, client=None) -> bool:
    """Stream a single PortfolioSnapshot into BigQuery."""
    if not is_available():
        return False
    row = _format_snapshot(snapshot)
    return stream_rows(TABLE_SNAPSHOTS, [row], client=client)


def stream_recommendation_outcomes(outcomes: Sequence[Any], client=None) -> bool:
    """Stream a list of RecommendationOutcome objects into BigQuery."""
    if not is_available() or not outcomes:
        return False
    rows = [_format_outcome(o) for o in outcomes]
    return stream_rows(TABLE_OUTCOMES, rows, client=client)


def stream_audit_log(audit_log, client=None) -> bool:
    """Stream an AuditLog record into BigQuery."""
    if not is_available():
        return False
    row = _format_audit_log(audit_log)
    return stream_rows(TABLE_AUDIT_LOGS, [row], client=client)


# --- Backfill and Status ------------------------------------------------------


def status() -> Dict[str, Any]:
    """Diagnostic info for /health and administrative status."""
    enabled = settings.BIGQUERY_ENABLED
    available = is_available()
    client = get_bigquery_client() if available else None
    return {
        "enabled": enabled,
        "available": available,
        "connected": client is not None,
        "project": client.project if client else settings.BIGQUERY_PROJECT_ID,
        "dataset": settings.BIGQUERY_DATASET,
        "location": settings.BIGQUERY_LOCATION,
    }


def sync_all(db, org_id: Optional[int] = None, client=None) -> Dict[str, int]:
    """Backfill records from the SQL database to BigQuery."""
    from db import AuditEvent as AuditLog, PortfolioSnapshot, RecommendationOutcome

    counts = {"snapshots": 0, "outcomes": 0, "audit_logs": 0}
    if not is_available():
        return counts

    client = client or get_bigquery_client()
    if client is None:
        return counts

    ensure_dataset_and_tables(client)

    # 1. Snapshots
    q_snap = db.query(PortfolioSnapshot)
    if org_id is not None:
        q_snap = q_snap.filter(PortfolioSnapshot.org_id == org_id)
    snapshots = q_snap.all()
    if snapshots:
        rows = [_format_snapshot(s) for s in snapshots]
        if stream_rows(TABLE_SNAPSHOTS, rows, client=client):
            counts["snapshots"] = len(rows)

    # 2. Outcomes
    q_out = db.query(RecommendationOutcome)
    if org_id is not None:
        q_out = q_out.filter(RecommendationOutcome.org_id == org_id)
    outcomes = q_out.all()
    if outcomes:
        rows = [_format_outcome(o) for o in outcomes]
        if stream_rows(TABLE_OUTCOMES, rows, client=client):
            counts["outcomes"] = len(rows)

    # 3. Audit Logs
    q_aud = db.query(AuditLog)
    if org_id is not None:
        q_aud = q_aud.filter(AuditLog.org_id == org_id)
    audits = q_aud.all()
    if audits:
        rows = [_format_audit_log(a) for a in audits]
        if stream_rows(TABLE_AUDIT_LOGS, rows, client=client):
            counts["audit_logs"] = len(rows)

    return counts
