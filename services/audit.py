"""Append-only, hash-chained audit log.

An audit table you can UPDATE is a record of what someone was last willing to
say happened. Chaining each row's hash to its predecessor's does not prevent
tampering — nothing in an application can — but it makes tampering *detectable*
without an external system: change one row's content, delete one row, or
reorder two, and `verify_chain` reports the first index where the recomputed
digest stops matching.

Scope is deliberately narrow. This records *decisions and state changes* —
who approved what, whose policy changed, which order was placed. Diagnostic
detail belongs in the application log; putting it here would bury the handful
of events that matter under noise and make the chain expensive to verify.

The chain is per-organization. A single global chain would serialize writes
across unrelated firms and let one firm's write volume affect another's.
"""

import hashlib
import json
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from db import AuditEvent, utcnow
from logging_setup import get_logger

logger = get_logger(__name__)

# Serializes hash-chain appends within a process. The chain requires reading
# the current tail and writing the next row atomically; two concurrent writers
# that both read the same tail produce two rows claiming the same predecessor,
# which breaks verification. Cross-process safety comes from the
# SELECT ... ORDER BY id DESC executing inside the caller's transaction.
_chain_lock = threading.Lock()

GENESIS_HASH = "0" * 64


# Actions worth chaining. Keeping this an explicit vocabulary rather than free
# text means a query for "every approval last quarter" is reliable.
class Action:
    LOGIN_SUCCEEDED = "auth.login.succeeded"
    LOGIN_FAILED = "auth.login.failed"
    LOGIN_LOCKED = "auth.login.locked"
    API_KEY_CREATED = "auth.api_key.created"
    API_KEY_REVOKED = "auth.api_key.revoked"

    CLIENT_CREATED = "client.created"
    CLIENT_UPDATED = "client.updated"
    CLIENT_ARCHIVED = "client.archived"
    ACCOUNT_CREATED = "account.created"

    POLICY_DRAFTED = "policy.drafted"
    POLICY_ACTIVATED = "policy.activated"
    RISK_ASSESSED = "risk.assessed"

    RUN_REQUESTED = "run.requested"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"

    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_REJECTED = "approval.rejected"

    ORDER_CREATED = "order.created"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"

    REPORT_GENERATED = "report.generated"
    REPORT_DELIVERED = "report.delivered"


def _canonical(payload: Dict[str, Any]) -> str:
    """Stable serialization for hashing.

    `sort_keys` matters: Python dict ordering is insertion-ordered, so the
    same logical event built by two code paths would otherwise hash
    differently and break the chain for no reason.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_hash(
    *,
    prev_hash: str,
    org_id: int,
    user_id: Optional[int],
    action: str,
    entity_type: Optional[str],
    entity_id: Optional[str],
    detail: Dict[str, Any],
    occurred_at: datetime,
) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "prev": prev_hash,
                "org": org_id,
                "user": user_id,
                "action": action,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "detail": detail,
                "at": occurred_at.isoformat(),
            }
        ).encode("utf-8")
    ).hexdigest()


def record(
    db: Session,
    *,
    org_id: int,
    action: str,
    user_id: Optional[int] = None,
    actor_label: Optional[str] = None,
    entity_type: Optional[str] = None,
    entity_id: Optional[Any] = None,
    detail: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> AuditEvent:
    """Append one event. Does not commit — the caller's transaction owns it.

    That is intentional: an audit row recording an action that was then rolled
    back is worse than no row, so the event and the thing it describes must
    commit together.
    """
    detail = detail or {}
    occurred_at = utcnow()

    with _chain_lock:
        tail = db.execute(
            select(AuditEvent.hash)
            .where(AuditEvent.org_id == org_id)
            .order_by(AuditEvent.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        prev_hash = tail or GENESIS_HASH

        event = AuditEvent(
            org_id=org_id,
            user_id=user_id,
            actor_label=actor_label,
            action=action,
            entity_type=entity_type,
            entity_id=str(entity_id) if entity_id is not None else None,
            detail=detail,
            ip_address=ip_address,
            occurred_at=occurred_at,
            prev_hash=prev_hash,
            hash=compute_hash(
                prev_hash=prev_hash,
                org_id=org_id,
                user_id=user_id,
                action=action,
                entity_type=entity_type,
                entity_id=str(entity_id) if entity_id is not None else None,
                detail=detail,
                occurred_at=occurred_at,
            ),
        )
        db.add(event)
        # Assigns the id and makes the row visible to a subsequent tail read
        # in the same transaction, without committing.
        db.flush()

    logger.debug("audit %s org=%s entity=%s:%s", action, org_id, entity_type, entity_id)
    return event


def verify_chain(db: Session, org_id: int, limit: Optional[int] = None) -> Tuple[bool, List[str]]:
    """Recompute the chain and report the first divergence.

    Returns (intact, problems). Problems name the offending row id so the
    surrounding rows can be inspected; the check stops being meaningful after
    the first break, so later rows are reported as unverifiable rather than
    as further independent failures.
    """
    query = (
        db.query(AuditEvent)
        .filter(AuditEvent.org_id == org_id)
        .order_by(AuditEvent.id.asc())
    )
    if limit:
        query = query.limit(limit)

    problems: List[str] = []
    expected_prev = GENESIS_HASH
    for event in query.all():
        if event.prev_hash != expected_prev:
            problems.append(
                f"event id={event.id} claims prev_hash={event.prev_hash!r} but the "
                f"preceding row hashes to {expected_prev!r} -- a row was altered, "
                f"deleted or inserted before this point"
            )
            return False, problems

        recomputed = compute_hash(
            prev_hash=event.prev_hash,
            org_id=event.org_id,
            user_id=event.user_id,
            action=event.action,
            entity_type=event.entity_type,
            entity_id=event.entity_id,
            detail=event.detail or {},
            occurred_at=event.occurred_at,
        )
        if recomputed != event.hash:
            problems.append(
                f"event id={event.id} ({event.action}) content does not match its "
                f"stored hash -- the row was modified after it was written"
            )
            return False, problems
        expected_prev = event.hash

    return True, problems
