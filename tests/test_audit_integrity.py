from db import AuditEvent, Organization, SessionLocal
from services import audit


def _ensure_org(db, org_id):
    org = db.query(Organization).filter(Organization.id == org_id).first()
    if not org:
        org = Organization(id=org_id, name=f"Audit Org {org_id}", slug=f"audit-org-{org_id}")
        db.add(org)
        db.commit()


def test_audit_hash_chain_genesis_and_continuation():
    """First event links to GENESIS_HASH; second event links to first event's hash."""
    db = SessionLocal()
    test_org_id = 9999
    try:
        _ensure_org(db, test_org_id)
        # Clear any preexisting events for test org
        db.query(AuditEvent).filter(AuditEvent.org_id == test_org_id).delete()
        db.commit()

        event1 = audit.record(
            db,
            org_id=test_org_id,
            action=audit.Action.CLIENT_CREATED,
            user_id=None,
            detail={"name": "Alice"},
        )
        db.commit()

        assert event1.prev_hash == audit.GENESIS_HASH
        assert len(event1.hash) == 64

        event2 = audit.record(
            db,
            org_id=test_org_id,
            action=audit.Action.POLICY_ACTIVATED,
            user_id=None,
            detail={"version": 1},
        )
        db.commit()

        assert event2.prev_hash == event1.hash
        assert len(event2.hash) == 64

        intact, problems = audit.verify_chain(db, test_org_id)
        assert intact is True
        assert problems == []
    finally:
        db.query(AuditEvent).filter(AuditEvent.org_id == test_org_id).delete()
        db.commit()
        db.close()


def test_audit_tamper_detection_modified_payload():
    """Modifying an event's detail breaks hash verification."""
    db = SessionLocal()
    test_org_id = 9998
    try:
        _ensure_org(db, test_org_id)
        db.query(AuditEvent).filter(AuditEvent.org_id == test_org_id).delete()
        db.commit()

        event1 = audit.record(db, org_id=test_org_id, action=audit.Action.ORDER_CREATED, detail={"amt": 100})
        db.commit()
        audit.record(db, org_id=test_org_id, action=audit.Action.ORDER_FILLED, detail={"amt": 100})
        db.commit()

        # Verify intact initially
        intact, problems = audit.verify_chain(db, test_org_id)
        assert intact is True

        # Tamper with event1's detail
        event1.detail = {"amt": 999999}
        db.commit()

        intact_after, problems_after = audit.verify_chain(db, test_org_id)
        assert intact_after is False
        assert len(problems_after) > 0
        assert f"id={event1.id}" in problems_after[0]
    finally:
        db.query(AuditEvent).filter(AuditEvent.org_id == test_org_id).delete()
        db.commit()
        db.close()


def test_audit_tamper_detection_deleted_row():
    """Deleting an intermediate row breaks the predecessor hash chain link."""
    db = SessionLocal()
    test_org_id = 9997
    try:
        _ensure_org(db, test_org_id)
        db.query(AuditEvent).filter(AuditEvent.org_id == test_org_id).delete()
        db.commit()

        audit.record(db, org_id=test_org_id, action=audit.Action.CLIENT_CREATED, detail={"v": 1})
        db.commit()
        event2 = audit.record(db, org_id=test_org_id, action=audit.Action.CLIENT_UPDATED, detail={"v": 2})
        db.commit()
        event3 = audit.record(db, org_id=test_org_id, action=audit.Action.CLIENT_UPDATED, detail={"v": 3})
        db.commit()

        # Delete intermediate event2
        db.delete(event2)
        db.commit()

        intact, problems = audit.verify_chain(db, test_org_id)
        assert intact is False
        assert any(f"id={event3.id}" in p for p in problems)
    finally:
        db.query(AuditEvent).filter(AuditEvent.org_id == test_org_id).delete()
        db.commit()
        db.close()
