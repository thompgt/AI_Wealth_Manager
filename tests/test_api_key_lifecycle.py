"""Unit tests for API key lifecycle visibility, revocation reasons, and zero-downtime rotation."""

from datetime import timedelta
import pytest
from fastapi.testclient import TestClient

from db import ApiKey, AuditEvent, Organization, SessionLocal, User, utcnow
from services.audit import Action
from security import create_access_token
from server import app

client = TestClient(app)


@pytest.fixture
def key_env():
    db = SessionLocal()
    org_id = 6666
    try:
        org = db.query(Organization).filter(Organization.id == org_id).first()
        if not org:
            org = Organization(id=org_id, name="Key Lifecycle Org", slug="key-org")
            db.add(org)
            db.commit()

        admin = db.query(User).filter(User.org_id == org_id, User.role == "admin").first()
        if not admin:
            admin = User(
                org_id=org_id,
                email="key_admin@example.com",
                password_hash="dummy_hash",
                role="admin",
            )
            db.add(admin)
            db.commit()

        advisor = db.query(User).filter(User.org_id == org_id, User.role == "advisor").first()
        if not advisor:
            advisor = User(
                org_id=org_id,
                email="key_advisor@example.com",
                password_hash="dummy_hash",
                role="advisor",
            )
            db.add(advisor)
            db.commit()

        admin_token = create_access_token(admin)
        advisor_token = create_access_token(advisor)

        yield {
            "db": db,
            "org_id": org_id,
            "admin": admin,
            "advisor": advisor,
            "admin_token": admin_token,
            "advisor_token": advisor_token,
        }
    finally:
        db.query(ApiKey).filter(ApiKey.org_id == org_id).delete()
        db.query(AuditEvent).filter(AuditEvent.org_id == org_id).delete()
        db.commit()
        db.close()


def test_list_api_keys_lifecycle_visibility(key_env):
    """GET /api/v1/auth/api-keys displays metadata, prefixes, expiration without disclosing secrets."""
    admin_headers = {"Authorization": f"Bearer {key_env['admin_token']}"}

    # Issue two keys
    create_resp = client.post(
        "/api/v1/auth/api-keys?name=Ingestion Service&role=advisor&expires_in_days=30",
        headers=admin_headers,
    )
    assert create_resp.status_code == 201
    key1_data = create_resp.json()

    # List keys as admin
    list_resp = client.get("/api/v1/auth/api-keys", headers=admin_headers)
    assert list_resp.status_code == 200
    keys = list_resp.json()
    assert len(keys) >= 1
    found = [k for k in keys if k["id"] == key1_data["id"]][0]
    assert found["prefix"] == key1_data["prefix"]
    assert found["role"] == "advisor"
    assert found["is_expired"] is False
    assert found["is_revoked"] is False
    assert "key" not in found  # Secret must NOT be returned in list


def test_revoke_api_key_with_reason(key_env):
    """Revoking an API key captures revocation reason and rejects subsequent auth."""
    admin_headers = {"Authorization": f"Bearer {key_env['admin_token']}"}

    create_resp = client.post(
        "/api/v1/auth/api-keys?name=Leaked Service Key&role=viewer",
        headers=admin_headers,
    )
    key_data = create_resp.json()
    key_secret = key_data["key"]
    key_id = key_data["id"]

    # Verify key works initially
    auth_resp = client.get("/api/v1/auth/me", headers={"X-API-Key": key_secret})
    assert auth_resp.status_code == 200

    # Revoke with reason
    reason = "credential_compromised_in_ci_logs"
    del_resp = client.delete(
        f"/api/v1/auth/api-keys/{key_id}?reason={reason}",
        headers=admin_headers,
    )
    assert del_resp.status_code == 204

    # Verify key now fails
    auth_resp2 = client.get("/api/v1/auth/me", headers={"X-API-Key": key_secret})
    assert auth_resp2.status_code == 401

    # Verify revocation reason is visible in key list
    list_resp = client.get("/api/v1/auth/api-keys", headers=admin_headers)
    revoked_meta = [k for k in list_resp.json() if k["id"] == key_id][0]
    assert revoked_meta["is_revoked"] is True
    assert revoked_meta["revocation_reason"] == reason


def test_rotate_api_key_immediate(key_env):
    """Rotating with grace_period_hours=0 immediately swaps keys."""
    admin_headers = {"Authorization": f"Bearer {key_env['admin_token']}"}

    create_resp = client.post(
        "/api/v1/auth/api-keys?name=Trading Bot&role=advisor",
        headers=admin_headers,
    )
    old_key_data = create_resp.json()
    old_secret = old_key_data["key"]
    old_id = old_key_data["id"]

    # Rotate immediately
    rot_resp = client.post(
        f"/api/v1/auth/api-keys/{old_id}/rotate?grace_period_hours=0",
        headers=admin_headers,
    )
    assert rot_resp.status_code == 201
    rot_data = rot_resp.json()
    new_secret = rot_data["key"]
    new_id = rot_data["id"]
    assert rot_data["superseded_key_id"] == old_id

    # Old key is immediately rejected
    resp_old = client.get("/api/v1/auth/me", headers={"X-API-Key": old_secret})
    assert resp_old.status_code == 401

    # New key works immediately
    resp_new = client.get("/api/v1/auth/me", headers={"X-API-Key": new_secret})
    assert resp_new.status_code == 200
    assert resp_new.json()["role"] == "advisor"


def test_rotate_api_key_with_grace_period_zero_downtime(key_env):
    """Rotating with grace_period_hours > 0 allows both keys to authenticate concurrently."""
    admin_headers = {"Authorization": f"Bearer {key_env['admin_token']}"}

    create_resp = client.post(
        "/api/v1/auth/api-keys?name=Batch Pipeline&role=advisor&expires_in_days=30",
        headers=admin_headers,
    )
    old_key_data = create_resp.json()
    old_secret = old_key_data["key"]
    old_id = old_key_data["id"]

    # Rotate with 24h grace window
    rot_resp = client.post(
        f"/api/v1/auth/api-keys/{old_id}/rotate?grace_period_hours=24",
        headers=admin_headers,
    )
    assert rot_resp.status_code == 201
    rot_data = rot_resp.json()
    new_secret = rot_data["key"]

    # BOTH old and new key authenticate during the grace period!
    resp_old = client.get("/api/v1/auth/me", headers={"X-API-Key": old_secret})
    assert resp_old.status_code == 200

    resp_new = client.get("/api/v1/auth/me", headers={"X-API-Key": new_secret})
    assert resp_new.status_code == 200

    # Verify audit trail records rotation event
    db = key_env["db"]
    audit_row = (
        db.query(AuditEvent)
        .filter(
            AuditEvent.org_id == key_env["org_id"],
            AuditEvent.action == Action.API_KEY_ROTATED,
            AuditEvent.entity_id == rot_data["id"],
        )
        .order_by(AuditEvent.id.desc())
        .first()
    )
    assert audit_row is not None
    assert audit_row.detail.get("old_key_id") == old_id
    assert audit_row.detail.get("grace_period_hours") == 24
