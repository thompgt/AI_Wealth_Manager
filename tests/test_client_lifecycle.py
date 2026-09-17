"""Unit tests for client data export and retention-aware purge lifecycle."""

from datetime import datetime, timedelta
from decimal import Decimal
import pytest
from fastapi.testclient import TestClient

from db import Account, AuditEvent, ClientProfile, Organization, SessionLocal, TaxLot, User, utcnow
from services.audit import Action
from security import create_access_token
from server import app

client = TestClient(app)


@pytest.fixture
def lifecycle_env():
    """Seed test org, admin user, advisor user, and client with accounts."""
    db = SessionLocal()
    org_id = 7777
    try:
        org = db.query(Organization).filter(Organization.id == org_id).first()
        if not org:
            org = Organization(id=org_id, name="Lifecycle Advisory", slug="lifecycle-adv")
            db.add(org)
            db.commit()

        admin = db.query(User).filter(User.org_id == org_id, User.role == "admin").first()
        if not admin:
            admin = User(
                org_id=org_id,
                email="lifecycle_admin@example.com",
                password_hash="dummy_hash",
                role="admin",
            )
            db.add(admin)
            db.commit()

        advisor = db.query(User).filter(User.org_id == org_id, User.role == "advisor").first()
        if not advisor:
            advisor = User(
                org_id=org_id,
                email="lifecycle_advisor@example.com",
                password_hash="dummy_hash",
                role="advisor",
            )
            db.add(advisor)
            db.commit()

        profile = ClientProfile(
            org_id=org_id,
            advisor_id=advisor.id,
            name="Alice Lifecycle",
            email="alice@privacy-rights.org",
            phone="+1-555-4321",
            date_of_birth=datetime(1985, 6, 20),
            notes="Requires annual tax optimization review",
            risk_tolerance="Growth",
            net_worth=Decimal("2500000.00"),
        )
        db.add(profile)
        db.commit()

        acct = Account(
            org_id=org_id,
            client_id=profile.id,
            name="Alice Taxable Brokerage",
            account_type="individual",
            tax_treatment="taxable",
            cash_balance=Decimal("50000.00"),
        )
        db.add(acct)
        db.commit()

        lot = TaxLot(
            org_id=org_id,
            account_id=acct.id,
            symbol="NVDA",
            quantity=Decimal("100"),
            remaining_quantity=Decimal("100"),
            cost_per_share=Decimal("110.50"),
            acquired_at=utcnow() - timedelta(days=200),
            term="short",
        )
        db.add(lot)
        db.commit()

        admin_token = create_access_token(admin)
        advisor_token = create_access_token(advisor)

        yield {
            "db": db,
            "org_id": org_id,
            "client_id": profile.id,
            "admin_token": admin_token,
            "advisor_token": advisor_token,
        }
    finally:
        db.query(ClientProfile).filter(ClientProfile.org_id == org_id).delete()
        db.commit()
        db.close()


def test_export_client_data_success(lifecycle_env):
    """GET /api/v1/clients/{id}/export returns complete dossier and audits the export."""
    headers = {"Authorization": f"Bearer {lifecycle_env['advisor_token']}"}
    resp = client.get(f"/api/v1/clients/{lifecycle_env['client_id']}/export", headers=headers)
    assert resp.status_code == 200

    data = resp.json()
    assert "exported_at" in data
    assert data["client"]["name"] == "Alice Lifecycle"
    assert data["client"]["email"] == "alice@privacy-rights.org"
    assert data["client"]["phone"] == "+1-555-4321"
    assert "1985-06-20" in data["client"]["date_of_birth"]
    assert len(data["accounts"]) == 1
    assert data["accounts"][0]["name"] == "Alice Taxable Brokerage"
    assert len(data["accounts"][0]["tax_lots"]) == 1
    assert data["accounts"][0]["tax_lots"][0]["symbol"] == "NVDA"

    # Verify audit event recorded
    db = lifecycle_env["db"]
    audit_row = (
        db.query(AuditEvent)
        .filter(
            AuditEvent.org_id == lifecycle_env["org_id"],
            AuditEvent.action == Action.CLIENT_EXPORTED,
            AuditEvent.entity_id == lifecycle_env["client_id"],
        )
        .first()
    )
    assert audit_row is not None


def test_purge_active_client_fails(lifecycle_env):
    """Purge fails with 409 if client is active (must be archived first)."""
    headers = {"Authorization": f"Bearer {lifecycle_env['admin_token']}"}
    resp = client.post(f"/api/v1/clients/{lifecycle_env['client_id']}/purge", headers=headers)
    assert resp.status_code == 409
    assert "must be archived" in resp.json()["detail"]


def test_purge_archived_client_retention_rejection_and_force_purge(lifecycle_env):
    """Archived client within retention period requires force=true to purge."""
    db = lifecycle_env["db"]
    client_id = lifecycle_env["client_id"]
    profile = db.query(ClientProfile).filter(ClientProfile.id == client_id).first()
    profile.status = "archived"
    profile.deleted_at = utcnow() - timedelta(days=30)  # archived 30 days ago (well within 1825d)
    db.commit()

    admin_headers = {"Authorization": f"Bearer {lifecycle_env['admin_token']}"}

    # Attempt without force -> 409 Conflict
    resp_reject = client.post(f"/api/v1/clients/{client_id}/purge", headers=admin_headers)
    assert resp_reject.status_code == 409
    assert "mandatory retention period" in resp_reject.json()["detail"]

    # Attempt with force=true -> 200 OK
    resp_purge = client.post(f"/api/v1/clients/{client_id}/purge?force=true", headers=admin_headers)
    assert resp_purge.status_code == 200
    assert resp_purge.json()["status"] == "purged"

    # Verify client is permanently deleted
    db.expire_all()
    deleted = db.query(ClientProfile).filter(ClientProfile.id == client_id).first()
    assert deleted is None

    # Verify audit trail contains CLIENT_PURGED event
    audit_row = (
        db.query(AuditEvent)
        .filter(
            AuditEvent.org_id == lifecycle_env["org_id"],
            AuditEvent.action == Action.CLIENT_PURGED,
            AuditEvent.entity_id == client_id,
        )
        .first()
    )
    assert audit_row is not None
    assert audit_row.detail.get("force") is True


def test_purge_requires_admin_capability(lifecycle_env):
    """An advisor cannot purge clients (admin-only capability)."""
    advisor_headers = {"Authorization": f"Bearer {lifecycle_env['advisor_token']}"}
    resp = client.post(
        f"/api/v1/clients/{lifecycle_env['client_id']}/purge?force=true",
        headers=advisor_headers,
    )
    assert resp.status_code == 403
