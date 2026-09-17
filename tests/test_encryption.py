"""Unit tests for field-level encryption of client PII at rest."""

from datetime import datetime, timezone
import pytest
from sqlalchemy import text
from config import Settings
from db import ClientProfile, Organization, SessionLocal
from services import encryption


def test_field_encryption_roundtrip():
    """encrypt_field and decrypt_field roundtrip accurately."""
    secret_email = "jane.doe@private-family-office.com"
    token = encryption.encrypt_field(secret_email)
    assert token is not None
    assert token != secret_email
    assert token.startswith("gAAAAA")
    assert secret_email not in token

    decrypted = encryption.decrypt_field(token)
    assert decrypted == secret_email


def test_field_encryption_handles_empty_and_none():
    """None and empty strings pass through unchanged."""
    assert encryption.encrypt_field(None) is None
    assert encryption.decrypt_field(None) is None
    assert encryption.encrypt_field("") == ""
    assert encryption.decrypt_field("") == ""


def test_field_decryption_backward_compatibility():
    """Unencrypted legacy plaintext strings return as-is without crashing."""
    legacy_plain = "legacy_unencrypted_email@domain.com"
    result = encryption.decrypt_field(legacy_plain)
    assert result == legacy_plain


def test_client_profile_pii_encrypted_at_rest_in_database():
    """Direct database inspection confirms fields are stored as ciphertext, decrypted by ORM."""
    db = SessionLocal()
    org_id = 8888
    try:
        org = db.query(Organization).filter(Organization.id == org_id).first()
        if not org:
            org = Organization(id=org_id, name="Encryption Test Org", slug="enc-test-org")
            db.add(org)
            db.commit()

        plain_email = "highnetworth@investor.org"
        plain_phone = "+1-555-019-2834"
        plain_notes = "Beneficiary of irrevocable family trust #492"
        dob = datetime(1982, 4, 15, 0, 0, 0)

        client = ClientProfile(
            org_id=org_id,
            name="Confidential Client",
            email=plain_email,
            phone=plain_phone,
            date_of_birth=dob,
            notes=plain_notes,
            risk_tolerance="Conservative",
            net_worth=50_000_000,
        )
        db.add(client)
        db.commit()
        client_id = client.id

        # 1. Raw SQL query directly against the table: proves ciphertext at rest
        raw_row = db.execute(
            text("SELECT email, phone, date_of_birth, notes FROM client_profiles WHERE id = :cid"),
            {"cid": client_id},
        ).fetchone()

        raw_email, raw_phone, raw_dob, raw_notes = raw_row
        # Ciphertext must start with Fernet marker 'gAAAAA'
        assert raw_email.startswith("gAAAAA")
        assert plain_email not in raw_email

        assert raw_phone.startswith("gAAAAA")
        assert plain_phone not in raw_phone

        assert raw_dob.startswith("gAAAAA")

        assert raw_notes.startswith("gAAAAA")
        assert plain_notes not in raw_notes

        # 2. ORM load: proves transparent in-memory decryption
        db.expire_all()
        reloaded = db.query(ClientProfile).filter(ClientProfile.id == client_id).first()
        assert reloaded.email == plain_email
        assert reloaded.phone == plain_phone
        assert reloaded.date_of_birth == dob
        assert reloaded.notes == plain_notes

    finally:
        db.query(ClientProfile).filter(ClientProfile.org_id == org_id).delete()
        db.commit()
        db.close()


def test_production_settings_requires_field_encryption_key():
    """Environment validation catches missing FIELD_ENCRYPTION_KEY in staging/production."""
    stg = Settings(
        ENVIRONMENT="production",
        JWT_SECRET="valid_production_secret_at_least_32_bytes_long",
        BOOTSTRAP_ADMIN_PASSWORD="super_secure_admin_password_123",
        GEMINI_API_KEY="valid_key",
        NEON_DATABASE_URL="postgresql://user:pass@ep-prod.neon.tech/db",
        CHECKPOINT_DB_PATH="",
        CORS_ALLOW_ORIGINS="https://app.wealthmanager.com",
        FIELD_ENCRYPTION_KEY="",  # missing!
    )
    problems = stg.validate_for_environment()
    assert any("FIELD_ENCRYPTION_KEY is unset" in p for p in problems)
