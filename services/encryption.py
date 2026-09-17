"""Field-level encryption at rest for sensitive client PII.

Protects sensitive personally identifiable information (PII) such as email,
phone, date of birth, and advisor notes using AES-128-CBC + HMAC-SHA256 (Fernet).
In underlying database tables, sensitive columns store ciphertext tokens,
preventing plaintext leakage in database dumps, unencrypted volume snapshots,
or query logs.
"""

import base64
import hashlib
from datetime import datetime
from typing import Any, Optional
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.types import DateTime, Text, TypeDecorator

from config import settings
from logging_setup import get_logger

logger = get_logger(__name__)

_FERNET_INSTANCE: Optional[Fernet] = None
_CACHED_KEY_MATERIAL: Optional[str] = None


def get_fernet_key() -> bytes:
    """Resolve or derive a valid 32-byte urlsafe base64 Fernet key."""
    configured = getattr(settings, "FIELD_ENCRYPTION_KEY", "").strip()
    if configured:
        # Check if already a valid 32-byte urlsafe base64 string
        try:
            raw = base64.urlsafe_b64decode(configured.encode("utf-8"))
            if len(raw) == 32:
                return configured.encode("utf-8")
        except Exception:
            pass
        # Deterministically derive 32-byte urlsafe base64 key from configured string
        digest = hashlib.sha256(configured.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest)

    # In development/test: derive deterministically from JWT_SECRET
    secret = getattr(settings, "JWT_SECRET", "DEV_ONLY_CHANGE_ME")
    digest = hashlib.sha256(f"pii-field-encryption:{secret}".encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def get_fernet() -> Fernet:
    """Return a cached Fernet instance."""
    global _FERNET_INSTANCE, _CACHED_KEY_MATERIAL
    current_key_str = (
        getattr(settings, "FIELD_ENCRYPTION_KEY", "")
        + ":"
        + getattr(settings, "JWT_SECRET", "")
    )
    if _FERNET_INSTANCE is None or _CACHED_KEY_MATERIAL != current_key_str:
        key = get_fernet_key()
        _FERNET_INSTANCE = Fernet(key)
        _CACHED_KEY_MATERIAL = current_key_str
    return _FERNET_INSTANCE


def encrypt_field(plaintext: Optional[str]) -> Optional[str]:
    """Encrypt a string into a base64 Fernet ciphertext token.

    If plaintext is None, returns None. Empty string returns empty string.
    """
    if plaintext is None:
        return None
    if not isinstance(plaintext, str):
        plaintext = str(plaintext)
    if plaintext == "":
        return ""
    token = get_fernet().encrypt(plaintext.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_field(ciphertext: Optional[str]) -> Optional[str]:
    """Decrypt a Fernet ciphertext token back into plaintext string.

    If ciphertext is None, returns None.
    If ciphertext does not match the Fernet token format (e.g. preexisting unencrypted
    data during migration), it is gracefully returned as-is for backward compatibility.
    """
    if ciphertext is None:
        return None
    if ciphertext == "":
        return ""
    # Fernet tokens start with 'gAAAAA'
    if not ciphertext.startswith("gAAAAA"):
        return ciphertext

    try:
        decrypted_bytes = get_fernet().decrypt(ciphertext.encode("utf-8"))
        return decrypted_bytes.decode("utf-8")
    except InvalidToken:
        logger.warning("Failed to decrypt field ciphertext; key mismatch or corrupted data.")
        return ciphertext
    except Exception as exc:
        logger.warning("Unexpected error during field decryption: %s", exc)
        return ciphertext


class EncryptedString(TypeDecorator):
    """SQLAlchemy TypeDecorator that transparently encrypts string columns at rest."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Optional[str], dialect: Any) -> Optional[str]:
        return encrypt_field(value)

    def process_result_value(self, value: Optional[str], dialect: Any) -> Optional[str]:
        return decrypt_field(value)


class EncryptedText(TypeDecorator):
    """SQLAlchemy TypeDecorator for long encrypted text fields (e.g. notes)."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Optional[str], dialect: Any) -> Optional[str]:
        return encrypt_field(value)

    def process_result_value(self, value: Optional[str], dialect: Any) -> Optional[str]:
        return decrypt_field(value)


class EncryptedDateTime(TypeDecorator):
    """SQLAlchemy TypeDecorator that stores ISO datetimes encrypted as Text at rest."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Optional[Any], dialect: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            iso_val = value
        elif isinstance(value, datetime):
            iso_val = value.isoformat()
        else:
            iso_val = str(value)
        return encrypt_field(iso_val)

    def process_result_value(self, value: Optional[str], dialect: Any) -> Optional[datetime]:
        if value is None:
            return None
        decrypted = decrypt_field(value)
        if not decrypted:
            return None
        try:
            return datetime.fromisoformat(decrypted)
        except Exception:
            return None
