"""Authentication, authorization and tenancy enforcement.

The previous version of this system had one shared `X-API-Key` for everything.
Any holder could read, modify and run analysis for every client in the
database, and the human approval that gates a recommendation was signed with a
caller-supplied string — so an approval could be attributed to any named
advisor by anyone able to type their name. For a system whose entire claim is
an auditable chain of custody over investment advice, that was the load-bearing
gap.

What replaces it:

* **Two principal kinds, one representation.** A browser session (JWT) and a
  machine key both resolve to a `Principal` carrying an org, a user and a role.
  Every downstream check reads the `Principal`, so there is no path where one
  authentication method skips a check the other applies.

* **Tenancy is enforced by construction.** `scoped_query` is the only
  sanctioned way to read a tenant table, and it filters on `org_id`
  unconditionally. Authorization that depends on each endpoint remembering to
  add a `WHERE` clause fails the first time someone adds an endpoint.

* **Role separation between proposing and approving.** An advisor may run
  analysis and draft trades; a compliance user may approve them. That is a
  control, not a nicety: it is the difference between a human-in-the-loop and
  a rubber stamp the same person applies to their own work.

* **404, not 403, for another tenant's rows.** Returning "forbidden" for an
  object that exists confirms it exists. Cross-tenant probes get the same
  answer as nonexistent ids.
"""

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Sequence, Tuple

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Query, Session

from config import settings
from db import ApiKey, Organization, User, get_db, utcnow
from logging_setup import get_logger

logger = get_logger(__name__)

# Ordered weakest to strongest. `has_at_least` compares by index, so adding a
# role means placing it correctly here and nowhere else.
ROLE_ORDER = ("viewer", "advisor", "compliance", "admin")

# Capability -> minimum role. Naming capabilities rather than checking roles
# at each call site means the policy is readable in one place, and a change
# ("let advisors approve their own trades below $10k") is a change here rather
# than an audit of every endpoint.
CAPABILITIES: Dict[str, str] = {
    "client:read": "viewer",
    "client:write": "advisor",
    "client:archive": "admin",
    "policy:read": "viewer",
    "policy:draft": "advisor",
    # Activating a policy changes the limits the system will enforce. That is
    # a compliance act, not an advisory one.
    "policy:activate": "compliance",
    "run:trigger": "advisor",
    "run:read": "viewer",
    # The whole point of the human-in-the-loop gate. Deliberately above
    # "advisor" so the person who requested the run cannot clear it.
    "approval:decide": "compliance",
    "order:draft": "advisor",
    "order:submit": "compliance",
    "order:cancel": "advisor",
    "report:read": "viewer",
    "audit:read": "compliance",
    "admin:users": "admin",
    "admin:keys": "admin",
}

API_KEY_PREFIX = "awm"
_BCRYPT_ROUNDS = 12


# --- Passwords ---------------------------------------------------------------


def hash_password(password: str) -> str:
    if len(password) < 12 and not settings.is_development:
        raise ValueError("Password must be at least 12 characters.")
    # bcrypt silently truncates at 72 bytes. Pre-hashing means a long
    # passphrase's full entropy contributes, rather than only its first 72
    # bytes -- and that truncation is a real credential-strength bug, not a
    # theoretical one.
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return bcrypt.hashpw(digest, bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        digest = hashlib.sha256(password.encode("utf-8")).digest()
        return bcrypt.checkpw(digest, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # A malformed stored hash must read as "wrong password", never as an
        # exception that a caller might treat as success.
        return False


# --- API keys ----------------------------------------------------------------


def generate_api_key() -> Tuple[str, str, str]:
    """Returns (full_key, prefix, key_hash). The full key is shown once.

    Keys are hashed with SHA-256 rather than bcrypt: they are 256-bit random
    values, so there is no dictionary to defend against, and they are verified
    on every request where a 200ms KDF would be a self-inflicted denial of
    service.
    """
    secret = secrets.token_urlsafe(32)
    prefix = secrets.token_hex(4)
    full_key = f"{API_KEY_PREFIX}_{prefix}_{secret}"
    return full_key, prefix, hash_api_key(full_key)


def hash_api_key(full_key: str) -> str:
    return hashlib.sha256(full_key.encode("utf-8")).hexdigest()


def _parse_api_key(full_key: str) -> Optional[str]:
    parts = full_key.split("_", 2)
    if len(parts) != 3 or parts[0] != API_KEY_PREFIX:
        return None
    return parts[1]


# --- Tokens ------------------------------------------------------------------


def create_access_token(user: User, *, token_type: str = "access") -> str:
    now = datetime.now(timezone.utc)
    ttl = (
        timedelta(minutes=settings.ACCESS_TOKEN_TTL_MINUTES)
        if token_type == "access"
        else timedelta(days=settings.REFRESH_TOKEN_TTL_DAYS)
    )
    payload = {
        "sub": str(user.id),
        "org": user.org_id,
        "role": user.role,
        # Carried so a password change or forced logout can invalidate tokens
        # that have not yet expired. Without it, "log everyone out" is
        # impossible short of rotating the signing secret for all users.
        "tv": user.token_version,
        "type": token_type,
        "iat": now,
        "exp": now + ttl,
        "jti": secrets.token_urlsafe(12),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
            # Pinning the algorithm list is what prevents an attacker
            # presenting an `alg: none` token, historically the most common
            # JWT vulnerability.
            options={"require": ["exp", "sub", "org"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired; sign in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid session token.")


# --- Principals --------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    """Who is making this request. The only input to every authz decision."""

    user_id: Optional[int]
    org_id: int
    role: str
    kind: str  # "user" | "api_key"
    label: str
    api_key_id: Optional[int] = None

    def has_at_least(self, minimum_role: str) -> bool:
        try:
            return ROLE_ORDER.index(self.role) >= ROLE_ORDER.index(minimum_role)
        except ValueError:
            # An unrecognized role denies rather than defaults. A typo in a
            # role name must not silently grant access.
            return False

    def can(self, capability: str) -> bool:
        required = CAPABILITIES.get(capability)
        if required is None:
            logger.error("Unknown capability %r requested; denying.", capability)
            return False
        return self.has_at_least(required)


def _principal_from_api_key(db: Session, raw_key: str) -> Principal:
    prefix = _parse_api_key(raw_key)
    generic = HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired API key.")
    if not prefix:
        raise generic

    # Look up by the non-secret prefix, then compare the hash in constant
    # time. Querying by hash directly would work too, but the prefix index
    # keeps the lookup cheap and lets a revoked key be identified in logs.
    candidates = db.query(ApiKey).filter(ApiKey.prefix == prefix).all()
    expected = hash_api_key(raw_key)
    for key in candidates:
        if not hmac.compare_digest(key.key_hash, expected):
            continue
        if key.revoked_at is not None:
            raise generic
        if key.expires_at is not None and key.expires_at < utcnow():
            raise generic
        user = db.query(User).filter(User.id == key.user_id, User.is_active.is_(True)).first()
        if user is None:
            raise generic
        # Best-effort: a failed last-used update must not fail the request.
        key.last_used_at = utcnow()
        try:
            db.commit()
        except Exception:
            db.rollback()
        # The key's role may be narrower than its owner's, never wider --
        # otherwise a key becomes a privilege-escalation primitive.
        effective = key.role
        if ROLE_ORDER.index(effective) > ROLE_ORDER.index(user.role):
            effective = user.role
        return Principal(
            user_id=user.id,
            org_id=key.org_id,
            role=effective,
            kind="api_key",
            label=f"api_key:{key.name}",
            api_key_id=key.id,
        )
    raise generic


def get_principal(
    request: Request,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> Principal:
    """Resolve the caller. Raises 401 if it cannot."""
    if authorization and authorization.lower().startswith("bearer "):
        payload = decode_token(authorization.split(" ", 1)[1].strip())
        if payload.get("type") != "access":
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Refresh tokens cannot be used to call the API; exchange it first.",
            )
        user = db.query(User).filter(User.id == int(payload["sub"])).first()
        if user is None or not user.is_active:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Account is inactive.")
        if user.token_version != payload.get("tv"):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, "Session has been revoked; sign in again."
            )
        request.state.principal_label = user.email
        return Principal(
            user_id=user.id,
            org_id=user.org_id,
            role=user.role,
            kind="user",
            label=user.email,
        )

    if x_api_key:
        principal = _principal_from_api_key(db, x_api_key)
        request.state.principal_label = principal.label
        return principal

    raise HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "Authentication required: send `Authorization: Bearer <token>` or `X-API-Key`.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require(capability: str):
    """Dependency factory gating an endpoint on a capability.

    Usage: `dependencies=[Depends(require("approval:decide"))]`, or take the
    returned Principal as a parameter when the handler needs the identity.
    """

    def _dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.can(capability):
            required = CAPABILITIES.get(capability, "unknown")
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"This action requires the '{required}' role or higher; "
                f"you are '{principal.role}'.",
            )
        return principal

    return _dependency


# --- Tenancy -----------------------------------------------------------------


def scoped_query(db: Session, model, principal: Principal) -> Query:
    """The only sanctioned way to read a tenant-owned table.

    Every model passed here must carry `org_id`; the assertion is a loud
    failure at import/dev time rather than a silent cross-tenant read in
    production.
    """
    if not hasattr(model, "org_id"):
        raise TypeError(
            f"{model.__name__} has no org_id and cannot be tenant-scoped. "
            "Query it directly and justify why that is safe."
        )
    return db.query(model).filter(model.org_id == principal.org_id)


def get_or_404(db: Session, model, entity_id: Any, principal: Principal, label: str = "Resource"):
    """Fetch one tenant-owned row or raise 404.

    Deliberately 404 and not 403 for a row belonging to another org: a 403
    confirms the id exists, which turns any id-guessing loop into a working
    enumeration oracle across tenants.
    """
    obj = scoped_query(db, model, principal).filter(model.id == entity_id).first()
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{label} not found.")
    return obj


# --- Login flow --------------------------------------------------------------


class AuthResult:
    def __init__(self, user: Optional[User], error: Optional[str] = None):
        self.user = user
        self.error = error


def authenticate(db: Session, org_slug: str, email: str, password: str) -> AuthResult:
    """Verify credentials with lockout. Caller commits.

    Every failure path returns the same message. Distinguishing "no such user"
    from "wrong password" from "locked" tells an attacker which addresses are
    real and which accounts are worth continuing to hammer.
    """
    generic = "Invalid credentials."
    org = db.query(Organization).filter(Organization.slug == org_slug.lower()).first()
    if org is None or not org.is_active:
        return AuthResult(None, generic)

    user = (
        db.query(User)
        .filter(User.org_id == org.id, User.email == email.lower().strip())
        .first()
    )
    if user is None:
        # Spend roughly the same time as a real verification would, so
        # response latency does not reveal whether the address exists.
        verify_password(password, hash_password("timing-equalizer"))
        return AuthResult(None, generic)

    if user.locked_until and user.locked_until > utcnow():
        return AuthResult(None, generic)

    if not user.is_active or not verify_password(password, user.password_hash):
        user.failed_login_count += 1
        if user.failed_login_count >= settings.LOGIN_MAX_FAILURES:
            user.locked_until = utcnow() + timedelta(minutes=settings.LOGIN_LOCKOUT_MINUTES)
            user.failed_login_count = 0
            logger.warning("Locked account after repeated failures", extra={"user_id": user.id})
        return AuthResult(None, generic)

    user.failed_login_count = 0
    user.locked_until = None
    user.last_login_at = utcnow()
    return AuthResult(user)


def client_ip(request: Request) -> Optional[str]:
    """Best-effort client address for the audit trail.

    `X-Forwarded-For` is trusted only in the sense that it is recorded; it is
    caller-controlled and must never be used for an access decision.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return request.client.host if request.client else None


def bootstrap_admin(db: Session) -> Optional[User]:
    """Create the first org and admin when the database is empty.

    Runs only when there are no users at all, so it can never overwrite an
    existing deployment's credentials. `validate_for_environment` refuses to
    boot outside development if the default password is still in place, which
    is what stops this convenience from becoming a publicly known admin
    account.
    """
    if db.query(User).first() is not None:
        return None

    slug = "".join(c if c.isalnum() else "-" for c in settings.BOOTSTRAP_ORG_NAME.lower()).strip("-")
    org = db.query(Organization).filter(Organization.slug == slug).first()
    if org is None:
        org = Organization(name=settings.BOOTSTRAP_ORG_NAME, slug=slug)
        db.add(org)
        db.flush()

    admin = User(
        org_id=org.id,
        email=settings.BOOTSTRAP_ADMIN_EMAIL.lower().strip(),
        full_name="Bootstrap Administrator",
        password_hash=hash_password(settings.BOOTSTRAP_ADMIN_PASSWORD),
        role="admin",
    )
    db.add(admin)
    db.commit()
    logger.info(
        "Bootstrapped organization %r with admin %s. Change this password.",
        org.name,
        admin.email,
    )
    return admin


def roles_at_least(minimum: str) -> Sequence[str]:
    """Roles satisfying a minimum. Useful for building UI affordances."""
    return ROLE_ORDER[ROLE_ORDER.index(minimum):]
