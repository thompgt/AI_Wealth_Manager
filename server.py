"""HTTP API.

Every endpoint below resolves a `Principal`, and every query against a
tenant-owned table goes through `scoped_query`, which filters on `org_id`
unconditionally. That is the single most important property of this file: the
previous version took `client_id` from the path with no ownership check at
all, so any holder of the one shared API key could read, modify and run
analysis for every client in the database.

Long work does not happen here. `POST /clients/{id}/runs` queues a job and
returns 202 with a job id; the previous version ran the entire multi-agent
graph inline, holding a request thread for minutes while the client's own
timeout raced it.
"""

import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from itertools import islice
from typing import Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

import metrics
from config import settings
from db import (
    Account,
    AgentRun,
    ApiKey,
    Approval,
    ClientProfile,
    InvestmentPolicy,
    Job,
    Order,
    Organization,
    Report,
    RiskAssessment,
    TaxLot,
    TradeProposal,
    get_db,
    init_db,
    to_float,
    utcnow,
)
from logging_setup import configure_logging, get_logger, log_context
from security import (
    Principal,
    authenticate,
    bootstrap_admin,
    client_ip,
    create_access_token,
    generate_api_key,
    get_or_404,
    get_principal,
    require,
    scoped_query,
)
from services import broker, jobs, policy as policy_service, run_service, tax_lots
from services.audit import Action, record as record_audit, verify_chain
from services.performance import (
    compute_performance,
    evaluate_outcomes,
    recommendation_scorecard,
)
from services.portfolio import compute_drift, concentration_breaches, load_portfolio
from services.providers import provider_health
from services.risk_profile import apply_hard_overrides, questionnaire_schema, score_answers

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    problems = settings.validate_for_environment()
    if problems:
        for problem in problems:
            logger.error("Unsafe configuration: %s", problem)
        raise RuntimeError(
            f"Refusing to start in ENVIRONMENT={settings.ENVIRONMENT} with "
            f"{len(problems)} unsafe setting(s); see the log above."
        )

    # Creates missing tables so a fresh checkout boots. Alembic owns schema
    # evolution; this never alters an existing table.
    init_db()

    from db import SessionLocal

    db = SessionLocal()
    try:
        bootstrap_admin(db)
        jobs.reap_stale_jobs(db)
    finally:
        db.close()

    jobs.start_worker()
    logger.info(
        "Started in %s. LLM %s, trading %s.",
        settings.ENVIRONMENT,
        "enabled" if settings.llm_configured else "DISABLED (deterministic fallbacks)",
        "enabled" if settings.TRADING_ENABLED else "disabled",
    )
    try:
        yield
    finally:
        jobs.stop_worker()


app = FastAPI(
    title="AI Wealth Manager",
    version="2.0.0",
    lifespan=lifespan,
    # The interactive docs enumerate the entire API surface. Useful locally,
    # gratuitous disclosure in a deployment.
    docs_url="/docs" if settings.is_development else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.is_development else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "Idempotency-Key"],
)


# --- Middleware --------------------------------------------------------------


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Correlate logs and record request metrics.

    The route *template* is used as the metric label, never the concrete path:
    labelling by `/clients/8123` produces one time series per client, which is
    how a metrics backend is taken down by its own instrumentation.
    """
    import uuid

    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:8]
    started = time.perf_counter()

    with log_context(request_id=request_id):
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error handling %s %s", request.method, request.url.path)
            raise
        elapsed = time.perf_counter() - started

    route = request.scope.get("route")
    template = getattr(route, "path", request.url.path)
    metrics.http_requests.labels(request.method, template, str(response.status_code)).inc()
    metrics.http_latency.labels(request.method, template).observe(elapsed)
    response.headers["X-Request-ID"] = request_id
    return response


# A per-principal sliding window, kept in memory. Sufficient for a single
# replica and honest about it: a multi-replica deployment needs a shared store,
# and the alternative of pretending otherwise is a limit that silently
# multiplies by the replica count.
#
# Bounded, because one of the keys is `login:{client_ip}` and an unauthenticated
# caller chooses that key. An unbounded dict makes the login endpoint a slow
# memory-exhaustion vector: a few million requests from spoofed or rotating
# source addresses, none of them ever needing to authenticate, and nothing ever
# evicted. An LRU discards the least recently seen key instead, and since the
# window is a minute, a key old enough to be evicted under pressure has almost
# certainly expired anyway.
#
# Guarded by a lock, because the read-modify-write below races: uvicorn runs
# sync endpoint functions in a threadpool, so two threads could both read a
# bucket one under the limit and both append, admitting more requests than the
# limit allows.
_RATE_STATE_MAX_KEYS = 20_000
_rate_state: "OrderedDict[str, List[float]]" = OrderedDict()
_rate_lock = threading.Lock()


def _rate_limit(key: str, limit: int, window_seconds: int) -> bool:
    now = time.time()
    cutoff = now - window_seconds

    with _rate_lock:
        bucket = _rate_state.get(key)
        if bucket is None:
            bucket = []
            _rate_state[key] = bucket
        else:
            bucket[:] = [t for t in bucket if t > cutoff]
        _rate_state.move_to_end(key)

        if len(bucket) >= limit:
            return False
        bucket.append(now)

        # Sweep a bounded number of expired buckets per call, so a burst of
        # one-off keys is reclaimed without ever walking the whole dict on a
        # request path.
        for stale_key in list(islice(_rate_state.keys(), 32)):
            if stale_key == key:
                continue
            stale = _rate_state[stale_key]
            if not stale or stale[-1] <= cutoff:
                del _rate_state[stale_key]

        while len(_rate_state) > _RATE_STATE_MAX_KEYS:
            _rate_state.popitem(last=False)

        return True


def rate_limited(principal: Principal = Depends(get_principal)) -> Principal:
    if not _rate_limit(f"req:{principal.org_id}:{principal.user_id}",
                       settings.RATE_LIMIT_PER_MINUTE, 60):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Rate limit exceeded ({settings.RATE_LIMIT_PER_MINUTE} requests per minute).",
        )
    return principal


# --- Schemas -----------------------------------------------------------------

RiskTolerance = Literal["Conservative", "Moderate", "Aggressive"]


class LoginRequest(BaseModel):
    org_slug: str
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str
    org_id: int
    email: str


class AccountIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    account_type: Literal[
        "individual", "joint", "traditional_ira", "roth_ira", "401k", "trust", "custodial"
    ] = "individual"
    tax_treatment: Literal["taxable", "tax_deferred", "tax_exempt"] = "taxable"
    custodian: Optional[str] = None
    cash_balance: float = Field(default=0, ge=0)


class HoldingIn(BaseModel):
    symbol: str = Field(min_length=1, max_length=24)
    quantity: float = Field(gt=0)
    cost_per_share: float = Field(ge=0)
    acquired_at: Optional[datetime] = None


class AccountWithHoldings(AccountIn):
    holdings: List[HoldingIn] = Field(default_factory=list)


class ClientCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: Optional[EmailStr] = None
    age: Optional[int] = Field(default=None, ge=18, le=120)
    risk_tolerance: RiskTolerance = "Moderate"
    time_horizon_years: int = Field(default=10, ge=0, le=100)
    goals: List[str] = Field(default_factory=list)
    net_worth: float = Field(default=0, ge=0)
    annual_income: Optional[float] = Field(default=None, ge=0)
    notes: Optional[str] = None
    accounts: List[AccountWithHoldings] = Field(default_factory=list)


class ClientUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    email: Optional[EmailStr] = None
    age: Optional[int] = Field(default=None, ge=18, le=120)
    risk_tolerance: Optional[RiskTolerance] = None
    time_horizon_years: Optional[int] = Field(default=None, ge=0, le=100)
    goals: Optional[List[str]] = None
    net_worth: Optional[float] = Field(default=None, ge=0)
    notes: Optional[str] = None


class AccountOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    account_type: str
    tax_treatment: str
    cash_balance: float
    is_active: bool


class ClientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    email: Optional[str]
    age: Optional[int]
    risk_tolerance: str
    time_horizon_years: int
    goals: List[str]
    net_worth: float
    status: str
    accounts: List[AccountOut] = Field(default_factory=list)


class ApprovalDecision(BaseModel):
    approved: bool
    # No `decided_by` field. The identity comes from the authenticated
    # session; accepting it from the body is what made the previous audit
    # trail forgeable by anyone able to type a colleague's name.
    note: Optional[str] = Field(default=None, max_length=2000)


class RiskAnswers(BaseModel):
    answers: Dict[str, str]


class PolicyDraft(BaseModel):
    max_position_pct: Optional[float] = Field(default=None, gt=0, le=1)
    max_sector_pct: Optional[float] = Field(default=None, gt=0, le=1)
    min_cash_pct: Optional[float] = Field(default=None, ge=0, le=1)
    max_cash_pct: Optional[float] = Field(default=None, ge=0, le=1)
    max_position_beta: Optional[float] = Field(default=None, gt=0)
    min_market_cap: Optional[float] = Field(default=None, ge=0)
    target_allocation: Optional[Dict[str, float]] = None
    drift_bands: Optional[Dict[str, float]] = None
    excluded_tickers: Optional[List[str]] = None
    excluded_sectors: Optional[List[str]] = None
    lot_selection_method: Optional[Literal["HIFO", "FIFO", "LIFO"]] = None
    harvest_losses: Optional[bool] = None
    benchmark_ticker: Optional[str] = None
    rebalance_frequency_days: Optional[int] = Field(default=None, ge=1, le=3650)
    notes: Optional[str] = None


class OrderRequest(BaseModel):
    account_id: int
    symbol: str = Field(min_length=1, max_length=24)
    side: Literal["BUY", "SELL"]
    quantity: float = Field(gt=0)
    order_type: Literal["market", "limit"] = "market"
    limit_price: Optional[float] = Field(default=None, gt=0)


class ExecutePlanRequest(BaseModel):
    run_id: str
    # Explicit opt-in. Executing a plan is not a side effect of approving an
    # analysis; it is its own authorized decision.
    confirm: bool = False


# --- Helpers -----------------------------------------------------------------


def _client_out(client: ClientProfile) -> ClientOut:
    return ClientOut(
        id=client.id,
        name=client.name,
        email=client.email,
        age=client.age,
        risk_tolerance=client.risk_tolerance,
        time_horizon_years=client.time_horizon_years,
        goals=list(client.goals or []),
        net_worth=to_float(client.net_worth),
        status=client.status,
        accounts=[
            AccountOut(
                id=a.id,
                name=a.name,
                account_type=a.account_type,
                tax_treatment=a.tax_treatment,
                cash_balance=to_float(a.cash_balance),
                is_active=a.is_active,
            )
            for a in client.accounts
            if a.is_active
        ],
    )


def _get_client(db: Session, client_id: int, principal: Principal) -> ClientProfile:
    client = (
        scoped_query(db, ClientProfile, principal)
        .filter(ClientProfile.id == client_id, ClientProfile.deleted_at.is_(None))
        .first()
    )
    if client is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Client not found.")
    return client


# --- Health and metrics ------------------------------------------------------


@app.get("/health")
def health(db: Session = Depends(get_db)):
    """Unauthenticated liveness probe.

    Reports the dependencies whose failure is otherwise invisible. A system
    where every market-data provider is down still answers HTTP 200 on every
    endpoint and produces reports off cached and fallback paths -- "up" and
    "working" are different questions, and only this endpoint distinguishes
    them.

    Deliberately terse: it is reachable without credentials, so it names no
    model, no version and no internal topology.
    """
    checks: Dict[str, str] = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        logger.exception("Health check database probe failed")
        checks["database"] = "unreachable"

    providers = provider_health()
    open_circuits = [p["provider"] for p in providers if p["circuit_open"]]
    checks["market_data"] = "degraded" if open_circuits else "ok"
    checks["llm"] = "ok" if settings.llm_configured else "not_configured"

    healthy = checks["database"] == "ok" and not open_circuits
    body = {"status": "ok" if healthy else "degraded", "checks": checks}
    return JSONResponse(body, status_code=200 if checks["database"] == "ok" else 503)


def _metrics_guard():
    """Pick the dependency that gates /metrics, once, at import time.

    Prometheus scrapes with a bare GET; it cannot present a bearer token or an
    X-API-Key without a proxy in front of it, so a local stack pointed at this
    process gets a 401 on every scrape and every panel reads empty.

    METRICS_ALLOW_ANONYMOUS swaps the capability check for a no-op dependency
    rather than removing the guard from the route, so the default path is
    still the authenticated one and there is no branch inside the handler that
    could be reached with the wrong assumption. No other route consults this
    flag.
    """
    if settings.METRICS_ALLOW_ANONYMOUS:
        logger.warning(
            "METRICS_ALLOW_ANONYMOUS is set: GET /metrics is served without "
            "authentication. Intended for a local Prometheus only."
        )

        def _unauthenticated() -> None:
            return None

        return _unauthenticated

    return require("audit:read")


@app.get("/metrics")
def prometheus_metrics(_guard: object = Depends(_metrics_guard())):
    """Prometheus exposition.

    Authenticated by default: the metric labels leak client volumes. See
    `_metrics_guard` for the local-scraping escape hatch.
    """
    return Response(metrics.render(), media_type="text/plain; version=0.0.4")


# --- Auth --------------------------------------------------------------------


@app.post("/api/v1/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    # Rate limited by address as well as by principal, because a login attempt
    # has no principal yet -- without this the lockout is the only brake on
    # credential stuffing across many accounts.
    if not _rate_limit(f"login:{client_ip(request)}", 10, 60):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many login attempts.")

    result = authenticate(db, payload.org_slug, payload.email, payload.password)
    if result.user is None:
        org = db.query(Organization).filter(Organization.slug == payload.org_slug.lower()).first()
        if org is not None:
            record_audit(
                db,
                org_id=org.id,
                action=Action.LOGIN_FAILED,
                actor_label=payload.email,
                detail={"email": payload.email},
                ip_address=client_ip(request),
            )
        db.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, result.error or "Invalid credentials.")

    user = result.user
    record_audit(
        db,
        org_id=user.org_id,
        action=Action.LOGIN_SUCCEEDED,
        user_id=user.id,
        actor_label=user.email,
        ip_address=client_ip(request),
    )
    db.commit()

    return TokenResponse(
        access_token=create_access_token(user),
        refresh_token=create_access_token(user, token_type="refresh"),
        expires_in=settings.ACCESS_TOKEN_TTL_MINUTES * 60,
        role=user.role,
        org_id=user.org_id,
        email=user.email,
    )


@app.get("/api/v1/auth/me")
def whoami(principal: Principal = Depends(get_principal)):
    return {
        "user_id": principal.user_id,
        "org_id": principal.org_id,
        "role": principal.role,
        "kind": principal.kind,
        "label": principal.label,
    }


@app.post("/api/v1/auth/api-keys", status_code=201)
def create_api_key(
    name: str = Query(min_length=1, max_length=120),
    role: Literal["viewer", "advisor", "compliance"] = "viewer",
    expires_in_days: int = Query(default=90, ge=1, le=730),
    principal: Principal = Depends(require("admin:keys")),
    db: Session = Depends(get_db),
):
    """Issue a scoped machine key. The secret is shown exactly once."""
    full_key, prefix, key_hash = generate_api_key()
    api_key = ApiKey(
        org_id=principal.org_id,
        user_id=principal.user_id,
        name=name,
        prefix=prefix,
        key_hash=key_hash,
        role=role,
        expires_at=utcnow() + timedelta(days=expires_in_days),
    )
    db.add(api_key)
    db.flush()
    record_audit(
        db,
        org_id=principal.org_id,
        action=Action.API_KEY_CREATED,
        user_id=principal.user_id,
        entity_type="api_key",
        entity_id=api_key.id,
        detail={"name": name, "role": role, "prefix": prefix},
    )
    db.commit()
    return {
        "id": api_key.id,
        "name": name,
        "prefix": prefix,
        "role": role,
        "expires_at": api_key.expires_at.isoformat(),
        "key": full_key,
        "warning": "This is the only time the key is shown. Store it securely.",
    }


@app.delete("/api/v1/auth/api-keys/{key_id}", status_code=204)
def revoke_api_key(
    key_id: int,
    principal: Principal = Depends(require("admin:keys")),
    db: Session = Depends(get_db),
):
    api_key = get_or_404(db, ApiKey, key_id, principal, "API key")
    api_key.revoked_at = utcnow()
    record_audit(
        db,
        org_id=principal.org_id,
        action=Action.API_KEY_REVOKED,
        user_id=principal.user_id,
        entity_type="api_key",
        entity_id=key_id,
    )
    db.commit()
    return Response(status_code=204)


# --- Clients -----------------------------------------------------------------


@app.post("/api/v1/clients", response_model=ClientOut, status_code=201)
def create_client(
    payload: ClientCreate,
    principal: Principal = Depends(require("client:write")),
    db: Session = Depends(get_db),
):
    client = ClientProfile(
        org_id=principal.org_id,
        advisor_id=principal.user_id,
        name=payload.name,
        email=payload.email,
        age=payload.age,
        risk_tolerance=payload.risk_tolerance,
        time_horizon_years=payload.time_horizon_years,
        goals=payload.goals,
        net_worth=Decimal(str(payload.net_worth)),
        annual_income=Decimal(str(payload.annual_income)) if payload.annual_income else None,
        notes=payload.notes,
        onboarded_at=utcnow(),
    )
    db.add(client)
    db.flush()

    for account_payload in payload.accounts:
        account = Account(
            org_id=principal.org_id,
            client_id=client.id,
            name=account_payload.name,
            account_type=account_payload.account_type,
            tax_treatment=account_payload.tax_treatment,
            custodian=account_payload.custodian,
            cash_balance=Decimal(str(account_payload.cash_balance)),
        )
        db.add(account)
        db.flush()
        if account_payload.holdings:
            tax_lots.seed_lots_from_positions(
                db,
                account,
                [
                    (h.symbol, h.quantity, h.cost_per_share, h.acquired_at)
                    for h in account_payload.holdings
                ],
            )

    record_audit(
        db,
        org_id=principal.org_id,
        action=Action.CLIENT_CREATED,
        user_id=principal.user_id,
        entity_type="client",
        entity_id=client.id,
        detail={"name": client.name, "accounts": len(payload.accounts)},
    )
    db.commit()
    db.refresh(client)
    return _client_out(client)


@app.get("/api/v1/clients", response_model=List[ClientOut])
def list_clients(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search: Optional[str] = None,
    principal: Principal = Depends(require("client:read")),
    db: Session = Depends(get_db),
):
    """Paginated. The previous version returned every client in the database."""
    query = scoped_query(db, ClientProfile, principal).filter(
        ClientProfile.deleted_at.is_(None)
    )
    if search:
        query = query.filter(ClientProfile.name.ilike(f"%{search}%"))
    clients = query.order_by(ClientProfile.name).offset(offset).limit(limit).all()
    return [_client_out(c) for c in clients]


@app.get("/api/v1/clients/{client_id}", response_model=ClientOut)
def get_client(
    client_id: int,
    principal: Principal = Depends(require("client:read")),
    db: Session = Depends(get_db),
):
    return _client_out(_get_client(db, client_id, principal))


@app.put("/api/v1/clients/{client_id}", response_model=ClientOut)
def update_client(
    client_id: int,
    payload: ClientUpdate,
    principal: Principal = Depends(require("client:write")),
    db: Session = Depends(get_db),
):
    client = _get_client(db, client_id, principal)
    changes = payload.model_dump(exclude_unset=True)
    for field, value in changes.items():
        setattr(client, field, Decimal(str(value)) if field == "net_worth" else value)
    record_audit(
        db,
        org_id=principal.org_id,
        action=Action.CLIENT_UPDATED,
        user_id=principal.user_id,
        entity_type="client",
        entity_id=client.id,
        detail={"changed": sorted(changes)},
    )
    db.commit()
    db.refresh(client)
    return _client_out(client)


@app.delete("/api/v1/clients/{client_id}", status_code=204)
def archive_client(
    client_id: int,
    principal: Principal = Depends(require("client:archive")),
    db: Session = Depends(get_db),
):
    """Soft delete. Books-and-records rules require the data to survive."""
    client = _get_client(db, client_id, principal)
    client.deleted_at = utcnow()
    client.status = "archived"
    record_audit(
        db,
        org_id=principal.org_id,
        action=Action.CLIENT_ARCHIVED,
        user_id=principal.user_id,
        entity_type="client",
        entity_id=client.id,
    )
    db.commit()
    return Response(status_code=204)


@app.post("/api/v1/clients/{client_id}/accounts", status_code=201)
def add_account(
    client_id: int,
    payload: AccountWithHoldings,
    principal: Principal = Depends(require("client:write")),
    db: Session = Depends(get_db),
):
    client = _get_client(db, client_id, principal)
    account = Account(
        org_id=principal.org_id,
        client_id=client.id,
        name=payload.name,
        account_type=payload.account_type,
        tax_treatment=payload.tax_treatment,
        custodian=payload.custodian,
        cash_balance=Decimal(str(payload.cash_balance)),
    )
    db.add(account)
    db.flush()
    if payload.holdings:
        tax_lots.seed_lots_from_positions(
            db,
            account,
            [(h.symbol, h.quantity, h.cost_per_share, h.acquired_at) for h in payload.holdings],
        )
    record_audit(
        db,
        org_id=principal.org_id,
        action=Action.ACCOUNT_CREATED,
        user_id=principal.user_id,
        entity_type="account",
        entity_id=account.id,
        detail={"client_id": client.id, "type": account.account_type},
    )
    db.commit()
    return {"id": account.id, "name": account.name, "tax_treatment": account.tax_treatment}


@app.get("/api/v1/clients/{client_id}/portfolio")
def get_portfolio(
    client_id: int,
    principal: Principal = Depends(require("client:read")),
    db: Session = Depends(get_db),
):
    """Live valuation, exposures, drift and policy breaches."""
    client = _get_client(db, client_id, principal)
    resolved = policy_service.resolve(db, client)
    view = load_portfolio(db, client)
    return {
        **view.to_dict(),
        "policy": resolved.to_dict(),
        "drift": [d.to_dict() for d in compute_drift(view, resolved)],
        "breaches": concentration_breaches(view, resolved),
        "holdings": [
            {
                "symbol": h.symbol,
                "account_id": h.account_id,
                "account_name": h.account_name,
                "tax_treatment": h.tax_treatment,
                "quantity": round(h.quantity, 6),
                "price": round(h.price, 4),
                "market_value": round(h.market_value, 2),
                "cost_basis": round(h.cost_basis, 2),
                "unrealized_gain": round(h.unrealized_gain, 2),
                "asset_class": h.asset_class,
                "sector": h.sector,
                "stale_price": h.stale_price,
            }
            for h in sorted(view.holdings, key=lambda x: -x.market_value)
        ],
    }


@app.get("/api/v1/clients/{client_id}/tax")
def get_tax_position(
    client_id: int,
    principal: Principal = Depends(require("client:read")),
    db: Session = Depends(get_db),
):
    client = _get_client(db, client_id, principal)
    view = load_portfolio(db, client)
    prices = {h.symbol: h.price for h in view.holdings}
    return {
        "realized_ytd": tax_lots.realized_gains(db, client),
        "harvest_candidates": tax_lots.harvestable_losses(db, client, prices),
        "unrealized": [
            position.to_dict()
            for account in client.accounts
            for position in tax_lots.unrealized_positions(db, account, prices)
        ],
    }


@app.get("/api/v1/clients/{client_id}/lots")
def get_tax_lots(
    client_id: int,
    account_id: Optional[int] = None,
    symbol: Optional[str] = None,
    principal: Principal = Depends(require("client:read")),
    db: Session = Depends(get_db),
):
    client = _get_client(db, client_id, principal)
    account_ids = [a.id for a in client.accounts]
    if account_id:
        if account_id not in account_ids:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Account not found.")
        account_ids = [account_id]

    query = db.query(TaxLot).filter(TaxLot.account_id.in_(account_ids))
    if symbol:
        query = query.filter(TaxLot.symbol == symbol.upper())
    lots = query.order_by(TaxLot.symbol, TaxLot.acquired_at).limit(500).all()

    return [
        {
            "id": lot.id,
            "account_id": lot.account_id,
            "symbol": lot.symbol,
            "quantity": to_float(lot.quantity),
            "remaining_quantity": to_float(lot.remaining_quantity),
            "cost_per_share": to_float(lot.cost_per_share),
            "acquired_at": lot.acquired_at.isoformat(),
            "term": tax_lots.term_for(lot.acquired_at),
            "disposed_at": lot.disposed_at.isoformat() if lot.disposed_at else None,
            "realized_gain": to_float(lot.realized_gain) if lot.realized_gain else None,
            "wash_sale_disallowed": (
                to_float(lot.wash_sale_disallowed) if lot.wash_sale_disallowed else None
            ),
        }
        for lot in lots
    ]


# --- Risk profile and policy -------------------------------------------------


@app.get("/api/v1/questionnaire")
def get_questionnaire(principal: Principal = Depends(require("client:read"))):
    return questionnaire_schema()


@app.post("/api/v1/clients/{client_id}/risk-assessment")
def submit_risk_assessment(
    client_id: int,
    payload: RiskAnswers,
    principal: Principal = Depends(require("client:write")),
    db: Session = Depends(get_db),
):
    """Score a questionnaire and update the client's tier.

    The tier is derived, never chosen. It gates every position limit in the
    system, and "the client picked Aggressive" is not a basis a suitability
    review accepts.
    """
    from services.risk_profile import expiry_from

    client = _get_client(db, client_id, principal)
    scored = apply_hard_overrides(
        score_answers(payload.answers),
        age=client.age,
        time_horizon_years=client.time_horizon_years,
    )

    assessment = RiskAssessment(
        org_id=principal.org_id,
        client_id=client.id,
        answers=payload.answers,
        raw_score=scored.raw_score,
        capacity_score=scored.capacity_score,
        tolerance_score=scored.tolerance_score,
        assigned_tier=scored.tier,
        expires_at=expiry_from(),
        taken_by_user_id=principal.user_id,
    )
    db.add(assessment)
    client.risk_tolerance = scored.tier

    record_audit(
        db,
        org_id=principal.org_id,
        action=Action.RISK_ASSESSED,
        user_id=principal.user_id,
        entity_type="client",
        entity_id=client.id,
        detail=scored.to_dict(),
    )
    db.commit()
    return {
        **scored.to_dict(),
        "proposed_policy": policy_service.policy_from_assessment(scored.tier),
    }


@app.get("/api/v1/clients/{client_id}/policy")
def get_policy(
    client_id: int,
    principal: Principal = Depends(require("policy:read")),
    db: Session = Depends(get_db),
):
    client = _get_client(db, client_id, principal)
    resolved = policy_service.resolve(db, client)
    versions = (
        db.query(InvestmentPolicy)
        .filter(InvestmentPolicy.client_id == client.id)
        .order_by(InvestmentPolicy.version.desc())
        .all()
    )
    return {
        "active": resolved.to_dict(),
        "versions": [
            {
                "version": p.version,
                "status": p.status,
                "effective_from": p.effective_from.isoformat() if p.effective_from else None,
                "effective_to": p.effective_to.isoformat() if p.effective_to else None,
                "approved_by_user_id": p.approved_by_user_id,
                "created_at": p.created_at.isoformat(),
            }
            for p in versions
        ],
    }


@app.post("/api/v1/clients/{client_id}/policy", status_code=201)
def draft_policy(
    client_id: int,
    payload: PolicyDraft,
    principal: Principal = Depends(require("policy:draft")),
    db: Session = Depends(get_db),
):
    """Create the next draft version, seeded from what is in force."""
    client = _get_client(db, client_id, principal)
    overrides = payload.model_dump(exclude_unset=True)
    policy = policy_service.draft_policy(
        db, client, created_by_user_id=principal.user_id, overrides=overrides
    )
    problems = policy_service.validate_policy(policy)
    record_audit(
        db,
        org_id=principal.org_id,
        action=Action.POLICY_DRAFTED,
        user_id=principal.user_id,
        entity_type="policy",
        entity_id=policy.id,
        detail={"client_id": client.id, "version": policy.version, "changed": sorted(overrides)},
    )
    db.commit()
    return {"version": policy.version, "status": policy.status, "validation_problems": problems}


@app.post("/api/v1/clients/{client_id}/policy/{version}/activate")
def activate_policy(
    client_id: int,
    version: int,
    principal: Principal = Depends(require("policy:activate")),
    db: Session = Depends(get_db),
):
    """Put a draft into force. Compliance role only.

    Activating a policy changes the limits the system will enforce for this
    client. That is a compliance act, not an advisory one, which is why it
    sits above the role that drafts it.
    """
    client = _get_client(db, client_id, principal)
    policy = policy_service.get_policy_version(db, client.id, version)
    if policy is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Policy version not found.")

    try:
        policy_service.activate_policy(db, policy, approved_by_user_id=principal.user_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))

    record_audit(
        db,
        org_id=principal.org_id,
        action=Action.POLICY_ACTIVATED,
        user_id=principal.user_id,
        entity_type="policy",
        entity_id=policy.id,
        detail={"client_id": client.id, "version": version},
    )
    db.commit()
    return {"version": version, "status": "active", "approved_by": principal.label}


# --- Runs --------------------------------------------------------------------


@app.post("/api/v1/clients/{client_id}/runs", status_code=202)
def trigger_run(
    client_id: int,
    principal: Principal = Depends(require("run:trigger")),
    db: Session = Depends(get_db),
):
    """Queue an analysis run. Returns immediately with a job id.

    The previous version executed the entire graph inline: minutes of market
    data, news and LLM calls holding a request thread, with the caller's own
    timeout racing the work and no way to see progress or cancel.
    """
    client = _get_client(db, client_id, principal)

    if not _rate_limit(f"run:{principal.org_id}", settings.RUN_RATE_LIMIT_PER_HOUR, 3600):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Run limit reached ({settings.RUN_RATE_LIMIT_PER_HOUR} per hour for this "
            f"organisation). Each run costs market-data and model calls.",
        )

    job = jobs.enqueue(
        db,
        job_type=run_service.JOB_TYPE,
        org_id=principal.org_id,
        client_id=client.id,
        requested_by_user_id=principal.user_id,
    )
    db.commit()
    return {
        "job_id": job.job_id,
        "status": job.status,
        "client_id": client.id,
        "poll": f"/api/v1/jobs/{job.job_id}",
    }


@app.get("/api/v1/jobs/{job_id}")
def get_job(
    job_id: str,
    principal: Principal = Depends(require("run:read")),
    db: Session = Depends(get_db),
):
    job = scoped_query(db, Job, principal).filter(Job.job_id == job_id).first()
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found.")
    return jobs.job_to_dict(job)


@app.get("/api/v1/jobs")
def list_jobs(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=25, ge=1, le=100),
    principal: Principal = Depends(require("run:read")),
    db: Session = Depends(get_db),
):
    query = scoped_query(db, Job, principal)
    if status_filter:
        query = query.filter(Job.status == status_filter)
    rows = query.order_by(Job.queued_at.desc()).limit(limit).all()
    return [jobs.job_to_dict(job) for job in rows]


@app.delete("/api/v1/jobs/{job_id}", status_code=202)
def cancel_job(
    job_id: str,
    principal: Principal = Depends(require("run:trigger")),
    db: Session = Depends(get_db),
):
    """Request cancellation. Cooperative: the worker stops between steps."""
    job = scoped_query(db, Job, principal).filter(Job.job_id == job_id).first()
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found.")
    if not jobs.request_cancel(db, job):
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"Job already finished with status {job.status!r}."
        )
    record_audit(
        db,
        org_id=principal.org_id,
        action=Action.RUN_CANCELLED,
        user_id=principal.user_id,
        entity_type="job",
        entity_id=job.job_id,
    )
    db.commit()
    return jobs.job_to_dict(job)


@app.get("/api/v1/runs/{run_id}")
def get_run(
    run_id: str,
    principal: Principal = Depends(require("run:read")),
    db: Session = Depends(get_db),
):
    """The full audit trail for one run: every node, model, cost and outcome."""
    records = (
        scoped_query(db, AgentRun, principal)
        .filter(AgentRun.run_id == run_id)
        .order_by(AgentRun.started_at)
        .all()
    )
    if not records:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found.")

    report = scoped_query(db, Report, principal).filter(Report.run_id == run_id).first()
    proposals = (
        scoped_query(db, TradeProposal, principal)
        .filter(TradeProposal.run_id == run_id)
        .order_by(TradeProposal.sequence)
        .all()
    )
    total_cost = sum(to_float(r.cost_usd) for r in records if r.cost_usd)

    return {
        "run_id": run_id,
        "client_id": records[0].client_id,
        "total_cost_usd": round(total_cost, 4),
        "total_tokens": sum((r.prompt_tokens or 0) + (r.completion_tokens or 0) for r in records),
        "nodes": [
            {
                "node_name": r.node_name,
                "status": r.status,
                "degraded": r.degraded,
                "started_at": r.started_at.isoformat(),
                "duration_ms": r.duration_ms,
                "model_used": r.model_used,
                "prompt_version": r.prompt_version,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "cost_usd": to_float(r.cost_usd) if r.cost_usd else None,
                "policy_version": r.policy_version,
                "data_provenance": r.data_provenance,
                "input_snapshot": r.input_snapshot,
                "output_snapshot": r.output_snapshot,
                "error_detail": r.error_detail,
            }
            for r in records
        ],
        "proposals": [
            {
                "symbol": p.symbol,
                "side": p.side,
                "notional": to_float(p.notional),
                "status": p.status,
                "blocked_reason": p.blocked_reason,
                "rationale": p.rationale,
                "addresses_flaw": p.addresses_flaw,
                "estimated_tax_cost": (
                    to_float(p.estimated_tax_cost) if p.estimated_tax_cost else None
                ),
            }
            for p in proposals
        ],
        "report_id": report.id if report else None,
    }


# --- Approvals ---------------------------------------------------------------


@app.get("/api/v1/approvals")
def list_pending_approvals(
    principal: Principal = Depends(require("run:read")),
    db: Session = Depends(get_db),
):
    pending = (
        scoped_query(db, Approval, principal)
        .filter(Approval.decision.is_(None))
        .order_by(Approval.requested_at.desc())
        .limit(100)
        .all()
    )
    return [
        {
            "id": a.id,
            "run_id": a.run_id,
            "client_id": a.client_id,
            "reason": a.reason,
            "requested_at": a.requested_at.isoformat(),
            "context": a.context,
        }
        for a in pending
    ]


@app.post("/api/v1/runs/{run_id}/approve")
def decide_run(
    run_id: str,
    decision: ApprovalDecision,
    request: Request,
    principal: Principal = Depends(require("approval:decide")),
    db: Session = Depends(get_db),
):
    """Approve or reject a paused run, and resume it.

    Requires the compliance role: an advisor who requested a run must not be
    able to clear their own work, or the human-in-the-loop gate is a formality.
    """
    approval = (
        scoped_query(db, Approval, principal)
        .filter(Approval.run_id == run_id)
        .order_by(Approval.requested_at.desc())
        .first()
    )
    if approval is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No approval request for this run.")
    if approval.decision is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"This run was already {approval.decision} at "
            f"{approval.decided_at.isoformat() if approval.decided_at else 'an unknown time'}.",
        )

    try:
        return run_service.decide_approval(
            db,
            approval,
            approved=decision.approved,
            user_id=principal.user_id,
            note=decision.note,
            ip_address=client_ip(request),
        )
    except Exception:
        # The decision and the resume commit together, so a failure here rolls
        # back both and the request can simply be retried. The previous
        # version committed the decision first, which left a failed resume
        # permanently wedged behind its own "already decided" check.
        db.rollback()
        logger.exception("Could not resume run %s", run_id)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Could not resume this run; the decision was not recorded. Retry the request.",
        )


# --- Trading -----------------------------------------------------------------


@app.post("/api/v1/clients/{client_id}/orders", status_code=201)
def place_order(
    client_id: int,
    payload: OrderRequest,
    principal: Principal = Depends(require("order:submit")),
    db: Session = Depends(get_db),
):
    """Create and execute one order against the simulated broker."""
    client = _get_client(db, client_id, principal)
    account = next((a for a in client.accounts if a.id == payload.account_id), None)
    if account is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Account not found for this client.")

    try:
        order = broker.create_order(
            db,
            account,
            symbol=payload.symbol,
            side=payload.side,
            quantity=payload.quantity,
            order_type=payload.order_type,
            limit_price=payload.limit_price,
            created_by_user_id=principal.user_id,
            status="approved",
        )
        broker.submit_order(db, order, approved_by_user_id=principal.user_id)
        db.commit()
    except broker.OrderRejected as exc:
        db.rollback()
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))

    return {
        "order_id": order.id,
        "status": order.status,
        "filled_quantity": to_float(order.filled_quantity),
        "average_fill_price": to_float(order.average_fill_price),
    }


@app.post("/api/v1/clients/{client_id}/execute-plan")
def execute_plan(
    client_id: int,
    payload: ExecutePlanRequest,
    principal: Principal = Depends(require("order:submit")),
    db: Session = Depends(get_db),
):
    """Execute the approved trade plan from a run.

    Requires both the compliance role and an explicit `confirm`. Executing is
    deliberately not a side effect of approving the analysis: approving means
    "this advice is sound", and executing means "move the money", and those
    are different decisions that deserve separate acts.
    """
    client = _get_client(db, client_id, principal)

    if not payload.confirm:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Set confirm=true to execute. This moves real positions.",
        )
    if not settings.TRADING_ENABLED:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Trading is disabled on this deployment (TRADING_ENABLED=false).",
        )

    approval = (
        scoped_query(db, Approval, principal)
        .filter(Approval.run_id == payload.run_id)
        .order_by(Approval.requested_at.desc())
        .first()
    )
    if approval is not None and approval.decision != "approved":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Run {payload.run_id} was not approved (decision: {approval.decision or 'pending'}).",
        )

    proposals = (
        scoped_query(db, TradeProposal, principal)
        .filter(
            TradeProposal.run_id == payload.run_id,
            TradeProposal.client_id == client.id,
            TradeProposal.status == "proposed",
        )
        .order_by(TradeProposal.sequence)
        .all()
    )
    if not proposals:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No executable proposals for this run. They may already have been executed.",
        )

    payloads = [
        {
            "symbol": p.symbol,
            "side": p.side,
            "quantity": to_float(p.quantity),
            "account_id": p.account_id,
            "sequence": p.sequence,
            "proposal_id": p.id,
        }
        for p in proposals
        if p.quantity and p.account_id
    ]

    executed = broker.execute_proposals(
        db, payloads, run_id=payload.run_id, approved_by_user_id=principal.user_id
    )
    executed_ids = {o.proposal_id for o in executed if o.proposal_id}
    for proposal in proposals:
        if proposal.id in executed_ids:
            proposal.status = "executed"
    db.commit()

    return {
        "run_id": payload.run_id,
        "orders_placed": len(executed),
        "proposals": len(payloads),
        "orders": [
            {
                "id": o.id,
                "symbol": o.symbol,
                "side": o.side,
                "status": o.status,
                "filled_quantity": to_float(o.filled_quantity),
                "average_fill_price": to_float(o.average_fill_price),
            }
            for o in executed
        ],
    }


@app.get("/api/v1/clients/{client_id}/orders")
def list_orders(
    client_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    principal: Principal = Depends(require("client:read")),
    db: Session = Depends(get_db),
):
    client = _get_client(db, client_id, principal)
    orders = (
        scoped_query(db, Order, principal)
        .filter(Order.client_id == client.id)
        .order_by(Order.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": o.id,
            "symbol": o.symbol,
            "side": o.side,
            "quantity": to_float(o.quantity),
            "status": o.status,
            "filled_quantity": to_float(o.filled_quantity),
            "average_fill_price": to_float(o.average_fill_price),
            "created_at": o.created_at.isoformat(),
            "filled_at": o.filled_at.isoformat() if o.filled_at else None,
            "failure_reason": o.failure_reason,
            "run_id": o.run_id,
        }
        for o in orders
    ]


# --- Performance and reports -------------------------------------------------


@app.get("/api/v1/clients/{client_id}/performance")
def get_performance(
    client_id: int,
    days: Optional[int] = Query(default=None, ge=1, le=3650),
    include_reconstructed: bool = Query(
        default=False,
        description=(
            "Include backfilled valuations, which apply current share counts to past "
            "prices and are therefore survivorship-biased. The response flags this."
        ),
    ),
    principal: Principal = Depends(require("client:read")),
    db: Session = Depends(get_db),
):
    client = _get_client(db, client_id, principal)
    since = utcnow() - timedelta(days=days) if days else None
    result = compute_performance(
        db,
        client,
        since=since,
        risk_free_rate=settings.RISK_FREE_RATE,
        include_reconstructed=include_reconstructed,
    )
    return {
        **result.to_dict(),
        "recommendation_scorecard": recommendation_scorecard(db, client),
    }


@app.post("/api/v1/maintenance/evaluate-outcomes")
def run_outcome_evaluation(
    principal: Principal = Depends(require("admin:users")),
    db: Session = Depends(get_db),
):
    """Score recommendations whose horizon has elapsed. Intended for a cron."""
    return {"evaluated": evaluate_outcomes(db)}


@app.get("/api/v1/clients/{client_id}/reports")
def list_reports(
    client_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    principal: Principal = Depends(require("report:read")),
    db: Session = Depends(get_db),
):
    client = _get_client(db, client_id, principal)
    reports = (
        scoped_query(db, Report, principal)
        .filter(Report.client_id == client.id)
        .order_by(Report.generated_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "run_id": r.run_id,
            "generated_at": r.generated_at.isoformat(),
            "approval_state": r.approval_state,
            "llm_enabled": r.llm_enabled,
            "degraded": r.degraded,
            "policy_version": r.policy_version,
        }
        for r in reports
    ]


@app.get("/api/v1/reports/{report_id}")
def get_report(
    report_id: int,
    principal: Principal = Depends(require("report:read")),
    db: Session = Depends(get_db),
):
    report = get_or_404(db, Report, report_id, principal, "Report")
    return {
        "id": report.id,
        "client_id": report.client_id,
        "run_id": report.run_id,
        "generated_at": report.generated_at.isoformat(),
        "report_text": report.report_text,
        "structured_payload": report.structured_payload,
        "approval_state": report.approval_state,
        "llm_enabled": report.llm_enabled,
        "degraded": report.degraded,
        "disclosure_version": report.disclosure_version,
    }


# --- Audit -------------------------------------------------------------------


@app.get("/api/v1/audit/verify")
def verify_audit_chain(
    principal: Principal = Depends(require("audit:read")),
    db: Session = Depends(get_db),
):
    """Recompute the hash chain and report any break."""
    intact, problems = verify_chain(db, principal.org_id)
    return {"intact": intact, "problems": problems}


@app.get("/api/v1/audit/events")
def list_audit_events(
    action: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=500),
    principal: Principal = Depends(require("audit:read")),
    db: Session = Depends(get_db),
):
    from db import AuditEvent

    query = scoped_query(db, AuditEvent, principal)
    if action:
        query = query.filter(AuditEvent.action == action)
    events = query.order_by(AuditEvent.occurred_at.desc()).limit(limit).all()
    return [
        {
            "id": e.id,
            "action": e.action,
            "user_id": e.user_id,
            "actor": e.actor_label,
            "entity_type": e.entity_type,
            "entity_id": e.entity_id,
            "occurred_at": e.occurred_at.isoformat(),
            "detail": e.detail,
            "ip_address": e.ip_address,
        }
        for e in events
    ]


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Log the detail, return none of it.

    A stack trace in an API response tells an attacker the framework, the file
    layout and often the query. The request id is echoed so a user can quote
    it and an operator can find the corresponding log line.
    """
    request_id = getattr(request.state, "request_id", None)
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred.",
            "request_id": request_id,
        },
    )
