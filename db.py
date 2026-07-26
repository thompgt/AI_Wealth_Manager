"""Database schema and session management.

Design notes that are easy to miss from the model definitions alone:

* **Every client-owned row carries `org_id`.** Not because the ORM needs it --
  it could be reached by joining through `client_profiles` -- but because
  authorization filters on it directly. A missing join in one query is an
  information leak across advisory firms; a mandatory column that every query
  filters on is much harder to get wrong.

* **Positions are derived, tax lots are the truth.** `positions` is a
  materialized rollup for fast reads. `tax_lots` is the authoritative record:
  it is what cost basis, holding period, realized gain and the wash-sale rule
  are computed from. Never write a position without writing the lots that
  justify it (`services/tax_lots.py` owns both).

* **Timestamps are timezone-naive UTC** by convention throughout, set via
  `utcnow()`. Mixing naive and aware datetimes in comparisons raises at
  runtime, so the convention has to be uniform. Anything crossing the API
  boundary is normalized in `server.py`.

* **Financial quantities are `Numeric`, not `Float`.** Money that round-trips
  through binary floating point accumulates error, and this system computes
  realized gains by subtracting two such numbers. `Float` was the single most
  consequential type error in the previous schema.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from config import settings

# --- Engine ------------------------------------------------------------------

db_url = settings.NEON_DATABASE_URL
if db_url.startswith("postgres://"):
    # SQLAlchemy 2 dropped the bare `postgres://` scheme; Neon and Heroku
    # still hand it out.
    db_url = db_url.replace("postgres://", "postgresql://", 1)

if "sqlite" in db_url:
    engine = create_engine(db_url, connect_args={"check_same_thread": False, "timeout": 30})
else:
    engine = create_engine(
        db_url,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_recycle=settings.DB_POOL_RECYCLE_SECONDS,
        pool_pre_ping=True,  # a recycled connection killed by the server is
        # otherwise discovered as a failed query, not a reconnect
    )


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_connection, connection_record):
    """SQLite defaults are wrong for a concurrent job worker.

    The default rollback journal takes a database-wide write lock, so the
    background worker persisting a run blocks every API read for the duration.
    WAL lets readers proceed during a write. `busy_timeout` turns the
    remaining lock contention from an immediate "database is locked" error
    into a bounded wait. Foreign keys are off by default in SQLite, which
    would silently permit orphaned lots.
    """
    if "sqlite" not in db_url:
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utcnow() -> datetime:
    """Naive UTC timestamp for the DateTime columns below.

    These columns are timezone-naive and the existing rows are naive UTC, so
    the convention has to stay naive. `datetime.utcnow()` is deprecated in
    Python 3.12+, so we build an aware UTC value and drop the tzinfo rather
    than keep calling the deprecated function in a hundred places.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Money and share quantities. 18,6 comfortably holds fractional shares of a
# high-priced name and a sovereign wealth fund's balance in the same column.
Money = Numeric(18, 6)
Qty = Numeric(20, 8)


def to_float(value) -> float:
    """Numeric columns come back as Decimal. Analytics code (numpy, pandas,
    PyPortfolioOpt) needs float. Convert at the boundary, once, rather than
    scattering `float(...)` through every caller."""
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


# --- Enumerated values -------------------------------------------------------
# Stored as strings rather than native DB enums: Postgres enums require a
# migration to add a value and SQLite has none, so a portable schema with a
# CheckConstraint is both simpler and easier to evolve.

ROLES = ("admin", "advisor", "compliance", "viewer")

ACCOUNT_TYPES = ("individual", "joint", "traditional_ira", "roth_ira", "401k", "trust", "custodial")
# Drives every tax decision. A wash sale is meaningless in a Roth; harvesting
# a loss in an IRA does nothing. The tax agent branches on this, not on the
# account type name.
TAX_TREATMENTS = ("taxable", "tax_deferred", "tax_exempt")

ASSET_CLASSES = (
    "us_equity",
    "intl_equity",
    "em_equity",
    "fixed_income",
    "real_assets",
    "commodity",
    "cash",
    "alternative",
)

ORDER_SIDES = ("BUY", "SELL")
ORDER_STATUSES = ("draft", "pending_approval", "approved", "rejected", "submitted", "filled",
                  "partially_filled", "cancelled", "failed", "expired")
PROPOSAL_STATUSES = ("proposed", "approved", "rejected", "blocked", "executed")
JOB_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled", "timed_out")
LOT_TERMS = ("short", "long")


# --- Tenancy and identity ----------------------------------------------------


class Organization(Base):
    """An advisory firm. The tenancy boundary: no query may cross it."""

    __tablename__ = "organizations"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    slug = Column(String(100), nullable=False, unique=True, index=True)
    is_active = Column(Boolean, nullable=False, default=True)
    # Firm-level defaults inherited by any client without an explicit IPS.
    default_policy = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)


class User(Base):
    """A human principal. Roles are coarse on purpose -- see security.py.

    `role` is checked on every mutating endpoint; the important separation is
    that an advisor may *propose* trades and a compliance user may *approve*
    them, so the same person cannot do both without an explicit admin role.
    """

    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    email = Column(String(320), nullable=False)
    full_name = Column(String(200), nullable=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(32), nullable=False, default="advisor")
    is_active = Column(Boolean, nullable=False, default=True)
    # Brute-force protection state. Reset on any successful login.
    failed_login_count = Column(Integer, nullable=False, default=0)
    locked_until = Column(DateTime, nullable=True)
    last_login_at = Column(DateTime, nullable=True)
    # Bumped to invalidate every outstanding token for this user (password
    # change, forced logout). Session tokens carry the value they were minted
    # with and are rejected when it no longer matches.
    token_version = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    organization = relationship("Organization")

    __table_args__ = (
        # Email uniqueness is per-org: the same person may hold accounts at
        # two firms, and a global unique index would let one firm's signup
        # probe whether an address exists at another.
        UniqueConstraint("org_id", "email", name="uq_users_org_email"),
        CheckConstraint(f"role IN {ROLES}", name="ck_users_role"),
        Index("ix_users_email_lookup", "email"),
    )


class ApiKey(Base):
    """A machine principal, scoped to one org and acting as one user.

    Only the hash is stored. `prefix` is the non-secret leading segment, kept
    so a key can be identified in a UI or a log line without holding the
    secret that would let you use it.
    """

    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(120), nullable=False)
    prefix = Column(String(16), nullable=False, index=True)
    key_hash = Column(String(255), nullable=False)
    role = Column(String(32), nullable=False, default="viewer")
    expires_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        CheckConstraint(f"role IN {ROLES}", name="ck_api_keys_role"),
    )


# --- Clients, accounts, positions --------------------------------------------


class ClientProfile(Base):
    __tablename__ = "client_profiles"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    # The advisor of record. Used for "my clients" views and for routing
    # approvals; authorization is by org, not by this column.
    advisor_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    name = Column(String(200), nullable=False, index=True)
    email = Column(String(320), nullable=True)
    phone = Column(String(50), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    age = Column(Integer, nullable=True)

    # The tier is derived from a scored questionnaire (see RiskAssessment),
    # never free-typed. It is denormalized here because every agent reads it.
    risk_tolerance = Column(String(32), nullable=False, default="Moderate")
    time_horizon_years = Column(Integer, nullable=False, default=10)
    goals = Column(JSON, nullable=False, default=list)
    net_worth = Column(Money, nullable=False, default=0)
    annual_income = Column(Money, nullable=True)
    liquidity_needs = Column(Money, nullable=True)
    # Free-text constraints the client has stated (e.g. "no tobacco", "keep
    # the inherited MSFT"). Surfaced to the research agent as context and to
    # the report as disclosure.
    notes = Column(Text, nullable=True)

    status = Column(String(32), nullable=False, default="active")  # active | prospect | archived
    kyc_status = Column(String(32), nullable=False, default="pending")  # pending | verified | failed
    kyc_verified_at = Column(DateTime, nullable=True)
    onboarded_at = Column(DateTime, nullable=True)
    # Soft delete: books-and-records rules require retention, so a client is
    # never physically removed by the API.
    deleted_at = Column(DateTime, nullable=True, index=True)

    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    accounts = relationship("Account", back_populates="client", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_clients_org_active", "org_id", "deleted_at"),
    )


class Account(Base):
    """A tax wrapper holding positions and cash.

    Splitting accounts out of the client is what makes the tax module correct
    rather than decorative: the same trade is a taxable event in an individual
    account and a non-event in a Roth, and the previous single-bucket model
    could not express the difference.
    """

    __tablename__ = "accounts"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    name = Column(String(120), nullable=False)
    account_type = Column(String(32), nullable=False, default="individual")
    tax_treatment = Column(String(32), nullable=False, default="taxable")
    custodian = Column(String(120), nullable=True)
    account_number_masked = Column(String(32), nullable=True)
    base_currency = Column(String(3), nullable=False, default="USD")
    cash_balance = Column(Money, nullable=False, default=0)
    # Cash committed to unsettled buys. Available cash is
    # `cash_balance - pending_cash_hold`; sizing must use the latter or the
    # system can spend the same dollar twice across two runs.
    pending_cash_hold = Column(Money, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)
    opened_at = Column(DateTime, nullable=False, default=utcnow)
    closed_at = Column(DateTime, nullable=True)

    client = relationship("ClientProfile", back_populates="accounts")
    positions = relationship("Position", back_populates="account", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint(f"account_type IN {ACCOUNT_TYPES}", name="ck_accounts_type"),
        CheckConstraint(f"tax_treatment IN {TAX_TREATMENTS}", name="ck_accounts_tax_treatment"),
    )


class Position(Base):
    """Materialized rollup of open tax lots. Rebuilt, never hand-edited."""

    __tablename__ = "positions"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    symbol = Column(String(24), nullable=False, index=True)
    quantity = Column(Qty, nullable=False, default=0)
    # Weighted average of the open lots. Display-only: gain calculations use
    # the individual lots, because average cost gives the wrong answer for
    # anything but a full liquidation.
    average_cost = Column(Money, nullable=False, default=0)
    updated_at = Column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    account = relationship("Account", back_populates="positions")

    __table_args__ = (
        UniqueConstraint("account_id", "symbol", name="uq_positions_account_symbol"),
    )


class TaxLot(Base):
    """One purchase, tracked to disposal. The authoritative holdings record.

    A partially sold lot keeps its original `quantity` and decrements
    `remaining_quantity`, so the acquisition history survives disposal --
    which is exactly what a wash-sale lookback and a cost-basis audit need.
    """

    __tablename__ = "tax_lots"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    symbol = Column(String(24), nullable=False, index=True)
    quantity = Column(Qty, nullable=False)
    remaining_quantity = Column(Qty, nullable=False)
    cost_per_share = Column(Money, nullable=False)
    acquired_at = Column(DateTime, nullable=False, default=utcnow, index=True)

    # Disposal detail, written when the lot is (partly) sold.
    disposed_at = Column(DateTime, nullable=True)
    proceeds = Column(Money, nullable=True)
    realized_gain = Column(Money, nullable=True)
    term = Column(String(8), nullable=True)  # short | long, at disposal

    # Wash-sale bookkeeping. A disallowed loss is not lost -- it is added to
    # the basis of the replacement lot, which is what `replacement_lot_id`
    # points at. Recording it here is the difference between a system that
    # warns about wash sales and one that accounts for them.
    wash_sale_disallowed = Column(Money, nullable=True)
    replacement_lot_id = Column(Integer, ForeignKey("tax_lots.id", ondelete="SET NULL"), nullable=True)

    execution_id = Column(Integer, ForeignKey("executions.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        Index("ix_tax_lots_open", "account_id", "symbol", "disposed_at"),
        CheckConstraint("term IS NULL OR term IN ('short','long')", name="ck_tax_lots_term"),
    )


class CashTransaction(Base):
    """Contributions, withdrawals, dividends, interest and fees.

    Kept separate from trade executions because performance measurement must
    distinguish a return from a deposit -- conflating them is how a system
    reports a 40% gain on the day a client wires in new money.
    """

    __tablename__ = "cash_transactions"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    # DEPOSIT | WITHDRAWAL | DIVIDEND | INTEREST | FEE | TAX
    transaction_type = Column(String(24), nullable=False)
    amount = Column(Money, nullable=False)  # signed: negative reduces cash
    symbol = Column(String(24), nullable=True)  # set for dividends
    description = Column(String(255), nullable=True)
    occurred_at = Column(DateTime, nullable=False, default=utcnow, index=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)


# --- Securities master -------------------------------------------------------


class Security(Base):
    """The investable universe.

    Moving this out of a Python list is what lifts the ceiling on research: a
    hardcoded 34-name constant could never include an ETF, a bond fund or an
    international name, so the recommendation space was capped by the source
    file. Rows here are refreshed from the market-data provider chain and can
    be restricted per firm without a deploy.
    """

    __tablename__ = "securities"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(24), nullable=False, unique=True, index=True)
    name = Column(String(200), nullable=True)
    asset_class = Column(String(32), nullable=False, default="us_equity", index=True)
    region = Column(String(32), nullable=False, default="US")
    sector = Column(String(64), nullable=True, index=True)
    industry = Column(String(120), nullable=True)
    currency = Column(String(3), nullable=False, default="USD")
    exchange = Column(String(24), nullable=True)
    security_type = Column(String(24), nullable=False, default="equity")  # equity | etf | fund | bond

    # Refreshed fundamentals. Nullable throughout: a screen must be able to
    # tell "this name has no P/E because it has no earnings" from "we failed
    # to fetch it", and a zero default destroys that distinction.
    market_cap = Column(Money, nullable=True)
    avg_dollar_volume = Column(Money, nullable=True)
    beta = Column(Float, nullable=True)
    pe_ratio = Column(Float, nullable=True)
    forward_pe = Column(Float, nullable=True)
    pb_ratio = Column(Float, nullable=True)
    ps_ratio = Column(Float, nullable=True)
    peg_ratio = Column(Float, nullable=True)
    dividend_yield = Column(Float, nullable=True)
    profit_margin = Column(Float, nullable=True)
    return_on_equity = Column(Float, nullable=True)
    debt_to_equity = Column(Float, nullable=True)
    revenue_growth = Column(Float, nullable=True)
    earnings_growth = Column(Float, nullable=True)
    free_cash_flow = Column(Money, nullable=True)
    expense_ratio = Column(Float, nullable=True)  # funds/ETFs only
    duration_years = Column(Float, nullable=True)  # fixed income only
    credit_quality = Column(String(16), nullable=True)

    # Computed by the screener from price history, cached here so a screen
    # over 500 names does not recompute 500 return series per run.
    momentum_6m = Column(Float, nullable=True)
    momentum_12m = Column(Float, nullable=True)
    volatility_1y = Column(Float, nullable=True)
    max_drawdown_1y = Column(Float, nullable=True)

    is_active = Column(Boolean, nullable=False, default=True, index=True)
    # Firm-wide restricted list (insider windows, conflicts, banned issuers).
    # Enforced as a hard block in suitability, not a preference.
    is_restricted = Column(Boolean, nullable=False, default=False)
    restriction_reason = Column(String(255), nullable=True)

    fundamentals_updated_at = Column(DateTime, nullable=True)
    metrics_updated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        CheckConstraint(f"asset_class IN {ASSET_CLASSES}", name="ck_securities_asset_class"),
        Index("ix_securities_screen", "is_active", "asset_class", "sector"),
    )


class MarketDataCache(Base):
    __tablename__ = "market_data_cache"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(24), nullable=False, index=True)
    as_of_date = Column(DateTime, nullable=False, index=True)
    close_price = Column(Money, nullable=False)
    # Which provider served this bar. Two providers disagreeing on a close is
    # a real event; without this column it is undiagnosable.
    provider = Column(String(32), nullable=True)
    fetched_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        # The previous schema lacked this while the write path did a
        # read-then-insert, so two concurrent runs could and did duplicate
        # bars, which silently double-weighted a day in the return series.
        UniqueConstraint("ticker", "as_of_date", name="uq_market_data_ticker_date"),
    )


# --- Policy and risk ---------------------------------------------------------


class InvestmentPolicy(Base):
    """A versioned Investment Policy Statement -- the client's rulebook.

    Every constraint the system enforces resolves from the policy row that was
    in force at run time, and every run stamps the version it used. That is
    what makes "why was this allowed in March but blocked in April?" an
    answerable question rather than an archaeology exercise across git.

    Rows are immutable once active: an edit supersedes the prior version
    rather than mutating it.
    """

    __tablename__ = "investment_policies"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    version = Column(Integer, nullable=False, default=1)
    status = Column(String(16), nullable=False, default="draft")  # draft | active | superseded

    effective_from = Column(DateTime, nullable=True)
    effective_to = Column(DateTime, nullable=True)
    approved_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    approved_at = Column(DateTime, nullable=True)

    # Constraints. Stored as columns rather than one JSON blob because they
    # are queried, validated and diffed between versions.
    max_position_pct = Column(Float, nullable=False, default=0.25)
    max_sector_pct = Column(Float, nullable=False, default=0.35)
    max_asset_class_pct = Column(Float, nullable=True)
    min_cash_pct = Column(Float, nullable=False, default=0.02)
    max_cash_pct = Column(Float, nullable=False, default=0.30)
    max_position_beta = Column(Float, nullable=True)
    max_portfolio_beta = Column(Float, nullable=True)
    min_market_cap = Column(Money, nullable=False, default=2_000_000_000)
    min_avg_dollar_volume = Column(Money, nullable=False, default=5_000_000)
    max_portfolio_volatility = Column(Float, nullable=True)
    min_position_notional = Column(Money, nullable=False, default=100)

    # Strategic asset allocation and the bands that trigger a rebalance.
    # {"us_equity": 0.55, "fixed_income": 0.30, ...}
    target_allocation = Column(JSON, nullable=False, default=dict)
    # {"us_equity": 0.05, ...} -- absolute drift from target, in weight terms.
    drift_bands = Column(JSON, nullable=False, default=dict)
    allowed_asset_classes = Column(JSON, nullable=False, default=list)
    excluded_tickers = Column(JSON, nullable=False, default=list)
    excluded_sectors = Column(JSON, nullable=False, default=list)

    # Tax preferences.
    lot_selection_method = Column(String(16), nullable=False, default="HIFO")  # HIFO | FIFO | LIFO
    harvest_losses = Column(Boolean, nullable=False, default=True)
    max_short_term_gain_budget = Column(Money, nullable=True)

    benchmark_ticker = Column(String(24), nullable=False, default="SPY")
    rebalance_frequency_days = Column(Integer, nullable=False, default=90)
    notes = Column(Text, nullable=True)

    created_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint("client_id", "version", name="uq_policy_client_version"),
        Index("ix_policy_active", "client_id", "status"),
        CheckConstraint("status IN ('draft','active','superseded')", name="ck_policy_status"),
    )


class RiskAssessment(Base):
    """A scored questionnaire, not a self-reported label.

    The tier that gates every position limit used to be a free-choice enum on
    the client form. Here it is the output of dated, versioned answers that
    can be re-attested -- which is both better advice and the artifact a
    suitability review actually asks for.
    """

    __tablename__ = "risk_assessments"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    questionnaire_version = Column(String(16), nullable=False, default="v1")
    answers = Column(JSON, nullable=False, default=dict)  # question_id -> chosen option
    raw_score = Column(Float, nullable=False, default=0)
    # Capacity (objective: horizon, income, liquidity) and tolerance
    # (behavioural) are scored separately, then combined by the lower of the
    # two. A client who *wants* aggressive risk but needs the money in three
    # years does not get an aggressive mandate.
    capacity_score = Column(Float, nullable=False, default=0)
    tolerance_score = Column(Float, nullable=False, default=0)
    assigned_tier = Column(String(32), nullable=False, default="Moderate")
    taken_at = Column(DateTime, nullable=False, default=utcnow)
    # Suitability information goes stale; regulators expect periodic refresh.
    expires_at = Column(DateTime, nullable=True)
    taken_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)


# --- Orders and execution ----------------------------------------------------


class TradeProposal(Base):
    """What the graph decided to do, before anyone approved it.

    Separating proposals from orders is what lets the audit trail show the
    recommendation that was *blocked* alongside the ones that shipped. An
    order table alone only records what happened, never what the guardrails
    stopped -- which is the more interesting half of this system.
    """

    __tablename__ = "trade_proposals"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=True, index=True)
    run_id = Column(String(64), nullable=False, index=True)

    symbol = Column(String(24), nullable=False)
    side = Column(String(8), nullable=False)
    quantity = Column(Qty, nullable=True)
    notional = Column(Money, nullable=False, default=0)
    target_weight = Column(Float, nullable=True)
    current_weight = Column(Float, nullable=True)

    rationale = Column(Text, nullable=True)
    addresses_flaw = Column(String(255), nullable=True)
    confidence = Column(Float, nullable=True)
    estimated_tax_cost = Column(Money, nullable=True)

    status = Column(String(16), nullable=False, default="proposed")
    blocked_reason = Column(String(500), nullable=True)
    sequence = Column(Integer, nullable=False, default=0)  # sells execute before buys

    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        CheckConstraint(f"side IN {ORDER_SIDES}", name="ck_proposal_side"),
        CheckConstraint(f"status IN {PROPOSAL_STATUSES}", name="ck_proposal_status"),
    )


class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    proposal_id = Column(Integer, ForeignKey("trade_proposals.id", ondelete="SET NULL"), nullable=True)
    run_id = Column(String(64), nullable=True, index=True)

    symbol = Column(String(24), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    order_type = Column(String(16), nullable=False, default="market")  # market | limit
    quantity = Column(Qty, nullable=False)
    limit_price = Column(Money, nullable=True)
    time_in_force = Column(String(8), nullable=False, default="day")

    status = Column(String(24), nullable=False, default="draft", index=True)
    filled_quantity = Column(Qty, nullable=False, default=0)
    average_fill_price = Column(Money, nullable=True)
    failure_reason = Column(String(500), nullable=True)

    # Who did what, and when. `approved_by_user_id` is taken from the
    # authenticated session, never from the request body.
    created_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    approved_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    # Caller-supplied key that makes order submission safe to retry: a
    # duplicated request returns the original order instead of doubling the
    # position.
    idempotency_key = Column(String(80), nullable=True)

    created_at = Column(DateTime, nullable=False, default=utcnow)
    submitted_at = Column(DateTime, nullable=True)
    filled_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)

    executions = relationship("Execution", back_populates="order", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("org_id", "idempotency_key", name="uq_orders_idempotency"),
        CheckConstraint(f"side IN {ORDER_SIDES}", name="ck_orders_side"),
        CheckConstraint(f"status IN {ORDER_STATUSES}", name="ck_orders_status"),
        Index("ix_orders_open", "account_id", "status"),
    )


class Execution(Base):
    """A fill. Immutable once written -- corrections are new rows."""

    __tablename__ = "executions"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    order_id = Column(Integer, ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    symbol = Column(String(24), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    quantity = Column(Qty, nullable=False)
    price = Column(Money, nullable=False)
    gross_amount = Column(Money, nullable=False)
    commission = Column(Money, nullable=False, default=0)
    # Modelled cost of crossing the spread. Recorded rather than folded into
    # the price so execution quality is measurable after the fact.
    slippage_cost = Column(Money, nullable=False, default=0)
    net_amount = Column(Money, nullable=False)

    realized_gain = Column(Money, nullable=True)  # sells only
    realized_term = Column(String(8), nullable=True)
    wash_sale_disallowed = Column(Money, nullable=True)

    executed_at = Column(DateTime, nullable=False, default=utcnow, index=True)
    settles_on = Column(DateTime, nullable=True)
    broker_execution_id = Column(String(80), nullable=True)
    venue = Column(String(32), nullable=False, default="SIMULATED")

    order = relationship("Order", back_populates="executions")

    __table_args__ = (
        CheckConstraint(f"side IN {ORDER_SIDES}", name="ck_executions_side"),
    )


# --- Performance -------------------------------------------------------------


class PortfolioSnapshot(Base):
    """Daily valuation, with flows recorded separately.

    Time-weighted return needs to neutralize the effect of deposits and
    withdrawals; that is impossible from a market-value series alone, which is
    why `net_flow` is a column and not a derived quantity.
    """

    __tablename__ = "portfolio_snapshots"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    # NULL means the whole household; a value scopes the snapshot to one account.
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=True, index=True)
    as_of = Column(DateTime, nullable=False, index=True)
    market_value = Column(Money, nullable=False, default=0)
    cash_value = Column(Money, nullable=False, default=0)
    net_flow = Column(Money, nullable=False, default=0)
    # Positions as of this date: {"AAPL": {"qty":.., "price":.., "value":..}}.
    # Needed to attribute return to holdings after the fact.
    holdings_snapshot = Column(JSON, nullable=False, default=dict)
    benchmark_ticker = Column(String(24), nullable=True)
    benchmark_close = Column(Money, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint("client_id", "account_id", "as_of", name="uq_snapshot_scope_date"),
    )


class RecommendationOutcome(Base):
    """Did the advice work? Scored after the fact, at fixed horizons.

    Without this table the system can never be wrong out loud -- it produces
    confident recommendations forever with no feedback signal. Tracking the
    forward return against the benchmark is the minimum honest accounting.
    """

    __tablename__ = "recommendation_outcomes"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    run_id = Column(String(64), nullable=False, index=True)
    symbol = Column(String(24), nullable=False, index=True)
    side = Column(String(8), nullable=False, default="BUY")
    recommended_at = Column(DateTime, nullable=False, default=utcnow, index=True)
    entry_price = Column(Money, nullable=True)
    confidence = Column(Float, nullable=True)
    was_executed = Column(Boolean, nullable=False, default=False)
    horizon_days = Column(Integer, nullable=False, default=90)
    evaluated_at = Column(DateTime, nullable=True)
    exit_price = Column(Money, nullable=True)
    return_pct = Column(Float, nullable=True)
    benchmark_return_pct = Column(Float, nullable=True)
    excess_return_pct = Column(Float, nullable=True)
    status = Column(String(16), nullable=False, default="open")  # open | evaluated | void

    __table_args__ = (
        UniqueConstraint("run_id", "symbol", "horizon_days", name="uq_outcome_run_symbol_horizon"),
    )


# --- Runs, jobs, audit -------------------------------------------------------


class Job(Base):
    """Durable queue row for work too slow to do inside a request.

    A graph run takes minutes of network and LLM latency. Running it inline
    held a server thread for the whole duration with no progress, no
    cancellation and no way to survive a deploy. The row is the unit of
    recovery: a worker that dies mid-run stops heartbeating and the job is
    reclaimed.
    """

    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(64), nullable=False, unique=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id", ondelete="CASCADE"),
                       nullable=True, index=True)
    job_type = Column(String(48), nullable=False, default="analysis_run")
    run_id = Column(String(64), nullable=True, index=True)
    status = Column(String(16), nullable=False, default="queued", index=True)
    priority = Column(Integer, nullable=False, default=100)
    payload = Column(JSON, nullable=False, default=dict)
    result = Column(JSON, nullable=True)

    progress_pct = Column(Float, nullable=False, default=0.0)
    current_step = Column(String(120), nullable=True)
    steps_total = Column(Integer, nullable=True)
    steps_done = Column(Integer, nullable=False, default=0)

    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=1)
    worker_id = Column(String(80), nullable=True)
    heartbeat_at = Column(DateTime, nullable=True, index=True)
    cancel_requested = Column(Boolean, nullable=False, default=False)

    error = Column(Text, nullable=True)
    requested_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    queued_at = Column(DateTime, nullable=False, default=utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint(f"status IN {JOB_STATUSES}", name="ck_jobs_status"),
        Index("ix_jobs_claim", "status", "priority", "queued_at"),
    )


class AgentRun(Base):
    """One node execution. The reproducibility record.

    `input_snapshot`, `model_used`, `prompt_version` and the token counts were
    declared but never written in the previous version, which meant the audit
    trail could say a recommendation happened but not what produced it. For an
    LLM that sizes positions, those columns are the whole point of the table.
    """

    __tablename__ = "agent_runs"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id", ondelete="CASCADE"),
                       nullable=True, index=True)
    run_id = Column(String(64), nullable=False, index=True)
    node_name = Column(String(64), nullable=False)
    attempt = Column(Integer, nullable=False, default=1)

    started_at = Column(DateTime, nullable=False, default=utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    input_snapshot = Column(JSON, nullable=False, default=dict)
    output_snapshot = Column(JSON, nullable=False, default=dict)

    model_used = Column(String(80), nullable=True)
    prompt_version = Column(String(32), nullable=True)
    temperature = Column(Float, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Numeric(12, 6), nullable=True)
    # Which IPS version's limits were applied. The answer to "why was this
    # allowed?" starts here.
    policy_version = Column(Integer, nullable=True)
    # Which providers served the data, and how stale it was.
    data_provenance = Column(JSON, nullable=True)

    status = Column(String(16), nullable=False, default="success")
    degraded = Column(Boolean, nullable=False, default=False)
    error_detail = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("run_id", "node_name", "started_at", name="uq_agent_run_node_start"),
        Index("ix_agent_runs_client_time", "client_id", "started_at"),
    )


class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    run_id = Column(String(64), nullable=False, index=True)
    generated_at = Column(DateTime, nullable=False, default=utcnow)
    report_text = Column(Text, nullable=False)
    structured_payload = Column(JSON, nullable=False, default=dict)
    version = Column(Integer, nullable=False, default=1)

    # A report produced from a rejected run is retained but marked, so it can
    # never be mistaken for advice that cleared review.
    approval_state = Column(String(16), nullable=False, default="pending")  # pending|approved|rejected
    policy_version = Column(Integer, nullable=True)
    llm_enabled = Column(Boolean, nullable=False, default=False)
    degraded = Column(Boolean, nullable=False, default=False)
    data_as_of = Column(DateTime, nullable=True)
    disclosure_version = Column(String(16), nullable=True)
    delivered_at = Column(DateTime, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("run_id", "version", name="uq_reports_run_version"),
    )


class Approval(Base):
    __tablename__ = "approvals"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id", ondelete="CASCADE"),
                       nullable=True, index=True)
    run_id = Column(String(64), nullable=False, index=True)
    reason = Column(String(120), nullable=True)
    context = Column(JSON, nullable=True)

    requested_at = Column(DateTime, nullable=False, default=utcnow)
    decided_at = Column(DateTime, nullable=True)
    # FK to the authenticated user, not caller-supplied text. The previous
    # free-text column let any caller attribute an approval to any named
    # advisor, which is precisely the field an examiner would test.
    decided_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    decision = Column(String(16), nullable=True)  # approved | rejected
    notes = Column(Text, nullable=True)
    ip_address = Column(String(64), nullable=True)

    __table_args__ = (
        Index("ix_approvals_pending", "run_id", "decision"),
    )


class AuditEvent(Base):
    """Append-only, hash-chained record of every consequential action.

    Each row hashes its own content together with the previous row's hash, so
    a silently edited or deleted row breaks the chain and is detectable
    (`services/audit.py::verify_chain`). Ordinary row-level history cannot
    make that claim, and "we have logs" is not the same as "we can prove the
    logs were not altered".
    """

    __tablename__ = "audit_events"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    actor_label = Column(String(120), nullable=True)  # preserved if the user is later removed
    action = Column(String(80), nullable=False, index=True)
    entity_type = Column(String(48), nullable=True)
    entity_id = Column(String(64), nullable=True)
    detail = Column(JSON, nullable=False, default=dict)
    ip_address = Column(String(64), nullable=True)
    occurred_at = Column(DateTime, nullable=False, default=utcnow, index=True)

    prev_hash = Column(String(64), nullable=True)
    hash = Column(String(64), nullable=False)

    __table_args__ = (
        Index("ix_audit_org_time", "org_id", "occurred_at"),
    )


# --- Session helpers ---------------------------------------------------------


def get_db():
    """FastAPI dependency: one session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class session_scope:
    """Context manager for code outside a request (workers, CLI, seeds).

    Commits on clean exit, rolls back on exception. Background work that
    half-commits a rebalance is worse than work that fails, so the rollback is
    the important half.
    """

    def __init__(self):
        self.db: Optional[Session] = None

    def __enter__(self) -> Session:
        self.db = SessionLocal()
        return self.db

    def __exit__(self, exc_type, exc, tb):
        assert self.db is not None
        try:
            if exc_type is None:
                self.db.commit()
            else:
                self.db.rollback()
        finally:
            self.db.close()
        return False


def init_db():
    """Create any missing tables.

    Alembic owns schema evolution (`alembic upgrade head`); this exists so a
    fresh local checkout and the test suite can stand up a database without
    running migrations first. It creates missing tables but never alters
    existing ones, so it is not a substitute for a migration.
    """
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print(f"Created {len(Base.metadata.tables)} tables at {db_url}")
