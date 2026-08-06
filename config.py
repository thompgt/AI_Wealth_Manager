"""Central configuration.

Every tunable in the system resolves here or to a client's Investment Policy
Statement (see `services/policy.py`). The split is deliberate:

  * Settings in this file are *operational* -- which database, which model,
    how long to wait, how hard to retry. They are the same for every client.
  * Anything that changes what the system is willing to *recommend* -- position
    caps, sector limits, beta ceilings, minimum cash -- is NOT here. It lives
    in a versioned, per-client IPS row so that "what limits were in force when
    this recommendation was made?" has an auditable answer. A threshold in a
    Python constant cannot answer that question.
"""

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Wealth Manager"

    # "development" relaxes the startup safety checks in server.py (see
    # `validate_for_environment`). Anything else is treated as a real
    # deployment and refuses to boot with placeholder secrets.
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: str = "INFO"
    # Plain text is readable in a terminal; JSON is parseable by a log
    # aggregator. Production defaults to JSON via validate_for_environment's
    # sibling `effective_log_format`.
    LOG_FORMAT: Literal["text", "json", "auto"] = "auto"

    # Dummy URL for local testing with SQLite if NEON_DATABASE_URL is absent
    NEON_DATABASE_URL: str = "sqlite:///./wealth_manager.db"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_RECYCLE_SECONDS: int = 1800

    # --- LLM -----------------------------------------------------------------
    GEMINI_API_KEY: str = "DUMMY_API_KEY"
    # Model id used by every LLM-backed agent. Kept in config rather than
    # hardcoded per agent so a model deprecation is a one-line config change,
    # not a code change in five files. gemini-1.5-* was retired by Google;
    # 2.5-flash is the current cost/latency-appropriate default.
    GEMINI_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.2
    LLM_TIMEOUT_SECONDS: int = 60
    # Transient failures (429 rate limits, 503s) are retried with exponential
    # backoff before an agent gives up and drops to its deterministic
    # fallback. Without this a single rate-limit blip silently degrades a
    # whole run.
    LLM_MAX_ATTEMPTS: int = 3
    LLM_BACKOFF_SECONDS: float = 2.0
    # Per-1M-token prices used to attribute spend to a run. These change; they
    # are config, not constants, so a price change does not require a deploy
    # to keep cost reporting honest.
    LLM_INPUT_COST_PER_MTOK: float = 0.30
    LLM_OUTPUT_COST_PER_MTOK: float = 2.50
    # Hard ceiling on LLM spend for a single graph run. A retry loop that
    # misbehaves should cost a known maximum, not an unbounded one.
    LLM_RUN_BUDGET_USD: float = 1.00

    # --- Market data ---------------------------------------------------------
    # Ordered failover chain. The first provider with credentials configured
    # leads; the rest are fallbacks. yfinance needs no key and is always
    # available last, which is why the system runs out of the box.
    MARKET_DATA_PROVIDERS: str = "polygon,tiingo,yfinance"
    POLYGON_API_KEY: Optional[str] = None
    TIINGO_API_KEY: Optional[str] = None
    MARKET_DATA_TIMEOUT_SECONDS: float = 15.0
    MARKET_DATA_MAX_ATTEMPTS: int = 3
    MARKET_DATA_BACKOFF_SECONDS: float = 0.75
    # Consecutive failures before a provider is taken out of rotation, and how
    # long it stays out. Without this a dead provider is retried on every
    # ticker of every run, turning one outage into minutes of added latency.
    MARKET_DATA_CIRCUIT_FAIL_THRESHOLD: int = 5
    MARKET_DATA_CIRCUIT_RESET_SECONDS: float = 120.0
    # Requests per second per provider, enforced client-side. Yahoo bans IPs
    # that fan out ~200 `.info` lookups with no pacing.
    MARKET_DATA_RATE_LIMIT_PER_SEC: float = 8.0

    # Analysis settings (used by the Portfolio Diagnostics agent)
    RISK_FREE_RATE: float = 0.04  # 4% risk-free rate for Sharpe ratio
    LOOKBACK_PERIOD_YEARS: int = 1
    # Default benchmark for performance attribution. Overridable per client
    # via the IPS.
    DEFAULT_BENCHMARK: str = "SPY"

    # Market data cache TTL, in hours, before a cached price is refetched
    MARKET_DATA_CACHE_TTL_HOURS: int = 24
    # In-process TTL, in minutes, for security metadata lookups. These are hit
    # repeatedly within a single run (the screener evaluates hundreds of
    # names, suitability re-checks each survivor, diagnostics looks up
    # sectors) and are the dominant source of run latency without a cache.
    TICKER_INFO_CACHE_TTL_MINUTES: int = 60
    # Bounded so a long-lived worker screening a large universe cannot grow
    # the cache without limit.
    TICKER_INFO_CACHE_MAX_ENTRIES: int = 4096
    # Staleness is measured in *trading* days, not clock time. These providers
    # serve daily closes, so a Sunday-morning run legitimately sees a Friday
    # print; a wall-clock threshold would flag every weekend and holiday run
    # as degraded and train everyone to ignore the signal. A quote older than
    # this many sessions is recorded as stale and surfaced by the API.
    QUOTE_MAX_AGE_TRADING_DAYS: int = 2

    # LangGraph checkpointer. Human-in-the-loop approvals pause a run and
    # resume it on a later HTTP request, so the checkpoint must outlive the
    # process and be shared across workers -- an in-memory saver silently
    # loses every pending approval on restart and breaks entirely under
    # `uvicorn --workers > 1`.
    CHECKPOINT_DB_PATH: str = "./checkpoints.sqlite"

    # --- Auth ----------------------------------------------------------------
    # Signing secret for session JWTs. Must be set to a real secret outside
    # development; see validate_for_environment.
    JWT_SECRET: str = "DEV_ONLY_CHANGE_ME"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_TTL_MINUTES: int = 60
    REFRESH_TOKEN_TTL_DAYS: int = 14
    # Bootstrap administrator, created on first boot when no users exist. The
    # password is rejected outside development if left at the default.
    BOOTSTRAP_ADMIN_EMAIL: str = "admin@example.com"
    BOOTSTRAP_ADMIN_PASSWORD: str = "DEV_ONLY_CHANGE_ME"
    BOOTSTRAP_ORG_NAME: str = "Demo Advisory"
    # Failed logins allowed within the window before the account is locked.
    LOGIN_MAX_FAILURES: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 15

    # Comma-separated list of origins allowed to call the API from a browser.
    CORS_ALLOW_ORIGINS: str = "http://localhost:8765,http://localhost:8766"
    # Requests per minute per authenticated principal, and the much tighter
    # budget for the endpoint that triggers a multi-agent run.
    RATE_LIMIT_PER_MINUTE: int = 120
    RUN_RATE_LIMIT_PER_HOUR: int = 30

    # --- Observability -------------------------------------------------------
    # Serve GET /metrics without credentials. Off by default, and deliberately
    # so: the exposition carries per-route request counts, per-node LLM spend
    # and job queue depth, which together describe how much business is
    # flowing through the system and when it is struggling. That is not
    # something to hand to an unauthenticated caller on a public interface.
    #
    # Turn it on only for a local Prometheus, which has no way to present an
    # API key or a bearer token without a proxy in front of it. Everything
    # else stays gated; this flag affects exactly one route.
    METRICS_ALLOW_ANONYMOUS: bool = False

    # --- Job queue -----------------------------------------------------------
    # A graph run takes minutes. It executes on a background worker pool and
    # the API returns a job id immediately.
    JOB_WORKER_COUNT: int = 2
    JOB_POLL_INTERVAL_SECONDS: float = 1.0
    JOB_TIMEOUT_SECONDS: int = 900
    # A job that a worker claimed but never finished (process killed mid-run)
    # is reclaimed after this long without a heartbeat.
    JOB_HEARTBEAT_STALE_SECONDS: int = 120
    # Set false on API-only replicas so a dedicated worker process owns
    # execution.
    JOB_WORKER_ENABLED: bool = True

    # --- Trading -------------------------------------------------------------
    # Master switch. Orders are simulated regardless of this flag -- there is
    # no live-brokerage adapter in this system -- but it gates whether an
    # approved proposal is allowed to mutate portfolio state at all.
    TRADING_ENABLED: bool = True
    # Simulated fill assumptions, in basis points of notional.
    SIM_SLIPPAGE_BPS: float = 5.0
    SIM_COMMISSION_BPS: float = 0.0
    SIM_COMMISSION_MIN_USD: float = 0.0
    # T+1 settlement for US equities since May 2024.
    SETTLEMENT_DAYS: int = 1
    # Fractional shares are common at modern custodians but not universal.
    ALLOW_FRACTIONAL_SHARES: bool = True
    MIN_ORDER_NOTIONAL_USD: float = 100.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # -- Derived helpers ------------------------------------------------------

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def llm_configured(self) -> bool:
        """True when a real Gemini key is present. Agents use this to report
        honestly that they are running in deterministic-fallback mode rather
        than silently emitting degraded output that looks like AI analysis."""
        return bool(self.GEMINI_API_KEY) and self.GEMINI_API_KEY != "DUMMY_API_KEY"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.CORS_ALLOW_ORIGINS.split(",") if o.strip()]

    @property
    def market_data_provider_chain(self) -> list[str]:
        """Providers to try, in order, filtered to those actually usable.

        A provider named in MARKET_DATA_PROVIDERS but missing its API key is
        dropped here rather than failing at call time -- otherwise every
        request pays a doomed attempt plus retries before failing over.
        """
        requires_key = {"polygon": self.POLYGON_API_KEY, "tiingo": self.TIINGO_API_KEY}
        chain = []
        for name in (p.strip().lower() for p in self.MARKET_DATA_PROVIDERS.split(",")):
            if not name:
                continue
            if name in requires_key and not requires_key[name]:
                continue
            chain.append(name)
        # yfinance needs no credentials, so it is always a viable last resort.
        # Guaranteeing it here is what makes "works with no API keys" true.
        if "yfinance" not in chain:
            chain.append("yfinance")
        return chain

    @property
    def effective_log_format(self) -> str:
        if self.LOG_FORMAT != "auto":
            return self.LOG_FORMAT
        return "text" if self.is_development else "json"

    def validate_for_environment(self) -> list[str]:
        """Returns a list of fatal misconfigurations for the current
        ENVIRONMENT. Empty list means safe to boot."""
        if self.is_development:
            return []

        problems = []
        if self.JWT_SECRET == "DEV_ONLY_CHANGE_ME":
            problems.append(
                "JWT_SECRET is still the built-in development default -- anyone "
                "with the public source can mint a valid session token for any "
                "user, including an admin. Set a real secret in the environment."
            )
        if self.BOOTSTRAP_ADMIN_PASSWORD == "DEV_ONLY_CHANGE_ME":
            problems.append(
                "BOOTSTRAP_ADMIN_PASSWORD is still the built-in development "
                "default. Set a real password before first boot, or the "
                "bootstrap administrator is publicly known."
            )
        if not self.llm_configured:
            problems.append(
                "GEMINI_API_KEY is unset/placeholder -- every LLM-backed agent "
                "would run permanently in deterministic-fallback mode."
            )
        if self.NEON_DATABASE_URL.startswith("sqlite"):
            problems.append(
                "NEON_DATABASE_URL points at SQLite, which cannot safely back "
                "a multi-worker deployment. Point it at Postgres."
            )
        if self.CHECKPOINT_DB_PATH and self.NEON_DATABASE_URL.startswith("sqlite"):
            problems.append(
                "CHECKPOINT_DB_PATH is a local SQLite file. Pending approvals "
                "would be visible only to the worker that created them."
            )
        if any(o.startswith("http://localhost") for o in self.cors_origins):
            problems.append(
                "CORS_ALLOW_ORIGINS still contains a localhost origin. Set the "
                "real dashboard origin(s) for this deployment."
            )
        return problems


settings = Settings()
