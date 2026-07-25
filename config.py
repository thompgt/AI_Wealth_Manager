from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Wealth Manager"

    # "development" relaxes the startup safety checks in server.py (see
    # `validate_for_environment`). Anything else is treated as a real
    # deployment and refuses to boot with placeholder secrets.
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: str = "INFO"

    # Dummy URL for local testing with SQLite if NEON_DATABASE_URL is absent
    NEON_DATABASE_URL: str = "sqlite:///./wealth_manager.db"

    # --- LLM -----------------------------------------------------------------
    GEMINI_API_KEY: str = "DUMMY_API_KEY"
    # Model id used by every LLM-backed agent (market regime, stock research,
    # finance report). Kept in config rather than hardcoded per agent so a
    # model deprecation is a one-line config change, not a code change in
    # three files. gemini-1.5-* was retired by Google; 2.5-flash is the
    # current cost/latency-appropriate default for this workload.
    GEMINI_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.2
    LLM_TIMEOUT_SECONDS: int = 60
    # Transient failures (429 rate limits, 503s) are retried with exponential
    # backoff before an agent gives up and drops to its deterministic
    # fallback. Without this a single rate-limit blip silently degrades a
    # whole run.
    LLM_MAX_ATTEMPTS: int = 3
    LLM_BACKOFF_SECONDS: float = 2.0

    # Analysis settings (used by the Portfolio Diagnostics agent)
    RISK_FREE_RATE: float = 0.04  # 4% risk-free rate for Sharpe ratio
    LOOKBACK_PERIOD_YEARS: int = 1

    # Market data cache TTL, in hours, before a cached price is refetched
    MARKET_DATA_CACHE_TTL_HOURS: int = 24
    # In-process TTL, in minutes, for yfinance `Ticker.info` lookups. These
    # are hit repeatedly within a single run (stock research screens ~30
    # names, suitability re-checks each survivor, diagnostics looks up
    # sectors) and are the dominant source of run latency without a cache.
    TICKER_INFO_CACHE_TTL_MINUTES: int = 60

    # LangGraph checkpointer. Human-in-the-loop approvals pause a run and
    # resume it on a later HTTP request, so the checkpoint must outlive the
    # process and be shared across workers -- an in-memory saver silently
    # loses every pending approval on restart and breaks entirely under
    # `uvicorn --workers > 1`.
    CHECKPOINT_DB_PATH: str = "./checkpoints.sqlite"

    # API auth: a single shared key checked via the X-API-Key header.
    # Change this in .env for any non-local deployment.
    API_AUTH_KEY: str = "DEV_ONLY_CHANGE_ME"
    # Comma-separated list of origins allowed to call the API from a browser.
    CORS_ALLOW_ORIGINS: str = "http://localhost:8765,http://localhost:8766"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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

    def validate_for_environment(self) -> list[str]:
        """Returns a list of fatal misconfigurations for the current
        ENVIRONMENT. Empty list means safe to boot."""
        if self.is_development:
            return []

        problems = []
        if self.API_AUTH_KEY == "DEV_ONLY_CHANGE_ME":
            problems.append(
                "API_AUTH_KEY is still the built-in development default -- "
                "every caller with the public source can authenticate. Set a "
                "real secret in the environment."
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
        return problems


settings = Settings()
