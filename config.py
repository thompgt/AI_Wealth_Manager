from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Wealth Manager"

    # Dummy URL for local testing with SQLite if NEON_DATABASE_URL is absent
    NEON_DATABASE_URL: str = "sqlite:///./wealth_manager.db"
    GEMINI_API_KEY: str = "DUMMY_API_KEY"

    # Analysis settings (used by the Portfolio Diagnostics agent)
    RISK_FREE_RATE: float = 0.04  # 4% risk-free rate for Sharpe ratio
    LOOKBACK_PERIOD_YEARS: int = 1

    # Market data cache TTL, in hours, before a cached price is refetched
    MARKET_DATA_CACHE_TTL_HOURS: int = 24

    # API auth: a single shared key checked via the X-API-Key header.
    # Change this in .env for any non-local deployment.
    API_AUTH_KEY: str = "DEV_ONLY_CHANGE_ME"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
