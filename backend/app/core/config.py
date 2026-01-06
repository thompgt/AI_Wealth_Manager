import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Wealth Manager"
    GOOGLE_API_KEY: str
    DATABASE_URL: str = "duckdb:///wealth_manager.db"
    
    # Analysis Settings
    RISK_FREE_RATE: float = 0.04  # 4% risk free rate for Sharpe Ratio
    LOOKBACK_PERIOD_YEARS: int = 1

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
