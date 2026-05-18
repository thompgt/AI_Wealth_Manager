from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Dummy URL for local testing with SQLite if NEON_DATABASE_URL is absent
    NEON_DATABASE_URL: str = "sqlite:///./wealth_manager.db"
    GEMINI_API_KEY: str = "DUMMY_API_KEY"
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
