"""Centralized application configuration with fail-fast startup validation."""

from functools import lru_cache
from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated settings; environment variables override .env values."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_title: str = "Research and Writing Agent Service"
    environment: str = Field(
        default="development", pattern="^(development|staging|production)$"
    )
    log_level: str = "INFO"

    # Pool limits apply per process; max >= min is not validated here.
    database_url: PostgresDsn = Field(
        default="postgresql://postgres:postgres@localhost:5432/checkpoints"
    )
    db_pool_min_size: int = Field(default=2, ge=1)
    db_pool_max_size: int = Field(default=10, ge=2)

    # Never log raw API keys or DATABASE_URL, which can contain credentials.
    groq_api_key: SecretStr
    tavily_api_key: SecretStr
    researcher_model_name: str = "qwen/qwen3.8-27b"
    writer_model_name: str = "openai/gpt-oss-120b"
    llm_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    llm_timeout: float = Field(default=30.0, gt=0.0)  # Seconds per provider request.
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    tavily_max_results: int = Field(default=3, ge=1, le=10)


    # Additional research passes requested by reviewers, excluding the first.
    max_redos: int = Field(default=2, ge=0, le=5)


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process; tests can reset with cache_clear()."""
    return Settings()
