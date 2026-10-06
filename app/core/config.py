"""Application configuration, loaded from environment variables / `.env` via pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.modify"


class DatabaseSettings(BaseSettings):
    """Database-only settings.

    Kept separate so Alembic can run migrations without requiring every
    application secret (OAuth, Anthropic, ...) to be present.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(description="PostgreSQL URL; postgres:// and postgresql:// are upgraded to asyncpg.")
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_echo: bool = False

    @field_validator("database_url")
    @classmethod
    def use_async_driver(cls, value: str) -> str:
        # Railway/Heroku expose sync-style URLs; SQLAlchemy's async engine needs an explicit driver.
        for prefix in ("postgres://", "postgresql://"):
            if value.startswith(prefix):
                return "postgresql+asyncpg://" + value.removeprefix(prefix)
        return value


class Settings(DatabaseSettings):
    app_name: str = "Email AI Agent"
    environment: Literal["development", "production"] = "production"
    log_level: str = "INFO"

    # Sessions
    secret_key: SecretStr
    session_cookie_name: str = "email_agent_session"
    session_https_only: bool = True
    session_max_age_seconds: int = 60 * 60 * 24 * 7

    # Google OAuth / Gmail
    google_client_id: str
    google_client_secret: SecretStr
    # When unset, the callback URL is derived from the incoming request.
    google_redirect_uri: str | None = None
    gmail_unread_query: str = "is:unread newer_than:1d"
    # Applied to emails the agent decides not to answer, so they stay findable instead of vanishing.
    gmail_ignored_label: str = "AI-Ignored"

    # Anthropic
    anthropic_api_key: SecretStr
    anthropic_model: str = "claude-sonnet-4-6"

    # Background agent (Gmail poller in the web process)
    agent_poll_interval_seconds: float = 60
    agent_error_backoff_seconds: float = 30
    reply_limit_per_sender_per_hour: int = 3

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    task_state_ttl_seconds: int = 60 * 60 * 24 * 7
    task_soft_time_limit_seconds: int = 240
    task_time_limit_seconds: int = 300

    # Retry policy for transient LLM API failures (429 / 503 / 529, connection errors)
    llm_max_retries: int = 6
    llm_request_timeout_seconds: float = 60
    llm_retry_base_seconds: float = 2
    llm_retry_max_seconds: float = 300

    # Knowledge base uploads
    max_upload_bytes: int = 10 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()  # values come from the environment
