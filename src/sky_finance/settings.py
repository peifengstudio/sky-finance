"""
Application settings — validated at startup.

Two layers:
  EnvSettings  — environment variables (and .env file), validated with pydantic-settings
  TomlSettings — config/settings.toml sections, validated with pydantic BaseModel

Quick-start
-----------
Call once at every process entry point (FastAPI lifespan, Celery worker signal):

    from sky_finance.settings import validate_settings
    validate_settings()   # prints a clear error and sys.exit(1) on failure

Typed access in new code:

    from sky_finance.settings import get_settings
    s = get_settings()
    print(s.env.database_url)
    print(s.toml.llm.model)

Existing code that calls os.environ.get(...) or reads settings.toml directly
continues to work unchanged — validation is additive, not a replacement.

SecretStr
---------
openai_api_key, anthropic_api_key, and slack_bot_token are stored as
pydantic.SecretStr so their values are redacted in logs and __repr__:

    >>> settings.env.openai_api_key
    SecretStr('**********')
    >>> settings.env.openai_api_key.get_secret_value()
    'sk-proj-...'

The SDK clients (OpenAI, Anthropic) read directly from the environment so you
do not need to call get_secret_value() unless you are instantiating a client
manually.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parents[2]  # …/sky-finance/
_SETTINGS_TOML = _PROJECT_ROOT / "config" / "settings.toml"
_ENV_FILE = _PROJECT_ROOT / ".env"


# ---------------------------------------------------------------------------
# Layer 1 — Environment variables
# ---------------------------------------------------------------------------


class EnvSettings(BaseSettings):
    """
    All environment variables consumed by sky-finance.

    Priority: real environment → .env file → field default.
    Unknown variables are silently ignored (extra="ignore").
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "postgresql://skyfinance:skyfinance@localhost:5432/skyfinance"

    # ── API keys  (SecretStr — values never appear in logs or repr) ──────────
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # ── Slack ─────────────────────────────────────────────────────────────────
    slack_bot_token: SecretStr = SecretStr("")
    slack_channel: str = "#sky-finance"
    slack_dev_channel: str = "#dev-logs"

    # ── Redis / Celery ────────────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b-instruct"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Celery task limits ────────────────────────────────────────────────────
    celery_task_soft_limit: int = Field(default=300, ge=1)
    celery_task_hard_limit: int = Field(default=600, ge=1)

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_backend: Literal["ollama", "openai"] = "ollama"

    @model_validator(mode="after")
    def _task_limits_order(self) -> EnvSettings:
        if self.celery_task_soft_limit >= self.celery_task_hard_limit:
            raise ValueError(
                f"CELERY_TASK_SOFT_LIMIT ({self.celery_task_soft_limit}) must be "
                f"less than CELERY_TASK_HARD_LIMIT ({self.celery_task_hard_limit})"
            )
        return self


# ---------------------------------------------------------------------------
# Layer 2 — config/settings.toml sections
# ---------------------------------------------------------------------------


class DatabaseConfig(BaseModel):
    min_connections: int = Field(default=2, ge=1)
    max_connections: int = Field(default=10, ge=1)
    max_waiting: int = Field(default=20, ge=0)
    connect_timeout: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def _min_le_max(self) -> DatabaseConfig:
        if self.min_connections > self.max_connections:
            raise ValueError("min_connections must be ≤ max_connections")
        return self


class LLMConfig(BaseModel):
    model: str = "qwen2.5:3b-instruct"
    base_url: str = "http://localhost:11434"
    max_tokens: int = Field(default=512, ge=1, le=131_072)


class EmbeddingsConfig(BaseModel):
    backend: Literal["ollama", "openai"] = "ollama"
    model: str = "nomic-embed-text"
    ollama_host: str = "http://localhost:11434"
    openai_model: str = "text-embedding-3-small"
    dimensions: int = Field(default=768, ge=1)
    batch_size: int = Field(default=32, ge=1, le=2048)


class ModelPricingConfig(BaseModel):
    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float = Field(default=0.0, ge=0)
    cache_write: float = Field(default=0.0, ge=0)


class ModelTierConfig(BaseModel):
    provider: Literal["ollama", "openai", "claude"]
    model: str
    base_url: str | None = None
    max_tokens: int = Field(default=2048, ge=1, le=131_072)
    pricing: ModelPricingConfig | None = None


class TomlSettings(BaseModel):
    database: DatabaseConfig = DatabaseConfig()
    llm: LLMConfig = LLMConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    models: dict[str, ModelTierConfig] = {}


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    env: EnvSettings
    toml: TomlSettings


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class ConfigurationError(Exception):
    """Raised by validate_settings() when one or more config values are invalid."""


def _load_toml() -> dict[str, Any]:
    if not _SETTINGS_TOML.exists():
        return {}
    with _SETTINGS_TOML.open("rb") as f:
        result: dict[str, Any] = tomllib.load(f)
        return result


def _fmt_error(exc: Exception, source: str) -> str:
    """Format a pydantic ValidationError into a human-readable block."""
    from pydantic import ValidationError

    if not isinstance(exc, ValidationError):
        return f"  {source}: {exc}"

    lines = [f"  {source}:"]
    for err in exc.errors():
        loc = " → ".join(str(p) for p in err["loc"]) if err["loc"] else "(root)"
        lines.append(f"    {loc}: {err['msg']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_settings() -> Settings:
    """
    Load and validate all configuration.

    Collects errors from both layers before raising so operators see everything
    wrong in a single run — no fix-one-restart-fix-another cycle.

    Raises:
        ConfigurationError: human-readable message listing every bad field.

    Example error output::

        ❌  sky-finance configuration is invalid:

          Environment (.env / shell):
            log_level: Input should be 'DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'
            celery_task_soft_limit: Input should be greater than or equal to 1

          config/settings.toml:
            database → min_connections: Input should be greater than or equal to 1

        Fix the above and restart.
    """
    from pydantic import ValidationError

    errors: list[str] = []
    env: EnvSettings | None = None
    toml: TomlSettings | None = None

    try:
        env = EnvSettings()
    except ValidationError as exc:
        errors.append(_fmt_error(exc, "Environment (.env / shell)"))

    try:
        toml = TomlSettings.model_validate(_load_toml())
    except ValidationError as exc:
        errors.append(_fmt_error(exc, "config/settings.toml"))

    if errors:
        body = "\n\n".join(errors)
        raise ConfigurationError(
            f"\n❌  sky-finance configuration is invalid:\n\n{body}\n\nFix the above and restart."
        )

    assert env is not None and toml is not None  # reached only when no errors
    return Settings(env=env, toml=toml)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the validated Settings singleton (cached after first call).

    Use this in application code for typed access.  Startup hooks should call
    ``validate_settings()`` directly so they control the error/exit behaviour.
    """
    return validate_settings()
