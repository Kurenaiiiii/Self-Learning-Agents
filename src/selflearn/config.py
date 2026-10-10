"""Typed configuration from env vars and optional config file.

Precedence: explicit constructor args > environment > config file > defaults.
Secrets are never logged; use :func:`redacted` for safe display.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


def load_dotenv(path: str = ".env") -> None:
    """Tiny stdlib .env loader. Never overrides real environment vars."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


VALID_PROVIDERS = ("openai_compatible", "anthropic", "ollama", "fake")


@dataclass(frozen=True)
class Config:
    """Validated runtime configuration."""

    provider: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    db_path: str = "./data/selflearn.db"
    log_level: str = "INFO"
    api_token: str = ""
    cache_dir: str = "./data/cache"
    timeout_s: float = 60.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        if self.provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Unknown provider {self.provider!r}. "
                f"Choose from: {', '.join(VALID_PROVIDERS)}"
            )
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")

    @property
    def has_api_key(self) -> bool:
        """Whether any API key is configured (never returns the key)."""
        return bool(self.api_key)


def _from_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def load_config(config_file: str | None = None) -> Config:
    """Load config from file (optional JSON) then environment."""
    load_dotenv()
    file_vals: dict[str, str] = {}
    if config_file and os.path.exists(config_file):
        with open(config_file, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            file_vals = {str(k): str(v) for k, v in raw.items()}

    def pick(file_key: str, *env_names: str, default: str = "") -> str:
        for env in env_names:
            if os.environ.get(env):
                return os.environ[env]
        if file_key in file_vals:
            return file_vals[file_key]
        return default

    provider = pick("provider", "SELFLEARN_PROVIDER", default="openai_compatible")
    base_url = pick("base_url", "SELFLEARN_BASE_URL")
    # ponytail: conventional fallbacks, no new abstraction.
    api_key = pick("api_key", "SELFLEARN_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    model = pick("model", "SELFLEARN_MODEL")
    db_path = pick("db_path", "SELFLEARN_DB_PATH", default="./data/selflearn.db")
    log_level = pick("log_level", "SELFLEARN_LOG_LEVEL", default="INFO")
    api_token = pick("api_token", "SELFLEARN_API_TOKEN")
    cache_dir = pick("cache_dir", "SELFLEARN_CACHE_DIR", default="./data/cache")

    return Config(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        db_path=db_path,
        log_level=log_level.upper(),
        api_token=api_token,
        cache_dir=cache_dir,
    )


def redacted(value: str, keep: int = 0) -> str:
    """Safe display of a secret: never returns the value itself."""
    if not value:
        return "(empty)"
    if keep <= 0:
        return "(set)" if value else "(empty)"
    return f"(set, ends ...{value[-keep:]})" if len(value) > keep else "(set)"


# Re-exported names for convenience.
__all__ = ["Config", "VALID_PROVIDERS", "load_config", "load_dotenv", "redacted"]
