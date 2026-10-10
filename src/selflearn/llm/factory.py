"""Build a provider from Config. Model always comes from config."""

from __future__ import annotations

from selflearn.config import Config
from selflearn.llm.anthropic import AnthropicProvider
from selflearn.llm.base import LLMProvider
from selflearn.llm.cache import DiskCache
from selflearn.llm.fake import FakeLLM
from selflearn.llm.ollama import OllamaProvider
from selflearn.llm.openai_compatible import OpenAICompatibleProvider


def provider_from_config(cfg: Config, *, cache: DiskCache | None = None) -> LLMProvider:
    """Instantiate the configured provider. Raises ValueError on bad config."""
    if cfg.provider == "fake":
        return FakeLLM()
    if cfg.provider == "openai_compatible":
        if not cfg.base_url:
            raise ValueError("openai_compatible needs SELFLEARN_BASE_URL")
        if not cfg.model:
            raise ValueError("openai_compatible needs SELFLEARN_MODEL")
        return OpenAICompatibleProvider(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout_s=cfg.timeout_s,
            max_retries=cfg.max_retries,
            cache=cache,
        )
    if cfg.provider == "anthropic":
        if not cfg.api_key:
            raise ValueError("anthropic needs SELFLEARN_API_KEY (or ANTHROPIC_API_KEY)")
        if not cfg.model:
            raise ValueError("anthropic needs SELFLEARN_MODEL")
        return AnthropicProvider(
            api_key=cfg.api_key,
            base_url=cfg.base_url or "https://api.anthropic.com",
            timeout_s=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )
    if cfg.provider == "ollama":
        if not cfg.model:
            raise ValueError("ollama needs SELFLEARN_MODEL")
        return OllamaProvider(
            base_url=cfg.base_url or "http://localhost:11434",
            timeout_s=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )
    raise ValueError(f"Unknown provider {cfg.provider!r}")


__all__ = ["provider_from_config"]
