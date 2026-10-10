"""`selflearn doctor`: config, connectivity, and model listing checks."""

from __future__ import annotations

from selflearn.config import Config, redacted
from selflearn.llm.factory import provider_from_config
from selflearn.llm.types import ChatMessage, ChatRequest, LLMError


def run_doctor(cfg: Config) -> str:
    """Check config and connectivity. Never includes secrets in output."""
    lines = [
        f"provider: {cfg.provider}",
        f"model: {cfg.model or '(not set)'}",
        f"base_url: {cfg.base_url or '(default)'}",
        f"api_key: {redacted(cfg.api_key)}",
        f"db_path: {cfg.db_path}",
    ]
    try:
        provider = provider_from_config(cfg)
    except ValueError as e:
        lines.append(f"config: ERROR {e}")
        return "\n".join(lines)
    lines.append("config: OK")
    try:
        models = provider.list_models()
    except LLMError as e:
        lines.append(f"connectivity: ERROR {e}")
        return "\n".join(lines)
    if models:
        shown = ", ".join(models[:10])
        more = f" (+{len(models) - 10} more)" if len(models) > 10 else ""
        lines.append(f"connectivity: OK ({len(models)} models: {shown}{more})")
        if cfg.model and cfg.model not in models:
            lines.append(f"warning: model {cfg.model!r} not in model list")
        return "\n".join(lines)
    # No list endpoint (or empty): one tiny ping instead.
    lines.append("models: list not supported by this API; sending tiny ping")
    try:
        resp = provider.chat(
            ChatRequest(
                messages=(ChatMessage(role="user", content="Reply with exactly: ok"),),
                model=cfg.model,
                temperature=0,
                max_tokens=64,
            )
        )
        lines.append(f"connectivity: OK (ping reply: {resp.text.strip()[:60]!r})")
        lines.append(
            f"usage: prompt={resp.usage.prompt_tokens} completion={resp.usage.completion_tokens}"
        )
    except LLMError as e:
        lines.append(f"connectivity: ERROR {e}")
    return "\n".join(lines)


__all__ = ["run_doctor"]
