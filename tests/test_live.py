"""Optional live provider tests. Skipped unless explicitly enabled. Never run in CI.

Enable: SELFLEARN_LIVE=1 (uses SELFLEARN_* env config for openai_compatible),
SELFLEARN_LIVE_ANTHROPIC=1 (needs ANTHROPIC_API_KEY), SELFLEARN_LIVE_OLLAMA=1
(needs a local server at SELFLEARN_BASE_URL or http://localhost:11434).
"""

from __future__ import annotations

import os

import pytest

from selflearn.config import load_config
from selflearn.llm.factory import provider_from_config
from selflearn.llm.types import ChatMessage, ChatRequest

needs_live = pytest.mark.skipif(
    os.environ.get("SELFLEARN_LIVE") != "1", reason="live test: set SELFLEARN_LIVE=1"
)
needs_anthropic = pytest.mark.skipif(
    os.environ.get("SELFLEARN_LIVE_ANTHROPIC") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="live test: set SELFLEARN_LIVE_ANTHROPIC=1 with ANTHROPIC_API_KEY",
)
needs_ollama = pytest.mark.skipif(
    os.environ.get("SELFLEARN_LIVE_OLLAMA") != "1", reason="live test: set SELFLEARN_LIVE_OLLAMA=1"
)


@needs_live
def test_live_openai_compatible_tiny_chat() -> None:
    cfg = load_config()
    provider = provider_from_config(cfg)
    resp = provider.chat(
        ChatRequest(
            messages=(ChatMessage(role="user", content="Reply with exactly: ok"),),
            model=cfg.model,
            temperature=0.0,
            max_tokens=64,
        )
    )
    assert "ok" in resp.text.lower()
    assert resp.usage.total_tokens > 0


@needs_anthropic
def test_live_anthropic_tiny_chat() -> None:
    cfg = load_config()
    assert cfg.provider == "anthropic"
    provider = provider_from_config(cfg)
    resp = provider.chat(
        ChatRequest(
            messages=(ChatMessage(role="user", content="Reply with exactly: ok"),),
            model=cfg.model,
            temperature=0.0,
            max_tokens=64,
        )
    )
    assert "ok" in resp.text.lower()


@needs_ollama
def test_live_ollama_tiny_chat() -> None:
    cfg = load_config()
    assert cfg.provider == "ollama"
    provider = provider_from_config(cfg)
    resp = provider.chat(
        ChatRequest(
            messages=(ChatMessage(role="user", content="Reply with exactly: ok"),),
            model=cfg.model,
            temperature=0.0,
            max_tokens=64,
        )
    )
    assert resp.text.strip()
