"""Phase 1 tests: config validation and log scrubbing."""

from __future__ import annotations

import pytest

from selflearn.config import Config, load_config, redacted
from selflearn.log import get_logger, scrub


def test_config_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        Config(provider="wat")


def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    for var in (
        "SELFLEARN_PROVIDER",
        "SELFLEARN_BASE_URL",
        "SELFLEARN_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "SELFLEARN_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = load_config()
    assert cfg.provider == "openai_compatible"
    assert cfg.has_api_key is False


def test_load_config_env_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELFLEARN_PROVIDER", "ollama")
    monkeypatch.setenv("SELFLEARN_MODEL", "llama3")
    assert load_config().model == "llama3"


def test_redacted_never_returns_secret() -> None:
    assert redacted("gsk_secret123") == "(set)"
    assert "secret" not in redacted("gsk_secret123")


def test_scrub_removes_keys() -> None:
    payload = {"api_key": "sk-abc", "nested": {"token": "x"}, "ok": 1}
    out = scrub(payload)
    assert out["api_key"] == "(redacted)"
    assert out["nested"] == {"token": "(redacted)"}
    assert out["ok"] == 1


def test_logger_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    log = get_logger("selflearn.test.phase1", level="INFO")
    log.info("hello phase1")
    out = capsys.readouterr().err
    assert "hello phase1" in out
