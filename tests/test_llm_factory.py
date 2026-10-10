"""Factory validation tests."""

from __future__ import annotations

import pytest

from selflearn.config import Config
from selflearn.llm.factory import provider_from_config


def test_factory_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        provider_from_config(Config(provider="openai_compatible", base_url="", model="m"))
    with pytest.raises(ValueError):
        provider_from_config(Config(provider="anthropic", api_key="", model="m"))
    with pytest.raises(ValueError):
        provider_from_config(Config(provider="ollama", model=""))
    with pytest.raises(ValueError):
        provider_from_config(Config(provider="nope"))


def test_factory_builds_each_provider() -> None:
    assert provider_from_config(Config(provider="fake")).name == "fake"
    assert (
        provider_from_config(
            Config(provider="openai_compatible", base_url="http://x", model="m")
        ).name
        == "openai_compatible"
    )
    assert (
        provider_from_config(Config(provider="anthropic", api_key="k", model="m")).name
        == "anthropic"
    )
    assert provider_from_config(Config(provider="ollama", model="m")).name == "ollama"
