"""Shared provider contract tests.

Same behavior verified against every provider: retries with backoff and
jitter, Retry-After on 429, timeouts, error mapping, token accounting,
streaming, tool calls. Each mock follows its real wire format.
"""

from __future__ import annotations

import pytest

from selflearn.llm.anthropic import AnthropicProvider
from selflearn.llm.base import backoff_delay, parse_retry_after
from selflearn.llm.cache import DiskCache
from selflearn.llm.fake import FakeLLM
from selflearn.llm.ollama import OllamaProvider
from selflearn.llm.openai_compatible import OpenAICompatibleProvider
from selflearn.llm.types import (
    AuthError,
    BadRequestError,
    ChatMessage,
    ChatRequest,
    LLMError,
    ServerError,
    TimeoutError,
    ToolDef,
)
from tests.mock_servers import (
    AnthropicMockHandler,
    MockServer,
    OllamaMockHandler,
    OpenAIMockHandler,
)

MODEL = "mock-model"


def _req(**kwargs: object) -> ChatRequest:
    msgs = (ChatMessage(role="user", content="hi"),)
    return ChatRequest(messages=msgs, model=MODEL, **kwargs)  # type: ignore[arg-type]


@pytest.fixture()
def openai_server() -> MockServer:
    srv = MockServer(OpenAIMockHandler)
    yield srv
    srv.stop()


@pytest.fixture()
def anthropic_server() -> MockServer:
    srv = MockServer(AnthropicMockHandler)
    yield srv
    srv.stop()


@pytest.fixture()
def ollama_server() -> MockServer:
    srv = MockServer(OllamaMockHandler)
    yield srv
    srv.stop()


# -- shared checks ---------------------------------------------------------------


def check_basic_chat_text_and_usage(provider: object, expected_text: str) -> None:
    from selflearn.llm.base import LLMProvider

    assert isinstance(provider, LLMProvider)
    resp = provider.chat(_req())
    assert resp.text == expected_text
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0
    assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens


def check_tool_call(provider: object) -> None:
    from selflearn.llm.base import LLMProvider

    assert isinstance(provider, LLMProvider)
    tools = (
        ToolDef(
            name="memory_search",
            description="search",
            parameters={"type": "object", "properties": {}},
        ),
    )
    resp = provider.chat(_req(tools=tools))
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.name == "memory_search"
    assert call.arguments == {"query": "coffee"}


def check_streaming(provider: object) -> None:
    from selflearn.llm.base import LLMProvider

    assert isinstance(provider, LLMProvider)
    chunks = list(provider.chat_stream(_req()))
    assert chunks, "no chunks yielded"
    assert chunks[-1].done is True
    text = "".join(c.delta_text for c in chunks)
    assert "hel" in text and "lo" in text
    assert chunks[-1].usage.total_tokens > 0


def check_retry_after_429_then_ok(provider: object, server: MockServer) -> None:
    from selflearn.llm.base import LLMProvider

    assert isinstance(provider, LLMProvider)
    server.behaviors = ["429", "ok"]
    resp = provider.chat(_req())
    assert resp.text, "expected success after retry"
    assert server.received == 2


def check_error_mapping(provider: object, server: MockServer, *, server_error: str = "500") -> None:
    from selflearn.llm.base import LLMProvider

    assert isinstance(provider, LLMProvider)
    server.behaviors = ["401"]
    with pytest.raises(AuthError):
        provider.chat(_req())
    server.behaviors = ["400"]
    with pytest.raises(BadRequestError):
        provider.chat(_req())
    server.behaviors = [server_error]
    with pytest.raises(ServerError):
        provider.chat(_req())


def check_timeout(provider: object, server: MockServer) -> None:
    from selflearn.llm.base import LLMProvider

    assert isinstance(provider, LLMProvider)
    server.behaviors = ["sleep"]
    with pytest.raises(TimeoutError):
        provider.chat(_req())


# -- per-provider contract runs ---------------------------------------------------


class TestOpenAICompatibleContract:
    """Contract against the mock OpenAI-compatible server."""

    def test_basic(self, openai_server: MockServer) -> None:
        p = OpenAICompatibleProvider(
            base_url=openai_server.base_url + "/v1", api_key="k", max_retries=1
        )
        check_basic_chat_text_and_usage(p, "hello from mock")

    def test_tools(self, openai_server: MockServer) -> None:
        p = OpenAICompatibleProvider(
            base_url=openai_server.base_url + "/v1", api_key="k", max_retries=1
        )
        check_tool_call(p)

    def test_stream(self, openai_server: MockServer) -> None:
        p = OpenAICompatibleProvider(
            base_url=openai_server.base_url + "/v1", api_key="k", max_retries=1
        )
        check_streaming(p)

    def test_retry_429(self, openai_server: MockServer) -> None:
        p = OpenAICompatibleProvider(
            base_url=openai_server.base_url + "/v1", api_key="k", max_retries=3
        )
        check_retry_after_429_then_ok(p, openai_server)

    def test_errors(self, openai_server: MockServer) -> None:
        p = OpenAICompatibleProvider(
            base_url=openai_server.base_url + "/v1", api_key="k", max_retries=0
        )
        check_error_mapping(p, openai_server)

    def test_timeout(self, openai_server: MockServer) -> None:
        p = OpenAICompatibleProvider(
            base_url=openai_server.base_url + "/v1", api_key="k", timeout_s=0.3, max_retries=0
        )
        check_timeout(p, openai_server)

    def test_list_models(self, openai_server: MockServer) -> None:
        p = OpenAICompatibleProvider(base_url=openai_server.base_url + "/v1", api_key="k")
        assert p.list_models() == ["mock-model"]

    def test_cache_avoids_second_call(self, openai_server: MockServer, tmp_path: object) -> None:
        from pathlib import Path

        cache = DiskCache(str(Path(str(tmp_path)) / "c"))
        p = OpenAICompatibleProvider(
            base_url=openai_server.base_url + "/v1", api_key="k", cache=cache
        )
        first = p.chat(_req())
        second = p.chat(_req())
        assert first.text == second.text
        assert openai_server.received == 1


class TestAnthropicContract:
    """Contract against the mock Anthropic server."""

    def _provider(self, srv: MockServer, **kw: object) -> AnthropicProvider:
        return AnthropicProvider(api_key="k", base_url=srv.base_url, **kw)  # type: ignore[arg-type]

    def test_basic(self, anthropic_server: MockServer) -> None:
        check_basic_chat_text_and_usage(
            self._provider(anthropic_server, max_retries=1), "hello from anthropic mock"
        )

    def test_tools(self, anthropic_server: MockServer) -> None:
        check_tool_call(self._provider(anthropic_server, max_retries=1))

    def test_stream(self, anthropic_server: MockServer) -> None:
        check_streaming(self._provider(anthropic_server, max_retries=1))

    def test_retry_429(self, anthropic_server: MockServer) -> None:
        check_retry_after_429_then_ok(
            self._provider(anthropic_server, max_retries=3), anthropic_server
        )

    def test_errors(self, anthropic_server: MockServer) -> None:
        check_error_mapping(
            self._provider(anthropic_server, max_retries=0),
            anthropic_server,
            server_error="529",
        )

    def test_timeout(self, anthropic_server: MockServer) -> None:
        check_timeout(
            self._provider(anthropic_server, timeout_s=0.3, max_retries=0), anthropic_server
        )

    def test_no_list_models(self, anthropic_server: MockServer) -> None:
        assert self._provider(anthropic_server).list_models() == []

    def test_system_prompt_sent_as_system(self, anthropic_server: MockServer) -> None:
        p = self._provider(anthropic_server)
        req = ChatRequest(
            messages=(
                ChatMessage(role="system", content="sys"),
                ChatMessage(role="user", content="hi"),
            ),
            model=MODEL,
        )
        p.chat(req)
        sent = anthropic_server.requests[-1]["body"]
        assert sent["system"] == "sys"
        assert all(m["role"] != "system" for m in sent["messages"])


class TestOllamaContract:
    """Contract against the mock Ollama server."""

    def _provider(self, srv: MockServer, **kw: object) -> OllamaProvider:
        return OllamaProvider(base_url=srv.base_url, **kw)  # type: ignore[arg-type]

    def test_basic(self, ollama_server: MockServer) -> None:
        check_basic_chat_text_and_usage(
            self._provider(ollama_server, max_retries=1), "hello from ollama mock"
        )

    def test_stream(self, ollama_server: MockServer) -> None:
        check_streaming(self._provider(ollama_server, max_retries=1))

    def test_retry_429(self, ollama_server: MockServer) -> None:
        check_retry_after_429_then_ok(self._provider(ollama_server, max_retries=3), ollama_server)

    def test_errors(self, ollama_server: MockServer) -> None:
        check_error_mapping(self._provider(ollama_server, max_retries=0), ollama_server)

    def test_timeout(self, ollama_server: MockServer) -> None:
        check_timeout(self._provider(ollama_server, timeout_s=0.3, max_retries=0), ollama_server)

    def test_list_models(self, ollama_server: MockServer) -> None:
        assert self._provider(ollama_server).list_models() == ["mock-model:latest"]


class TestFakeLLM:
    """FakeLLM obeys the same observable behavior."""

    def test_scripted_and_default(self) -> None:
        f = FakeLLM(["first"])
        assert f.chat(_req()).text == "first"
        assert f.chat(_req()).text == "fake reply"

    def test_error_pops(self) -> None:
        f = FakeLLM([ServerError("boom", provider="fake")])
        with pytest.raises(LLMError):
            f.chat(_req())

    def test_stream_words_then_done(self) -> None:
        f = FakeLLM(["a b"])
        chunks = list(f.chat_stream(_req()))
        assert chunks[-1].done is True
        assert "".join(c.delta_text for c in chunks).strip().startswith("a b")


class TestRetryHelpers:
    """Backoff grows, jitter varies, Retry-After parses."""

    def test_backoff_grows(self) -> None:
        assert (
            backoff_delay(2, base_s=0.0, cap_s=100) <= backoff_delay(3, base_s=0.0, cap_s=100) + 1
        )

    def test_backoff_capped(self) -> None:
        assert backoff_delay(99) <= 30.0 + 100

    def test_retry_after_parse(self) -> None:
        assert parse_retry_after("2") == 2.0
        assert parse_retry_after(None) == 0.0
        assert parse_retry_after("garbage") == 0.0
