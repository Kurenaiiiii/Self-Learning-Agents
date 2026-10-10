"""Shared normalized types for all LLM providers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """One normalized tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolDef:
    """One tool the model may call (OpenAI-style function shape)."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatMessage:
    """One normalized chat message."""

    role: str  # system | user | assistant | tool
    content: str = ""
    name: str = ""  # tool name for tool results
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""  # for role=tool: which call this answers


@dataclass(frozen=True)
class ChatRequest:
    """Provider-agnostic chat request. Model comes from config, not code."""

    messages: tuple[ChatMessage, ...]
    model: str
    tools: tuple[ToolDef, ...] = ()
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass(frozen=True)
class Usage:
    """Normalized token accounting across providers."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens in + out."""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ChatResponse:
    """Normalized non-streaming chat response."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    model: str = ""


@dataclass(frozen=True)
class ChatChunk:
    """One normalized streaming delta."""

    delta_text: str
    done: bool = False
    usage: Usage = field(default_factory=Usage)


# Streaming iterator type shared by all providers.
Stream = Iterator[ChatChunk]


class LLMError(Exception):
    """Base error for all providers. Never includes secrets."""

    def __init__(self, message: str, *, provider: str = "", retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class AuthError(LLMError):
    """Bad/missing credentials (HTTP 401/403). Never retried."""

    def __init__(self, message: str, *, provider: str = "") -> None:
        super().__init__(message, provider=provider, retryable=False)


class RateLimitError(LLMError):
    """Rate limited (HTTP 429). Retried honoring Retry-After."""

    def __init__(self, message: str, *, provider: str = "", retry_after_s: float = 0.0) -> None:
        super().__init__(message, provider=provider, retryable=True)
        self.retry_after_s = retry_after_s


class BadRequestError(LLMError):
    """Invalid request (HTTP 400/404/422). Never retried."""

    def __init__(self, message: str, *, provider: str = "") -> None:
        super().__init__(message, provider=provider, retryable=False)


class ServerError(LLMError):
    """Upstream 5xx. Retried with backoff."""

    def __init__(self, message: str, *, provider: str = "") -> None:
        super().__init__(message, provider=provider, retryable=True)


class TimeoutError(LLMError):
    """Request timed out. Retried with backoff."""

    def __init__(self, message: str, *, provider: str = "") -> None:
        super().__init__(message, provider=provider, retryable=True)


class NetworkError(LLMError):
    """Connection-level failure. Retried with backoff."""

    def __init__(self, message: str, *, provider: str = "") -> None:
        super().__init__(message, provider=provider, retryable=True)


class ParseError(LLMError):
    """Upstream returned unparseable data. Not retried (would repeat)."""

    def __init__(self, message: str, *, provider: str = "") -> None:
        super().__init__(message, provider=provider, retryable=False)


__all__ = [
    "AuthError",
    "BadRequestError",
    "ChatChunk",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "LLMError",
    "NetworkError",
    "ParseError",
    "RateLimitError",
    "ServerError",
    "Stream",
    "TimeoutError",
    "ToolCall",
    "ToolDef",
    "Usage",
]
