"""LLM layer: provider-agnostic chat, tools, streaming, usage, retries, cache."""

from selflearn.llm.anthropic import AnthropicProvider
from selflearn.llm.base import LLMProvider
from selflearn.llm.cache import DiskCache, request_hash
from selflearn.llm.factory import provider_from_config
from selflearn.llm.fake import FakeLLM
from selflearn.llm.ollama import OllamaProvider
from selflearn.llm.openai_compatible import OpenAICompatibleProvider
from selflearn.llm.types import (
    AuthError,
    BadRequestError,
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    LLMError,
    NetworkError,
    ParseError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ToolCall,
    ToolDef,
    Usage,
)

__all__ = [
    "AnthropicProvider",
    "AuthError",
    "BadRequestError",
    "ChatChunk",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "DiskCache",
    "FakeLLM",
    "LLMError",
    "LLMProvider",
    "NetworkError",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "ParseError",
    "RateLimitError",
    "ServerError",
    "TimeoutError",
    "ToolCall",
    "ToolDef",
    "Usage",
    "provider_from_config",
    "request_hash",
]
