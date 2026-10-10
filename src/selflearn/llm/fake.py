"""FakeLLM: deterministic scripted provider for tests, CI smoke, and offline dev."""

from __future__ import annotations

from collections import deque
from typing import Any

from selflearn.llm.base import LLMProvider
from selflearn.llm.types import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    LLMError,
    Stream,
    ToolCall,
    Usage,
)


class FakeLLM(LLMProvider):
    """Scripted responses popped in order. Records every request received."""

    name = "fake"

    def __init__(
        self,
        responses: list[ChatResponse | LLMError | str] | None = None,
        *,
        default_text: str = "fake reply",
        timeout_s: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(timeout_s=timeout_s, max_retries=max_retries)
        self._queue: deque[ChatResponse | LLMError | str] = deque(responses or [])
        self.default_text = default_text
        self.requests: list[ChatRequest] = []
        self.calls = 0

    def queue(self, item: ChatResponse | LLMError | str) -> None:
        """Append a scripted response, error, or plain text reply."""
        self._queue.append(item)

    def _next(self, req: ChatRequest) -> ChatResponse:
        self.requests.append(req)
        self.calls += 1
        if not self._queue:
            return ChatResponse(
                text=self.default_text,
                usage=Usage(prompt_tokens=10, completion_tokens=5),
                model="fake",
            )
        item = self._queue.popleft()
        if isinstance(item, LLMError):
            raise item
        if isinstance(item, str):
            return ChatResponse(
                text=item,
                usage=Usage(prompt_tokens=10, completion_tokens=len(item)),
                model="fake",
            )
        return item

    def chat(self, req: ChatRequest) -> ChatResponse:
        """Pop the next scripted response."""
        return self._next(req)

    def chat_stream(self, req: ChatRequest) -> Stream:
        """Yield the scripted text word by word, then a done chunk with usage."""

        def gen() -> Stream:
            resp = self._next(req)
            for word in resp.text.split(" "):
                yield ChatChunk(delta_text=word + " ")
            yield ChatChunk(delta_text="", done=True, usage=resp.usage)
            # ponytail: trailing tool calls surface after text, good enough for fake.
            for _call in resp.tool_calls:
                pass

        return gen()

    def tool_response(self, calls: list[ToolCall], text: str = "") -> ChatResponse:
        """Helper: build a tool-calling response."""
        return ChatResponse(
            text=text,
            tool_calls=tuple(calls),
            usage=Usage(prompt_tokens=12, completion_tokens=8),
            model="fake",
        )

    def last_request(self) -> ChatRequest | None:
        """Most recent request, or None if none yet."""
        return self.requests[-1] if self.requests else None

    def last_user_text(self) -> str:
        """Last user message text, or empty string."""
        req = self.last_request()
        if req is None:
            return ""
        for msg in reversed(req.messages):
            if msg.role == "user":
                return msg.content
        return ""

    def as_dict(self) -> dict[str, Any]:
        """Debug view (no secrets; fake has none)."""
        return {"provider": self.name, "queued": len(self._queue), "calls": self.calls}


__all__ = ["FakeLLM"]
