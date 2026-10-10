"""Anthropic provider: native Messages API.

System prompt field, content blocks, tool-use format, own headers and errors.
Anthropic has no embeddings API and no list-models endpoint (see docs).
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from selflearn.llm.base import (
    LLMProvider,
    decode_json,
    http_post_json,
    map_http_error,
    parse_retry_after,
    retry_delay,
    should_retry,
)
from selflearn.llm.types import (
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    LLMError,
    ParseError,
    Stream,
    ToolCall,
    ToolDef,
    Usage,
)

ANTHROPIC_VERSION = "2023-06-01"


def _blocks_to_text(blocks: list[Any]) -> tuple[str, tuple[ToolCall, ...]]:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        kind = b.get("type")
        if kind == "text":
            texts.append(str(b.get("text", "")))
        elif kind == "tool_use":
            raw = b.get("input", {})
            calls.append(
                ToolCall(
                    id=str(b.get("id", "")),
                    name=str(b.get("name", "")),
                    arguments=raw if isinstance(raw, dict) else {},
                )
            )
    return "".join(texts), tuple(calls)


def _to_wire_messages(
    messages: tuple[ChatMessage, ...],
) -> tuple[str, list[dict[str, Any]]]:
    """Split system prompt out; convert the rest to content blocks."""
    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
            continue
        if m.role == "tool":
            wire.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content,
                        }
                    ],
                }
            )
            continue
        blocks: list[dict[str, Any]] = []
        if m.content:
            blocks.append({"type": "text", "text": m.content})
        for c in m.tool_calls:
            blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
        wire.append({"role": m.role, "content": blocks or [{"type": "text", "text": ""}]})
    return "\n".join(system_parts), wire


def _to_wire_tools(tools: tuple[ToolDef, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters or {"type": "object", "properties": {}},
        }
        for t in tools
    ]


class AnthropicProvider(LLMProvider):
    """Native Anthropic Messages API. urllib only, no vendor SDK."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout_s: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(timeout_s=timeout_s, max_retries=max_retries)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _headers(self, *, stream: bool = False) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "User-Agent": "selflearn/0.1.0 (urllib)",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    def _payload(self, req: ChatRequest, *, stream: bool) -> dict[str, Any]:
        system, msgs = _to_wire_messages(req.messages)
        body: dict[str, Any] = {
            "model": req.model,
            "messages": msgs,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": stream,
        }
        if system:
            body["system"] = system
        if req.tools:
            body["tools"] = _to_wire_tools(req.tools)
        return body

    def chat(self, req: ChatRequest) -> ChatResponse:
        """POST /v1/messages with retries and 429/Retry-After handling."""
        return self._with_retries(lambda: self._chat_once(req))

    def _chat_once(self, req: ChatRequest) -> ChatResponse:
        url = f"{self.base_url}/v1/messages"
        status, headers, body = http_post_json(
            url,
            self._headers(),
            self._payload(req, stream=False),
            timeout_s=self.timeout_s,
            provider=self.name,
        )
        if status != 200:
            raise map_http_error(
                status,
                body,
                provider=self.name,
                retry_after_s=parse_retry_after(headers.get("retry-after")),
            )
        data = decode_json(body, provider=self.name)
        if data.get("type") == "error" or "content" not in data:
            raise ParseError(f"{self.name}: unexpected response shape", provider=self.name)
        text, calls = _blocks_to_text(data.get("content", []))
        usage = data.get("usage", {})
        return ChatResponse(
            text=text,
            tool_calls=calls,
            usage=Usage(
                prompt_tokens=int(usage.get("input_tokens", 0) or 0),
                completion_tokens=int(usage.get("output_tokens", 0) or 0),
            ),
            model=str(data.get("model", req.model)),
        )

    def chat_stream(self, req: ChatRequest) -> Stream:
        """SSE event stream (message_start/delta/content_block_delta/message_delta)."""
        url = f"{self.base_url}/v1/messages"
        data = json.dumps(self._payload(req, stream=True)).encode()
        request = urllib.request.Request(url, data=data, headers=self._headers(stream=True))

        def gen() -> Stream:
            attempt = 0
            while True:
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                        if int(resp.status) != 200:
                            raise map_http_error(int(resp.status), resp.read(), provider=self.name)
                        yield from _iter_anthropic_sse(resp)
                    return
                except LLMError as exc:
                    if not should_retry(exc, attempt, self.max_retries):
                        raise
                    time.sleep(retry_delay(exc, attempt))
                    attempt += 1

        return gen()

    def list_models(self) -> list[str]:
        """Anthropic exposes no list-models endpoint; returns []."""
        return []


def _iter_anthropic_sse(resp: Any) -> Stream:
    text_parts: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    event = ""
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if line.startswith("event:"):
            event = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        try:
            data: Any = json.loads(payload)
        except ValueError:
            continue
        kind = data.get("type", "")
        if kind == "message_start":
            usage = (data.get("message", {}) or {}).get("usage", {}) or {}
            prompt_tokens = int(usage.get("input_tokens", 0) or 0)
        elif kind == "content_block_delta":
            delta = data.get("delta", {}) or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                text_parts.append(delta["text"])
                yield ChatChunk(delta_text=delta["text"])
        elif kind == "message_delta":
            usage = data.get("usage", {}) or {}
            completion_tokens = int(usage.get("output_tokens", 0) or 0)
        elif kind == "message_stop" or event == "message_stop":
            break
    yield ChatChunk(
        delta_text="",
        done=True,
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


__all__ = ["ANTHROPIC_VERSION", "AnthropicProvider"]
