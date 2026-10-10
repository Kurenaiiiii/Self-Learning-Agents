"""OpenAI-compatible provider: Chat Completions wire format + configurable base_url.

Works with OpenAI and any compatible service or server. No vendor-specific code.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from selflearn.llm.base import (
    LLMProvider,
    decode_json,
    http_get_json,
    http_post_json,
    map_http_error,
    parse_retry_after,
    retry_delay,
    should_retry,
)
from selflearn.llm.cache import DiskCache, request_hash
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


def _to_wire_message(msg: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in msg.tool_calls
        ]
    if msg.role == "tool":
        out["tool_call_id"] = msg.tool_call_id
        if msg.name:
            out["name"] = msg.name
    return out


def _to_wire_tools(tools: tuple[ToolDef, ...]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def _from_wire_choice(choice: dict[str, Any]) -> tuple[str, tuple[ToolCall, ...]]:
    msg = choice.get("message", {})
    text = msg.get("content") or ""
    calls: list[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else {}
        except ValueError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))
    return str(text), tuple(calls)


def _usage_from_wire(obj: dict[str, Any]) -> Usage:
    u = obj.get("usage") or {}
    return Usage(
        prompt_tokens=int(u.get("prompt_tokens", 0) or 0),
        completion_tokens=int(u.get("completion_tokens", 0) or 0),
    )


class OpenAICompatibleProvider(LLMProvider):
    """Chat Completions over any base_url. urllib only, no vendor SDKs."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        timeout_s: float = 60.0,
        max_retries: int = 3,
        cache: DiskCache | None = None,
    ) -> None:
        super().__init__(timeout_s=timeout_s, max_retries=max_retries)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.cache = cache

    def _headers(self, *, stream: bool = False) -> dict[str, str]:
        h = {"Content-Type": "application/json", "User-Agent": "selflearn/0.1.0 (urllib)"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if stream:
            h["Accept"] = "text/event-stream"
        return h

    def _payload(self, req: ChatRequest, *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": req.model,
            "messages": [_to_wire_message(m) for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": stream,
        }
        if req.tools:
            body["tools"] = _to_wire_tools(req.tools)
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def _cache_key(self, req: ChatRequest) -> str:
        return request_hash(
            {"provider": self.name, "base_url": self.base_url, **self._payload(req, stream=False)}
        )

    def chat(self, req: ChatRequest) -> ChatResponse:
        """POST chat/completions with retries, Retry-After, and optional cache."""
        if self.cache is not None:
            key = self._cache_key(req)
            hit = self.cache.get(key)
            if hit is not None:
                return _response_from_cache(hit, req.model)
        return self._with_retries(lambda: self._chat_once(req))

    def _chat_once(self, req: ChatRequest) -> ChatResponse:
        url = f"{self.base_url}/chat/completions"
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
        choices = data.get("choices")
        if not choices:
            raise ParseError(f"{self.name}: missing choices", provider=self.name)
        text, calls = _from_wire_choice(choices[0])
        resp = ChatResponse(
            text=text,
            tool_calls=calls,
            usage=_usage_from_wire(data),
            model=str(data.get("model", req.model)),
        )
        if self.cache is not None:
            self.cache.set(self._cache_key(req), _response_to_cache(resp))
        return resp

    def chat_stream(self, req: ChatRequest) -> Stream:
        """SSE stream; retries cover connection setup only."""
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(self._payload(req, stream=True)).encode()
        request = urllib.request.Request(url, data=data, headers=self._headers(stream=True))

        def gen() -> Stream:
            attempt = 0
            while True:
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                        if int(resp.status) != 200:
                            raise map_http_error(int(resp.status), resp.read(), provider=self.name)
                        yield from _iter_sse(resp)
                    return
                except LLMError as exc:
                    if not should_retry(exc, attempt, self.max_retries):
                        raise
                    time.sleep(retry_delay(exc, attempt))
                    attempt += 1

        return gen()

    def list_models(self) -> list[str]:
        """GET {base}/models. Returns [] when unsupported."""
        status, body = http_get_json(
            f"{self.base_url}/models",
            self._headers(),
            timeout_s=self.timeout_s,
            provider=self.name,
        )
        if status != 200:
            return []
        try:
            data = decode_json(body, provider=self.name)
            items = data.get("data", [])
            return [str(i.get("id", "")) for i in items if isinstance(i, dict) and i.get("id")]
        except LLMError:
            return []


def _iter_sse(resp: Any) -> Stream:
    text_parts: list[str] = []
    usage = Usage()
    while True:
        line = resp.readline()
        if not line:
            break
        s = line.decode("utf-8", errors="replace").strip()
        if not s.startswith("data:"):
            continue
        payload = s[5:].strip()
        if payload == "[DONE]":
            break
        try:
            evt: Any = json.loads(payload)
        except ValueError:
            continue
        for choice in evt.get("choices", []):
            delta = choice.get("delta", {})
            piece = delta.get("content") or ""
            if piece:
                text_parts.append(piece)
                yield ChatChunk(delta_text=piece)
        if evt.get("usage"):
            u = evt["usage"]
            usage = Usage(
                prompt_tokens=int(u.get("prompt_tokens", 0) or 0),
                completion_tokens=int(u.get("completion_tokens", 0) or 0),
            )
    yield ChatChunk(delta_text="", done=True, usage=usage)


def _response_to_cache(resp: ChatResponse) -> dict[str, Any]:
    return {
        "text": resp.text,
        "model": resp.model,
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "tool_calls": [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in resp.tool_calls
        ],
    }


def _response_from_cache(hit: dict[str, Any], model: str) -> ChatResponse:
    calls = tuple(
        ToolCall(id=c.get("id", ""), name=c.get("name", ""), arguments=c.get("arguments", {}))
        for c in hit.get("tool_calls", [])
        if isinstance(c, dict)
    )
    return ChatResponse(
        text=str(hit.get("text", "")),
        tool_calls=calls,
        usage=Usage(
            prompt_tokens=int(hit.get("prompt_tokens", 0) or 0),
            completion_tokens=int(hit.get("completion_tokens", 0) or 0),
        ),
        model=str(hit.get("model", model)),
    )


__all__ = ["OpenAICompatibleProvider"]
