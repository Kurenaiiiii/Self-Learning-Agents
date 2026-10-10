"""Ollama provider: local server via its native /api/chat (NDJSON) + /api/tags.

If base_url ends with /v1, the OpenAI-compatible endpoint is used instead,
via the shared OpenAI-compatible implementation.
"""

from __future__ import annotations

import json
from typing import Any

from selflearn.llm.base import (
    LLMProvider,
    decode_json,
    http_get_json,
    http_post_json,
    map_http_error,
    parse_retry_after,
)
from selflearn.llm.openai_compatible import OpenAICompatibleProvider
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


def _to_ollama_message(msg: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in msg.tool_calls
        ]
    return out


def _to_ollama_tools(tools: tuple[ToolDef, ...]) -> list[dict[str, Any]]:
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


class OllamaProvider(LLMProvider):
    """Native Ollama chat API. No key needed for a local server."""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(timeout_s=timeout_s, max_retries=max_retries)
        self.base_url = base_url.rstrip("/")
        self._compat: OpenAICompatibleProvider | None = None
        if self.base_url.endswith("/v1"):
            # ponytail: reuse compat stack for Ollama's OpenAI endpoint.
            self._compat = OpenAICompatibleProvider(base_url=self.base_url)

    def _chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def chat(self, req: ChatRequest) -> ChatResponse:
        """POST /api/chat (non-stream) with retries. Or compat endpoint for /v1."""
        if self._compat is not None:
            return self._compat.chat(req)
        return self._with_retries(lambda: self._chat_once(req))

    def _chat_once(self, req: ChatRequest) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": [_to_ollama_message(m) for m in req.messages],
            "stream": False,
            "options": {"temperature": req.temperature, "num_predict": req.max_tokens},
        }
        if req.tools:
            payload["tools"] = _to_ollama_tools(req.tools)
        status, headers, body = http_post_json(
            self._chat_url(),
            {"Content-Type": "application/json"},
            payload,
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
        msg = data.get("message")
        if not isinstance(msg, dict):
            raise ParseError(f"{self.name}: missing message", provider=self.name)
        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            calls.append(
                ToolCall(
                    id=str(tc.get("id", "")),
                    name=str(fn.get("name", "")),
                    arguments=args if isinstance(args, dict) else {},
                )
            )
        return ChatResponse(
            text=str(msg.get("content", "") or ""),
            tool_calls=tuple(calls),
            usage=Usage(
                prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
                completion_tokens=int(data.get("eval_count", 0) or 0),
            ),
            model=str(data.get("model", req.model)),
        )

    def chat_stream(self, req: ChatRequest) -> Stream:
        """NDJSON stream of {"message": {"content": ...}, "done": bool}."""
        if self._compat is not None:
            return self._compat.chat_stream(req)
        import urllib.request  # local import: only path needing it

        payload: dict[str, Any] = {
            "model": req.model,
            "messages": [_to_ollama_message(m) for m in req.messages],
            "stream": True,
            "options": {"temperature": req.temperature, "num_predict": req.max_tokens},
        }
        if req.tools:
            payload["tools"] = _to_ollama_tools(req.tools)
        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            self._chat_url(),
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "selflearn/0.1.0 (urllib)",
            },
        )

        def gen() -> Stream:
            import time as _time  # local import: keep module import-light

            from selflearn.llm.base import retry_delay, should_retry

            attempt = 0
            while True:
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                        if int(resp.status) != 200:
                            raise map_http_error(int(resp.status), resp.read(), provider=self.name)
                        yield from _iter_ndjson(resp)
                    return
                except LLMError as exc:
                    if not should_retry(exc, attempt, self.max_retries):
                        raise
                    _time.sleep(retry_delay(exc, attempt))
                    attempt += 1

        return gen()

    def list_models(self) -> list[str]:
        """GET /api/tags. Returns [] when the server is unreachable."""
        base = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        try:
            status, body = http_get_json(
                f"{base}/api/tags", {}, timeout_s=self.timeout_s, provider=self.name
            )
        except LLMError:
            return []
        if status != 200:
            return []
        try:
            data = decode_json(body, provider=self.name)
            models = data.get("models", [])
            return [str(m.get("name", "")) for m in models if isinstance(m, dict) and m.get("name")]
        except LLMError:
            return []


def _iter_ndjson(resp: Any) -> Stream:
    usage = Usage()
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            evt: Any = json.loads(line)
        except ValueError:
            continue
        msg = evt.get("message", {}) or {}
        piece = msg.get("content") or ""
        if piece:
            yield ChatChunk(delta_text=str(piece))
        if evt.get("done"):
            usage = Usage(
                prompt_tokens=int(evt.get("prompt_eval_count", 0) or 0),
                completion_tokens=int(evt.get("eval_count", 0) or 0),
            )
            break
    yield ChatChunk(delta_text="", done=True, usage=usage)


__all__ = ["OllamaProvider"]
