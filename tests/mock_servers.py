"""Mock LLM servers, each faithful to its real wire format.

Used by the shared provider contract tests. Each mock validates request
shape (auth headers, required fields) and returns realistic errors.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class MockServer:
    """One background HTTP server with a scripted behavior queue."""

    def __init__(self, handler_cls: type[BaseHTTPRequestHandler]) -> None:
        self.requests: list[dict[str, Any]] = []
        self.behaviors: list[str] = []  # popped per request; default "ok"
        self.received = 0
        handler_cls.mock = self  # type: ignore[attr-defined]
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        """Server root URL."""
        return f"http://127.0.0.1:{self.port}"

    def next_behavior(self) -> str:
        """Pop next scripted behavior, defaulting to ok."""
        self.received += 1
        if self.behaviors:
            return self.behaviors.pop(0)
        return "ok"

    def stop(self) -> None:
        """Shut the server down."""
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def _read_json(handler: BaseHTTPRequestHandler) -> tuple[dict[str, str], dict[str, Any]]:
    length = int(handler.headers.get("Content-Length", 0) or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        body: Any = json.loads(raw.decode("utf-8") or "{}")
    except ValueError:
        body = {}
    return {k.lower(): v for k, v in dict(handler.headers).items()}, (
        body if isinstance(body, dict) else {}
    )


def _send(handler: BaseHTTPRequestHandler, status: int, obj: dict[str, Any]) -> None:
    data = json.dumps(obj).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _send_error(
    handler: BaseHTTPRequestHandler,
    status: int,
    obj: dict[str, Any],
    *,
    retry_after: str = "",
) -> None:
    data = json.dumps(obj).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    if retry_after:
        handler.send_header("Retry-After", retry_after)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


# -- OpenAI-compatible mock ---------------------------------------------------

OPENAI_OK = {
    "id": "chatcmpl-mock",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "mock-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello from mock"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
}

OPENAI_TOOL = {
    "id": "chatcmpl-mock",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "mock-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "memory_search",
                            "arguments": '{"query": "coffee"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29},
}


class OpenAIMockHandler(BaseHTTPRequestHandler):
    """Mock Chat Completions endpoint. Validates OpenAI wire shape."""

    mock: MockServer

    def log_message(self, *args: object) -> None:
        """Silence test output."""

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            _send(self, 200, {"object": "list", "data": [{"id": "mock-model", "object": "model"}]})
        else:
            _send_error(
                self, 404, {"error": {"message": "not found", "type": "invalid_request_error"}}
            )

    def do_POST(self) -> None:
        headers, body = _read_json(self)
        self.mock.requests.append({"path": self.path, "headers": headers, "body": body})
        behavior = self.mock.next_behavior()
        if not self.path.endswith("chat/completions"):
            _send_error(
                self, 404, {"error": {"message": "not found", "type": "invalid_request_error"}}
            )
            return
        if "authorization" not in headers:
            _send_error(self, 401, {"error": {"message": "missing auth", "type": "auth_error"}})
            return
        if not isinstance(body.get("messages"), list) or not body.get("model"):
            _send_error(
                self,
                400,
                {"error": {"message": "bad messages/model", "type": "invalid_request_error"}},
            )
            return
        if behavior == "429":
            _send_error(
                self,
                429,
                {"error": {"message": "rate limited", "type": "rate_limit_error"}},
                retry_after="0",
            )
            return
        if behavior == "500":
            _send_error(self, 500, {"error": {"message": "upstream boom"}})
            return
        if behavior == "401":
            _send_error(self, 401, {"error": {"message": "bad key"}})
            return
        if behavior == "400":
            _send_error(self, 400, {"error": {"message": "bad request"}})
            return
        if behavior == "sleep":
            time.sleep(2.0)
            _send(self, 200, OPENAI_OK)
            return
        if behavior == "badjson":
            raw = b"{not json"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if body.get("stream"):
            self._stream_ok()
            return
        if behavior == "tool" or body.get("tools"):
            _send(self, 200, OPENAI_TOOL)
            return
        _send(self, 200, OPENAI_OK)

    def _stream_ok(self) -> None:
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "hel"}}]},
            {"choices": [{"index": 0, "delta": {"content": "lo"}}]},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for c in chunks:
            line = f"data: {json.dumps(c)}\n\n".encode()
            self.wfile.write(line)
        usage = 'data: {"usage": {"prompt_tokens": 11, "completion_tokens": 7}}\n\n'
        self.wfile.write(usage.encode())
        self.wfile.write(b"data: [DONE]\n\n")


# -- Anthropic mock ------------------------------------------------------------

ANTHROPIC_OK = {
    "id": "msg_mock",
    "type": "message",
    "role": "assistant",
    "model": "mock-model",
    "content": [{"type": "text", "text": "hello from anthropic mock"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 13, "output_tokens": 6},
}

ANTHROPIC_TOOL = {
    "id": "msg_mock",
    "type": "message",
    "role": "assistant",
    "model": "mock-model",
    "content": [
        {"type": "text", "text": ""},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "memory_search",
            "input": {"query": "coffee"},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 21, "output_tokens": 9},
}


class AnthropicMockHandler(BaseHTTPRequestHandler):
    """Mock Messages endpoint. Validates Anthropic wire shape."""

    mock: MockServer

    def log_message(self, *args: object) -> None:
        """Silence test output."""

    def do_POST(self) -> None:
        headers, body = _read_json(self)
        self.mock.requests.append({"path": self.path, "headers": headers, "body": body})
        behavior = self.mock.next_behavior()
        if self.path != "/v1/messages":
            _send_error(
                self,
                404,
                {"type": "error", "error": {"type": "not_found_error", "message": "nope"}},
            )
            return
        if "x-api-key" not in headers or "anthropic-version" not in headers:
            _send_error(
                self,
                401,
                {"type": "error", "error": {"type": "authentication_error", "message": "bad key"}},
            )
            return
        if not isinstance(body.get("messages"), list) or not body.get("model"):
            _send_error(
                self,
                400,
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "bad messages/model"},
                },
            )
            return
        if isinstance(body.get("system"), list):
            _send_error(
                self,
                400,
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "system must be string"},
                },
            )
            return
        if behavior == "429":
            _send_error(
                self,
                429,
                {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
                retry_after="0",
            )
            return
        if behavior == "529":
            _send_error(
                self,
                529,
                {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
            )
            return
        if behavior == "401":
            _send_error(
                self,
                401,
                {"type": "error", "error": {"type": "authentication_error", "message": "bad key"}},
            )
            return
        if behavior == "400":
            _send_error(
                self,
                400,
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "bad request"},
                },
            )
            return
        if behavior == "sleep":
            time.sleep(2.0)
            _send(self, 200, ANTHROPIC_OK)
            return
        if body.get("stream"):
            self._stream_ok()
            return
        if behavior == "tool" or body.get("tools"):
            _send(self, 200, ANTHROPIC_TOOL)
            return
        _send(self, 200, ANTHROPIC_OK)

    def _stream_ok(self) -> None:
        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 13, "output_tokens": 0}},
                },
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hel"}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "lo"}},
            ),
            ("message_delta", {"type": "message_delta", "usage": {"output_tokens": 6}}),
            ("message_stop", {"type": "message_stop"}),
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for name, obj in events:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(obj)}\n\n".encode())


# -- Ollama mock -----------------------------------------------------------------

OLLAMA_OK = {
    "model": "mock-model",
    "message": {"role": "assistant", "content": "hello from ollama mock"},
    "done": True,
    "prompt_eval_count": 9,
    "eval_count": 8,
}


class OllamaMockHandler(BaseHTTPRequestHandler):
    """Mock native Ollama API. Validates /api/chat shape."""

    mock: MockServer

    def log_message(self, *args: object) -> None:
        """Silence test output."""

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            _send(self, 200, {"models": [{"name": "mock-model:latest"}]})
        else:
            _send(self, 404, {"error": "not found"})

    def do_POST(self) -> None:
        headers, body = _read_json(self)
        self.mock.requests.append({"path": self.path, "headers": headers, "body": body})
        behavior = self.mock.next_behavior()
        if self.path != "/api/chat":
            _send(self, 404, {"error": "not found"})
            return
        if not isinstance(body.get("messages"), list) or not body.get("model"):
            _send(self, 400, {"error": "bad messages/model"})
            return
        if behavior == "429":
            _send_error(self, 429, {"error": "rate limited"}, retry_after="0")
            return
        if behavior == "401":
            _send(self, 401, {"error": "unauthorized"})
            return
        if behavior == "500":
            _send(self, 500, {"error": "boom"})
            return
        if behavior == "400":
            _send(self, 400, {"error": "bad request"})
            return
        if behavior == "sleep":
            time.sleep(2.0)
            _send(self, 200, OLLAMA_OK)
            return
        if body.get("stream"):
            self._stream_ok()
            return
        _send(self, 200, OLLAMA_OK)

    def _stream_ok(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for piece in ("hel", "lo"):
            line = json.dumps({"message": {"content": piece}, "done": False}) + "\n"
            self.wfile.write(line.encode())
        final = (
            json.dumps(
                {"message": {"content": ""}, "done": True, "prompt_eval_count": 9, "eval_count": 8}
            )
            + "\n"
        )
        self.wfile.write(final.encode())


__all__ = [
    "AnthropicMockHandler",
    "MockServer",
    "OllamaMockHandler",
    "OpenAIMockHandler",
]
