"""Structured JSON logging with request/trace ids. Secrets never logged."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

_request_id: ContextVar[str] = ContextVar("selflearn_request_id", default="")
_trace_id: ContextVar[str] = ContextVar("selflearn_trace_id", default="")

_SECRET_KEYS = ("api_key", "apikey", "token", "secret", "authorization", "bearer")


def new_request_id() -> str:
    """Create and bind a new request id for this context."""
    rid = uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def new_trace_id() -> str:
    """Create and bind a new trace id for this context."""
    tid = uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


def scrub(obj: Any) -> Any:
    """Redact secret-looking keys/values from a structure before logging."""
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in _SECRET_KEYS:
                out[k] = "(redacted)"
            elif isinstance(k, str) and "key" in k.lower() and isinstance(v, str):
                out[k] = "(redacted)"
            else:
                out[k] = scrub(v)
        return out
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, str):
        for prefix in ("sk-", "sk-ant-", "gsk_", "Bearer "):
            if prefix in obj:
                return "(redacted)"
        return obj
    return obj


class JsonFormatter(logging.Formatter):
    """Single-line JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": _request_id.get(),
            "trace_id": _trace_id.get(),
        }
        for attr in ("latency_ms", "tokens_in", "tokens_out", "provider", "model"):
            if hasattr(record, attr):
                payload[attr] = getattr(record, attr)
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(scrub(payload))


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a logger with the JSON formatter attached once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


__all__ = [
    "JsonFormatter",
    "get_logger",
    "new_request_id",
    "new_trace_id",
    "scrub",
]
