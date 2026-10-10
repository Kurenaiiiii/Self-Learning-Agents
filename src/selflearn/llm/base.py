"""Provider interface, retry policy, and stdlib HTTP helpers."""

from __future__ import annotations

import json
import random
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from selflearn.llm.types import (
    BadRequestError,
    ChatRequest,
    ChatResponse,
    LLMError,
    NetworkError,
    RateLimitError,
    ServerError,
    Stream,
    TimeoutError,
)

SleepFn = Callable[[float], None]


def backoff_delay(attempt: int, base_s: float = 0.5, cap_s: float = 30.0) -> float:
    """Exponential backoff with jitter: min(cap, base*2^attempt) + uniform(0, 0.25*2^attempt)."""
    exp = base_s * (2**attempt)
    jitter: float = random.uniform(0, 0.25 * (2**attempt))
    return float(min(cap_s, exp + jitter))


def parse_retry_after(value: str | None) -> float:
    """Parse a Retry-After header (delta-seconds). 0.0 when absent/invalid."""
    if not value:
        return 0.0
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        return 0.0


def should_retry(exc: BaseException, attempt: int, max_retries: int) -> bool:
    """Retry only retryable errors while attempts remain."""
    return attempt < max_retries and isinstance(exc, LLMError) and exc.retryable


def retry_delay(exc: BaseException, attempt: int) -> float:
    """Honor Retry-After on 429, else backoff with jitter."""
    if isinstance(exc, RateLimitError) and exc.retry_after_s > 0:
        extra: float = random.uniform(0, 0.5)
        return exc.retry_after_s + extra
    return backoff_delay(attempt)


class LLMProvider(ABC):
    """One interface every provider implements."""

    name: str = "base"

    def __init__(self, *, timeout_s: float = 60.0, max_retries: int = 3) -> None:
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    @abstractmethod
    def chat(self, req: ChatRequest) -> ChatResponse:
        """Single non-streaming chat call (with retries)."""
        ...

    @abstractmethod
    def chat_stream(self, req: ChatRequest) -> Stream:
        """Streaming chat call (retries apply to connection setup)."""
        ...

    def list_models(self) -> list[str]:
        """Models this provider exposes. Empty when the API has no such endpoint."""
        return []

    # -- shared retry driver -------------------------------------------------
    def _with_retries(
        self,
        fn: Callable[[], ChatResponse],
        *,
        sleep: SleepFn = time.sleep,
    ) -> ChatResponse:
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - normalized below
                if not should_retry(exc, attempt, self.max_retries):
                    if isinstance(exc, LLMError):
                        raise
                    raise NetworkError(f"{self.name}: {exc}", provider=self.name) from exc
                sleep(retry_delay(exc, attempt))
                attempt += 1


def http_post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout_s: float,
    provider: str,
) -> tuple[int, dict[str, str], bytes]:
    """POST JSON with urllib. Returns (status, headers, body). Never raises HTTPError."""
    data = json.dumps(payload).encode("utf-8")
    headers = {"User-Agent": "selflearn/0.1.0 (urllib)", **headers}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = int(resp.status)
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            return status, resp_headers, resp.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
        except Exception:  # noqa: BLE001 - best effort body
            body = b""
        heads = {k.lower(): v for k, v in (e.headers.items() if e.headers else [])}
        return int(e.code), heads, body
    except socket.timeout as e:  # noqa: UP041 - bare TimeoutError is our LLMError here
        raise TimeoutError(f"{provider}: timed out", provider=provider) from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, socket.timeout):
            raise TimeoutError(f"{provider}: timed out", provider=provider) from e
        raise NetworkError(f"{provider}: {e.reason}", provider=provider) from e


def http_get_json(
    url: str,
    headers: dict[str, str],
    *,
    timeout_s: float,
    provider: str,
) -> tuple[int, bytes]:
    """GET JSON with urllib. Returns (status, body)."""
    headers = {"User-Agent": "selflearn/0.1.0 (urllib)", **headers}
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
        except Exception:  # noqa: BLE001 - best effort body
            body = b""
        return int(e.code), body
    except socket.timeout as e:  # noqa: UP041 - bare TimeoutError is our LLMError here
        raise TimeoutError(f"{provider}: timed out", provider=provider) from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, socket.timeout):
            raise TimeoutError(f"{provider}: timed out", provider=provider) from e
        raise NetworkError(f"{provider}: {e.reason}", provider=provider) from e


def decode_json(body: bytes, *, provider: str) -> dict[str, Any]:
    """Parse a JSON object body or raise a mapped error."""
    from selflearn.llm.types import ParseError  # local import: avoid cycle

    try:
        parsed: Any = json.loads(body.decode("utf-8") if body else "{}")
    except (ValueError, UnicodeDecodeError) as e:
        raise ParseError(f"{provider}: invalid JSON response", provider=provider) from e
    if not isinstance(parsed, dict):
        raise ParseError(f"{provider}: expected JSON object", provider=provider)
    return parsed


def map_http_error(
    status: int, body: bytes, *, provider: str, retry_after_s: float = 0.0
) -> LLMError:
    """Map an HTTP failure to a typed error. Message scrubbed of secrets upstream."""
    try:
        detail = body.decode("utf-8", errors="replace")[:500]
    except Exception:  # noqa: BLE001 - never fail error mapping
        detail = ""
    msg = f"{provider}: HTTP {status}" + (f" {detail}" if detail else "")
    if status in (401, 403):
        from selflearn.llm.types import AuthError  # local import: avoid cycle

        return AuthError(msg, provider=provider)
    if status == 429:
        return RateLimitError(msg, provider=provider, retry_after_s=retry_after_s)
    if status in (400, 404, 422):
        return BadRequestError(msg, provider=provider)
    if status >= 500:
        return ServerError(msg, provider=provider)
    return BadRequestError(msg, provider=provider)


__all__ = [
    "LLMProvider",
    "backoff_delay",
    "decode_json",
    "http_get_json",
    "http_post_json",
    "map_http_error",
    "parse_retry_after",
    "retry_delay",
    "should_retry",
]
