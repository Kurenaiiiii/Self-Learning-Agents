"""Opt-in disk cache keyed by request hash. Used by benchmark for repro/cost control."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any


def request_hash(payload: dict[str, Any]) -> str:
    """Stable sha256 over canonical JSON of the request."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DiskCache:
    """Tiny file cache: one JSON file per key. No secrets ever written by callers."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = "".join(c for c in key if c.isalnum() or c in "-_")[:128]
        return os.path.join(self.directory, f"{safe}.json")

    def get(self, key: str) -> dict[str, Any] | None:
        """Return cached value or None on any miss/corruption (never raise)."""
        try:
            with open(self._path(key), encoding="utf-8") as f:
                data: Any = json.load(f)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store value. Caller must ensure no secrets inside."""
        try:
            with open(self._path(key), "w", encoding="utf-8") as f:
                json.dump(value, f)
        except OSError:
            pass  # cache is best-effort; never break the call


__all__ = ["DiskCache", "request_hash"]
