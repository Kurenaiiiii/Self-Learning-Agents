"""Privacy: secret redaction at write time, forget-me, export, retention.

What goes to the LLM provider (documented in docs, enforced here at storage):
chat messages, retrieval queries, extraction/reflection transcripts. Secrets
are redacted before any memory text is STORED; live conversation content is
necessarily sent to the provider to get a reply.
"""

from __future__ import annotations

import json
import re

from selflearn.memory.extraction import CandidateMemory
from selflearn.memory.learning import apply_forgetting
from selflearn.store.sqlite import MemoryStore

_PATTERNS = [
    r"sk-ant-[A-Za-z0-9\-_]{8,}",
    r"sk-[A-Za-z0-9\-_]{8,}",
    r"gsk_[A-Za-z0-9\-_]{8,}",
    r"xox[baprs]-[A-Za-z0-9\-_]{8,}",
    r"AKIA[0-9A-Z]{16}",
    r"Bearer\s+[A-Za-z0-9\-._~+/=]{8,}",
    r"(?i)(api[_-]?key|api[_-]?token|secret|password)\s*[:=]\s*\S+",
]
_REDACTED = "[REDACTED]"
_COMPILED = [re.compile(p) for p in _PATTERNS]


def redact_text(text: str) -> str:
    """Replace obvious secret/key patterns. Idempotent, never raises."""
    try:
        out = text
        for rx in _COMPILED:
            out = rx.sub(_REDACTED, out)
        return out
    except re.error:
        return text


def redact_candidate(cand: CandidateMemory) -> CandidateMemory:
    """Redacted copy of a candidate (no-op when nothing matches)."""
    clean = redact_text(cand.text)
    if clean == cand.text:
        return cand
    return CandidateMemory(
        type=cand.type, text=clean, importance=cand.importance, source=cand.source
    )


def forget_me(store: MemoryStore, *, user: str) -> int:
    """Hard-delete ALL of a user's data (memories + history). Returns rows removed."""
    return store.delete_user(user=user)


def export_user_json(store: MemoryStore, *, user: str) -> str:
    """Full per-user export as JSON (all namespaces, incl. deleted)."""
    return json.dumps(store.export_user(user=user), indent=2, default=str)


def enforce_retention(
    store: MemoryStore,
    *,
    user: str,
    namespace: str = "default",
    max_age_days: float = 365.0,
    min_importance: float = 0.2,
) -> dict[str, int]:
    """Configurable retention: decay long-unaccessed, prune old + unimportant."""
    return apply_forgetting(
        store,
        user=user,
        namespace=namespace,
        max_age_days=max_age_days,
        min_importance=min_importance,
    )


__all__ = [
    "enforce_retention",
    "export_user_json",
    "forget_me",
    "redact_candidate",
    "redact_text",
]
