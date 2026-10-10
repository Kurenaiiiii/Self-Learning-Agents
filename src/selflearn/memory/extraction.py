"""Memory extraction: conversation turns -> atomic memories as validated JSON.

One LLM call per extraction. Malformed output gets one repair retry, then a
safe fallback (empty list). Extraction never crashes the chat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from selflearn.llm.base import LLMProvider
from selflearn.llm.types import ChatMessage, ChatRequest, LLMError
from selflearn.store.models import VALID_TYPES

EXTRACTION_SCHEMA_HINT = (
    '{"memories": [{"type": "fact|preference|episode|lesson", '
    '"text": "atomic statement", "importance": 0.0-1.0}]}'
)

EXTRACTION_PROMPT = (
    "Extract durable memories about the user from this conversation.\n"
    "Rules: one atomic statement per memory; skip chit-chat; "
    "skip anything already stated as factually wrong.\n"
    "Types: fact (stable truth), preference (likes/wants), "
    "episode (event), lesson (how to serve them).\n"
    "Reply with ONLY strict JSON matching this schema, "
    "no code fences, no prose:\n" + EXTRACTION_SCHEMA_HINT
)

REPAIR_PROMPT = (
    "Your last reply was not valid JSON. Reply again with ONLY strict JSON "
    "matching this schema, no code fences, no prose:\n" + EXTRACTION_SCHEMA_HINT
)


@dataclass(frozen=True)
class CandidateMemory:
    """One extracted memory awaiting consolidation."""

    type: str
    text: str
    importance: float
    source: str = "extraction"


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _validate_item(item: Any) -> CandidateMemory | None:
    if not isinstance(item, dict):
        return None
    mtype = item.get("type")
    text = item.get("text")
    if mtype not in VALID_TYPES or not isinstance(text, str) or not text.strip():
        return None
    try:
        importance = float(item.get("importance", 0.5))
    except (TypeError, ValueError):
        importance = 0.5
    return CandidateMemory(
        type=mtype,
        text=text.strip()[:2000],
        importance=max(0.0, min(1.0, importance)),
    )


def parse_extraction_output(raw: str) -> list[CandidateMemory]:
    """Parse + validate model output. Returns [] on any malformed input."""
    try:
        data: Any = json.loads(_strip_fences(raw))
    except ValueError:
        return []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("memories", [])
    else:
        return []
    if not isinstance(items, list):
        return []
    out: list[CandidateMemory] = []
    for item in items:
        cand = _validate_item(item)
        if cand is not None:
            out.append(cand)
    return out


def extract_memories(
    llm: LLMProvider,
    *,
    model: str,
    conversation: str,
    temperature: float = 0.0,
) -> list[CandidateMemory]:
    """Extract candidates from conversation text. Never raises on bad output."""
    base = (
        ChatMessage(role="system", content=EXTRACTION_PROMPT),
        ChatMessage(role="user", content=conversation[:12000]),
    )

    def ask(extra: tuple[ChatMessage, ...]) -> str:
        try:
            resp = llm.chat(
                ChatRequest(
                    messages=base + extra,
                    model=model,
                    temperature=temperature,
                    max_tokens=1024,
                )
            )
            return resp.text
        except LLMError:
            return ""

    cands = parse_extraction_output(ask(()))
    if cands:
        return cands
    # One repair retry, seeded with the broken output for context.
    repaired = parse_extraction_output(ask((ChatMessage(role="user", content=REPAIR_PROMPT),)))
    return repaired


__all__ = [
    "CandidateMemory",
    "extract_memories",
    "parse_extraction_output",
]
