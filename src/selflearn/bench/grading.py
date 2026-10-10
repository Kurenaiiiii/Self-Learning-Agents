"""Deterministic grading: normalized keyword matching. No LLM involved."""

from __future__ import annotations

import re
from dataclasses import dataclass

_WS_RE = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(_WS_RE.sub(" ", text.lower()).split())


@dataclass(frozen=True)
class Grade:
    """Outcome of grading one answer."""

    correct: bool
    keywords_found: int
    keywords_total: int
    stale_hit: bool  # an outdated fact leaked into the answer


def grade_answer(answer: str, *, keywords: list[str], must_absent: list[str]) -> Grade:
    """All keywords present and no must_absent phrase present => correct."""
    norm = normalize(answer)
    found = sum(1 for kw in keywords if normalize(kw) in norm)
    stale = any(normalize(bad) in norm for bad in must_absent)
    return Grade(
        correct=bool(keywords) and found == len(keywords) and not stale,
        keywords_found=found,
        keywords_total=len(keywords),
        stale_hit=stale,
    )


__all__ = ["Grade", "grade_answer", "normalize"]
