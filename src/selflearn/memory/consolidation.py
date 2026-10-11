"""Consolidation: ADD | UPDATE | SUPERSEDE | NOOP per candidate memory.

Deterministic guards first (exact + near duplicates), then one validated LLM
decision per candidate against similar existing memories. Superseded memories
stay in history but are excluded from retrieval by default.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from selflearn.embeddings.hybrid import Bm25Index, NgramTfidfIndex, gram_cosine
from selflearn.llm.base import LLMProvider
from selflearn.llm.types import ChatMessage, ChatRequest, LLMError
from selflearn.memory.extraction import CandidateMemory
from selflearn.store.models import Memory

Decision = Literal["ADD", "UPDATE", "SUPERSEDE", "NOOP"]

CONSOLIDATION_PROMPT = (
    "You maintain a user's long-term memory. "
    "For the CANDIDATE memory and EXISTING memories, pick one action.\n"
    "- NOOP: candidate adds nothing (duplicate, trivial, chit-chat).\n"
    "- UPDATE: same fact with more detail; give target_id to update.\n"
    "- SUPERSEDE: candidate contradicts an existing memory; give target_id. "
    "Old stays in history. Contradiction = same subject and relation with a "
    "different value (other city, other drink, other job). Correction cues "
    "like 'actually', 'not X', 'correction', 'I meant', 'moved from', "
    "'switched to', 'no longer' almost always mean SUPERSEDE the memory "
    "they argue with.\n"
    "- ADD: genuinely new information.\n"
    "Examples: existing 'lives in Berlin' + candidate 'lives in Lisbon' -> "
    "SUPERSEDE. Existing 'drinks cow milk' + candidate 'actually drinks oat "
    "milk, not cow milk' -> SUPERSEDE. Existing 'uses VS Code' + candidate "
    "'uses VS Code with vim plugin' -> UPDATE.\n"
    'Reply with ONLY strict JSON: {"action": "ADD|UPDATE|SUPERSEDE|NOOP", '
    '"target_id": "<id or empty>", "reason": "<short>"}'
)

# Correction cues: user explicitly revises a previous statement.
CORRECTION_CUES = (
    r"actually|correction|update\b|changed|instead|switched to|moved from|"
    r"not\b|never\b|no longer|was wrong|my mistake|i meant|"
    r"scratch that|rather\b|now\b|oops\b|dropped|quit\b|stopped"
)
_CUE_RE = re.compile(rf"\b(?:{CORRECTION_CUES})", re.IGNORECASE)

# Strong cues name the replacement outright; one shared anchor suffices.
_STRONG_CUES = ("instead", "switched to", "moved from", "correction", "scratch that")
_STRONG_RE = re.compile(
    r"\b(?:instead|switched to|moved from|correction|scratch that)", re.IGNORECASE
)

# Words whose object the candidate rejects or abandons ("not Berlin",
# "instead of cow milk", "tea over coffee", "dropped Spanish").
_NEGATION_RES = (
    re.compile(r"\bnot\s+(?:a|an|the)?\s*([a-z0-9]+)", re.IGNORECASE),
    re.compile(r"\bno\s+longer\s+(?:a|an|the)?\s*([a-z0-9]+)", re.IGNORECASE),
    re.compile(r"\binstead\s+of\s+(?:a|an|the)?\s*([a-z0-9]+)", re.IGNORECASE),
    re.compile(r"\bover\s+(?:a|an|the)?\s*([a-z0-9]+)", re.IGNORECASE),
    re.compile(
        r"\b(?:dropped|quit|stopped|abandoned)\s+(?:a|an|the)?\s*([a-z0-9]+)", re.IGNORECASE
    ),
)

_STOPWORDS = frozenset(
    "the a an i you he she it we they user my your his her its our their "
    "is are was were be been being do does did and or but in on at to of "
    "for with from by as that this these those it s t m d ll re ve".split()
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


def is_correction(text: str) -> bool:
    """Whether the text carries an explicit correction cue."""
    return _CUE_RE.search(text) is not None


def contradicts(candidate_text: str, existing_text: str) -> bool:
    """Same subject+relation with a different value, signaled by a cue.

    Requires a correction cue plus a shared anchor (2 content tokens, or 1
    for strong cues that name the replacement), plus evidence of a changed
    value: distinctive tokens on both sides, or the candidate explicitly
    negating/abandoning a token from the existing memory ("not Berlin",
    "instead of cow milk", "dropped Spanish"). Conservative: no cue, no fire.
    """
    if not is_correction(candidate_text):
        return False
    cand, exist = _content_tokens(candidate_text), _content_tokens(existing_text)
    anchor = cand & exist
    need = 1 if _STRONG_RE.search(candidate_text) else 2
    if len(anchor) < need:
        return False
    only_cand = {t for t in cand - exist if len(t) > 2}
    only_exist = {t for t in exist - cand if len(t) > 2}
    if only_cand and only_exist:
        return True
    # Candidate absorbs the old value ("X, not Y"): check it rejects a token
    # that the existing memory asserts.
    for rx in _NEGATION_RES:
        for match in rx.finditer(candidate_text):
            if match.group(1).lower() in exist:
                return True
    return False


def find_contradiction(candidate: CandidateMemory, existing: list[Memory]) -> Memory | None:
    """First similar existing memory the candidate explicitly corrects."""
    if not is_correction(candidate.text):
        return None
    for mem in find_similar(candidate, existing, top_k=5):
        if contradicts(candidate.text, mem.text):
            return mem
    return None


@dataclass(frozen=True)
class ConsolidationResult:
    """Outcome for one candidate."""

    action: Decision
    target_id: str
    reason: str


def find_similar(
    candidate: CandidateMemory,
    existing: list[Memory],
    *,
    top_k: int = 5,
) -> list[Memory]:
    """Similar existing memories by max(BM25-rank, vector-rank) fusion."""
    if not existing:
        return []
    bm25 = Bm25Index()
    vec = NgramTfidfIndex()
    for mem in existing:
        bm25.add(mem.id, mem.text)
        vec.add(mem.id, mem.text)
    order: dict[str, int] = {}
    for rank_list in (
        bm25.rank(candidate.text, top_k=top_k),
        vec.rank(candidate.text, top_k=top_k),
    ):
        for pos, (doc_id, _score) in enumerate(rank_list):
            order[doc_id] = min(order.get(doc_id, pos), pos)
    by_id = {m.id: m for m in existing}
    return [by_id[d] for d in sorted(order, key=order.__getitem__) if d in by_id][:top_k]


def exact_duplicate(candidate: CandidateMemory, existing: list[Memory]) -> Memory | None:
    """Case-insensitive exact text match."""
    want = " ".join(candidate.text.lower().split())
    for mem in existing:
        if " ".join(mem.text.lower().split()) == want:
            return mem
    return None


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]", "", text.lower())
    return " ".join(cleaned.split())


def near_duplicate(
    candidate: CandidateMemory, existing: list[Memory], *, threshold: float = 0.85
) -> Memory | None:
    """Normalized pairwise char n-gram TF cosine above threshold => duplicate."""
    want = _normalize(candidate.text)
    for mem in existing:
        if gram_cosine(want, _normalize(mem.text)) >= threshold:
            return mem
    return None


def _parse_decision(raw: str, similar: list[Memory]) -> ConsolidationResult | None:
    try:
        data = json.loads(raw.strip())
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("action") not in (
        "ADD",
        "UPDATE",
        "SUPERSEDE",
        "NOOP",
    ):
        return None
    action: Decision = data["action"]
    target = str(data.get("target_id", "") or "")
    reason = str(data.get("reason", "") or "")[:300]
    if action in ("UPDATE", "SUPERSEDE"):
        if not any(m.id == target for m in similar):
            return None  # unknown target: reject, caller falls back
        return ConsolidationResult(action=action, target_id=target, reason=reason)
    return ConsolidationResult(action=action, target_id="", reason=reason)


def decide(
    llm: LLMProvider,
    *,
    model: str,
    candidate: CandidateMemory,
    similar: list[Memory],
    temperature: float = 0.0,
) -> ConsolidationResult:
    """LLM decision with deterministic fallbacks. Never raises."""
    if not similar:
        return ConsolidationResult(action="ADD", target_id="", reason="no similar memories")
    existing_block = "\n".join(f"- [{m.id}] ({m.type}) {m.text}" for m in similar)
    user_block = f"CANDIDATE: ({candidate.type}) {candidate.text}\nEXISTING:\n{existing_block}"
    try:
        resp = llm.chat(
            ChatRequest(
                messages=(
                    ChatMessage(role="system", content=CONSOLIDATION_PROMPT),
                    ChatMessage(role="user", content=user_block[:8000]),
                ),
                model=model,
                temperature=temperature,
                max_tokens=256,
            )
        )
        parsed = _parse_decision(resp.text, similar)
        if parsed is not None:
            return parsed
    except LLMError:
        pass
    return ConsolidationResult(action="ADD", target_id="", reason="decision fallback")


def consolidate_candidate(
    llm: LLMProvider,
    *,
    model: str,
    candidate: CandidateMemory,
    existing: list[Memory],
    temperature: float = 0.0,
) -> ConsolidationResult:
    """Full pipeline for one candidate: guards first, LLM decision second."""
    if exact_duplicate(candidate, existing) is not None:
        return ConsolidationResult(action="NOOP", target_id="", reason="exact duplicate")
    dup = near_duplicate(candidate, existing)
    if dup is not None:
        return ConsolidationResult(action="NOOP", target_id=dup.id, reason="near duplicate")
    contra = find_contradiction(candidate, existing)
    if contra is not None:
        return ConsolidationResult(
            action="SUPERSEDE", target_id=contra.id, reason="explicit correction of same fact"
        )
    similar = find_similar(candidate, existing)
    return decide(llm, model=model, candidate=candidate, similar=similar, temperature=temperature)


__all__ = [
    "ConsolidationResult",
    "consolidate_candidate",
    "contradicts",
    "decide",
    "exact_duplicate",
    "find_contradiction",
    "find_similar",
    "is_correction",
    "near_duplicate",
]
