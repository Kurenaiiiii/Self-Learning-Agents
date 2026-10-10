"""Memory pipeline: extraction + consolidation (+ retrieval, agent later)."""

from selflearn.memory.consolidation import (
    ConsolidationResult,
    consolidate_candidate,
    decide,
    exact_duplicate,
    find_similar,
    near_duplicate,
)
from selflearn.memory.extraction import CandidateMemory, extract_memories, parse_extraction_output
from selflearn.memory.learning import (
    apply_feedback,
    apply_forgetting,
    reflect_session,
    save_lessons,
)
from selflearn.memory.privacy import (
    enforce_retention,
    export_user_json,
    forget_me,
    redact_candidate,
    redact_text,
)
from selflearn.memory.retrieval import (
    DEFAULT_WEIGHTS,
    RankedMemory,
    Retriever,
    estimate_tokens,
    recency_score,
)

__all__ = [
    "CandidateMemory",
    "ConsolidationResult",
    "DEFAULT_WEIGHTS",
    "RankedMemory",
    "Retriever",
    "apply_feedback",
    "apply_forgetting",
    "consolidate_candidate",
    "decide",
    "enforce_retention",
    "estimate_tokens",
    "exact_duplicate",
    "export_user_json",
    "extract_memories",
    "find_similar",
    "forget_me",
    "near_duplicate",
    "parse_extraction_output",
    "recency_score",
    "redact_candidate",
    "redact_text",
    "reflect_session",
    "save_lessons",
]
