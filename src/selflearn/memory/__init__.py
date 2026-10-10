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
    "consolidate_candidate",
    "decide",
    "estimate_tokens",
    "exact_duplicate",
    "extract_memories",
    "find_similar",
    "near_duplicate",
    "parse_extraction_output",
    "recency_score",
]
