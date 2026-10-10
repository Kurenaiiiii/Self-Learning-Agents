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

__all__ = [
    "CandidateMemory",
    "ConsolidationResult",
    "consolidate_candidate",
    "decide",
    "exact_duplicate",
    "extract_memories",
    "find_similar",
    "near_duplicate",
    "parse_extraction_output",
]
