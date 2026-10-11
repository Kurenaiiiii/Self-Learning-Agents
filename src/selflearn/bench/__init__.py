"""Benchmark: reproducible memory measurement with labeled reports."""

from selflearn.bench.grading import Grade, grade_answer, normalize
from selflearn.bench.harness import BenchConfig, history_bucket, run_benchmark, run_scenario
from selflearn.bench.scenarios import ALL_SCENARIOS, CATEGORIES, LONG_SCENARIOS, SCENARIOS

STRATEGIES = (
    "no_memory",
    "full_history",
    "summary_only",
    "bm25_only",
    "vector_only",
    "hybrid",
)

__all__ = [
    "ALL_SCENARIOS",
    "CATEGORIES",
    "LONG_SCENARIOS",
    "SCENARIOS",
    "STRATEGIES",
    "BenchConfig",
    "Grade",
    "grade_answer",
    "history_bucket",
    "normalize",
    "run_benchmark",
    "run_scenario",
]
