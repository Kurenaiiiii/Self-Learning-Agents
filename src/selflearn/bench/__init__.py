"""Benchmark harness (full implementation lands with the scenarios)."""

from __future__ import annotations

from typing import Any

from selflearn.config import Config

STRATEGIES = ("no_memory", "full_history", "summary_only", "bm25_only", "vector_only", "hybrid")


def run_benchmark(
    cfg: Config,
    *,
    strategy: str = "hybrid",
    runs: int = 3,
    scenarios_path: str | None = None,
) -> dict[str, Any]:
    """Run the benchmark (implemented in the benchmark phase)."""
    raise NotImplementedError("benchmark harness not implemented yet")


__all__ = ["STRATEGIES", "run_benchmark"]
