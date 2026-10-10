"""Benchmark smoke tests (FakeLLM) + grading unit tests."""

from __future__ import annotations

from selflearn.bench import BenchConfig, grade_answer
from selflearn.bench.harness import run_benchmark
from selflearn.bench.scenarios import CATEGORIES, SCENARIOS
from selflearn.config import Config
from selflearn.llm.fake import FakeLLM


def test_scenario_count_and_categories() -> None:
    assert len(SCENARIOS) >= 40
    assert set(CATEGORIES) == {s.category for s in SCENARIOS}
    for scn in SCENARIOS:
        assert scn.sessions and scn.questions
        for q in scn.questions:
            assert q.keywords


def test_grading_exact() -> None:
    g = grade_answer("I live in Lisbon now.", keywords=["lisbon"], must_absent=["berlin"])
    assert g.correct and not g.stale_hit
    stale = grade_answer(
        "I live in Berlin and Lisbon.", keywords=["lisbon"], must_absent=["berlin"]
    )
    assert not stale.correct and stale.stale_hit
    missing = grade_answer("I live somewhere.", keywords=["lisbon"], must_absent=[])
    assert not missing.correct


def test_benchmark_smoke_fake() -> None:
    """CI smoke: harness runs end to end, labeled as smoke, never as evidence."""
    cfg = Config(provider="fake", model="fake")
    bench = BenchConfig(strategies=("no_memory", "hybrid"), runs=1, limit=2, throttle_s=0)
    report = run_benchmark(cfg, bench=bench, llm=FakeLLM(["ok"]))
    assert report["meta"]["label"] == "CI smoke only"
    assert set(report["summary"]) == {"no_memory", "hybrid"}
    assert report["meta"]["scenarios"] == 2
    for row in report["rows"]:
        assert {"correct", "hit", "prompt_tokens", "latency_s"} <= set(row)
