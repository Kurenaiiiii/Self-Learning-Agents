"""Benchmark smoke tests (FakeLLM) + grading unit tests."""

from __future__ import annotations

from selflearn.bench import ALL_SCENARIOS, LONG_SCENARIOS, BenchConfig, grade_answer, history_bucket
from selflearn.bench.harness import _history_within_budget, run_benchmark
from selflearn.bench.scenarios import CATEGORIES, SCENARIOS
from selflearn.config import Config
from selflearn.llm.fake import FakeLLM
from selflearn.memory.retrieval import estimate_tokens


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


def test_long_scenarios_shape() -> None:
    """16 long histories: 30+ sessions, ~8k+ tokens, chatter flagged skip_learn."""
    assert len(LONG_SCENARIOS) >= 15
    assert len(ALL_SCENARIOS) == len(SCENARIOS) + len(LONG_SCENARIOS)
    for scn in LONG_SCENARIOS:
        assert len(scn.sessions) >= 30, scn.id
        total = estimate_tokens("\n".join(f"{r}: {t}" for s in scn.sessions for r, t in s))
        assert total >= 8000, (scn.id, total)
        learned = [i for i in range(len(scn.sessions)) if i not in scn.skip_learn]
        assert 1 <= len(learned) <= 6, (scn.id, learned)  # only fact sessions learned
        assert scn.questions and all(q.keywords for q in scn.questions)


def test_long_history_truncates_oldest() -> None:
    scn = LONG_SCENARIOS[0]
    full, cut = _history_within_budget(scn, max_tokens=8000)
    assert cut is True  # planted facts at the front must fall out of budget
    short, cut_short = _history_within_budget(SCENARIOS[0], max_tokens=8000)
    assert cut_short is False and "Lisbon" in short
    assert history_bucket(100) == "short"
    assert history_bucket(5000) == "mid"
    assert history_bucket(9000) == "long"
