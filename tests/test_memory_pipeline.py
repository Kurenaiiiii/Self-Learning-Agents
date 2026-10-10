"""Extraction + consolidation tests with FakeLLM, incl. malformed output."""

from __future__ import annotations

import json

import pytest

from selflearn.llm.fake import FakeLLM
from selflearn.llm.types import ServerError
from selflearn.memory import (
    CandidateMemory,
    consolidate_candidate,
    exact_duplicate,
    extract_memories,
    find_similar,
    near_duplicate,
    parse_extraction_output,
)
from selflearn.store import MemoryStore

GOOD_JSON = (
    '{"memories": ['
    '{"type": "preference", "text": "likes oat latte", "importance": 0.7}, '
    '{"type": "fact", "text": "lives in Lisbon", "importance": 0.9}]}'
)


@pytest.fixture()
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_parse_valid_and_fenced() -> None:
    cands = parse_extraction_output(GOOD_JSON)
    assert [c.text for c in cands] == ["likes oat latte", "lives in Lisbon"]
    fenced = "```json\n" + GOOD_JSON + "\n```"
    assert len(parse_extraction_output(fenced)) == 2


def test_parse_malformed_returns_empty() -> None:
    for bad in (
        "",
        "not json",
        "[1, 2]",
        '{"memories": "nope"}',
        '{"x": 1}',
        '{"memories": [{"type": "nope", "text": "x"}]}',
        '{"memories": [{"type": "fact", "text": "   "}]}',
    ):
        assert parse_extraction_output(bad) == [], bad


def test_extract_happy_path() -> None:
    cands = extract_memories(FakeLLM([GOOD_JSON]), model="m", conversation="i love oat latte")
    assert len(cands) == 2 and cands[0].type == "preference"


def test_extract_repairs_after_garbage() -> None:
    llm = FakeLLM(["garbage {{{", GOOD_JSON])
    cands = extract_memories(llm, model="m", conversation="hi")
    assert len(cands) == 2  # second (repair) call produced valid JSON


def test_extract_safe_fallback_on_persistent_garbage() -> None:
    llm = FakeLLM(["garbage", "still garbage"])
    assert extract_memories(llm, model="m", conversation="hi") == []


def test_extract_never_raises_on_llm_error() -> None:
    llm = FakeLLM([ServerError("boom", provider="fake")])
    assert extract_memories(llm, model="m", conversation="hi") == []


def test_exact_and_near_duplicates(store: MemoryStore) -> None:
    store.create(user="u", type="fact", text="Lives in  Lisbon")
    existing = store.list(user="u", include_superseded=True)
    assert exact_duplicate(CandidateMemory("fact", "lives in lisbon", 0.5), existing) is not None
    assert exact_duplicate(CandidateMemory("fact", "lives in Porto", 0.5), existing) is None
    assert near_duplicate(CandidateMemory("fact", "Lives in Lisbon!", 0.5), existing) is not None
    assert near_duplicate(CandidateMemory("fact", "likes chess", 0.5), existing) is None
    assert near_duplicate(CandidateMemory("fact", "x", 0.5), []) is None
    assert find_similar(CandidateMemory("fact", "x", 0.5), []) == []


def test_consolidate_noop_on_duplicates(store: MemoryStore) -> None:
    store.create(user="u", type="fact", text="lives in Lisbon")
    existing = store.list(user="u", include_superseded=True)
    llm = FakeLLM()
    res = consolidate_candidate(
        llm,
        model="m",
        candidate=CandidateMemory("fact", "Lives in Lisbon", 0.5),
        existing=existing,
    )
    assert res.action == "NOOP"
    assert llm.calls == 0  # guards short-circuit: no LLM call spent


def test_consolidate_supersede_contradiction(store: MemoryStore) -> None:
    old = store.create(user="u", type="fact", text="lives in Berlin")
    existing = store.list(user="u", include_superseded=True)
    llm = FakeLLM([json.dumps({"action": "SUPERSEDE", "target_id": old.id, "reason": "moved"})])
    res = consolidate_candidate(
        llm,
        model="m",
        candidate=CandidateMemory("fact", "lives in Lisbon now", 0.8),
        existing=existing,
    )
    assert res.action == "SUPERSEDE" and res.target_id == old.id


def test_consolidate_rejects_unknown_target(store: MemoryStore) -> None:
    store.create(user="u", type="fact", text="lives in Berlin")
    existing = store.list(user="u", include_superseded=True)
    llm = FakeLLM(['{"action": "UPDATE", "target_id": "bogus", "reason": "x"}'])
    res = consolidate_candidate(
        llm,
        model="m",
        candidate=CandidateMemory("fact", "enjoys sailing trips", 0.5),
        existing=existing,
    )
    assert res.action == "ADD"  # invalid target falls back to ADD


def test_consolidate_add_when_nothing_similar() -> None:
    llm = FakeLLM()
    res = consolidate_candidate(
        llm,
        model="m",
        candidate=CandidateMemory("fact", "brand new info here", 0.5),
        existing=[],
    )
    assert res.action == "ADD"
    assert llm.calls == 0
