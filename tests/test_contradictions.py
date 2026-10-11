"""Regression tests: casual contradiction phrasings must SUPERSEDE deterministically."""

from __future__ import annotations

import pytest

from selflearn.llm.fake import FakeLLM
from selflearn.memory import (
    CandidateMemory,
    consolidate_candidate,
    contradicts,
    is_correction,
)
from selflearn.store import MemoryStore

# (existing memory text, candidate text) — every pair must SUPERSEDE.
CONTRADICTIONS: list[tuple[str, str]] = [
    ("user lives in Berlin", "the user actually lives in Lisbon, not Berlin"),
    ("user moved from Berlin to Lisbon", "correction: user moved from Munich, not Berlin"),
    ("user drinks cow milk", "user switched to oat milk instead of cow milk"),
    ("user uses VS Code", "user no longer uses VS Code, moved to Neovim"),
    ("user drives a Honda Civic", "user drives a Tesla, not a Honda"),
    ("dentist appointment on Friday", "update: dentist appointment is Monday, not Friday"),
    ("user follows a vegan diet", "my mistake, user follows a vegetarian diet, not vegan"),
    ("user prefers coffee", "scratch that, user prefers tea over coffee"),
    ("user works as frontend developer", "user now works as backend engineer, not frontend"),
    ("user has a golden retriever named Biscuit", "oops, user dog is named Max, not Biscuit"),
    ("user takes meetings at 9am", "user rather takes meetings after 10am, not 9am"),
    ("user is learning Spanish", "user dropped Spanish and is learning Portuguese now"),
    ("gym membership renews every January", "user changed gym renewal to February, not January"),
    ("passport expires in June 2027", "user passport expires in 2028, not 2027"),
    ("user runs every morning", "user swims every morning instead of running, knee hurts"),
    ("user reads fiction novels", "actually user only reads non-fiction history books"),
    ("user meeting in Paris on Tuesday", "meeting moved from Paris to Lyon, not Paris"),
]

# (existing, candidate) — must NOT deterministic-fire (LLM decides instead).
NON_CONTRADICTIONS: list[tuple[str, str]] = [
    ("user drinks oat latte", "user drinks espresso on weekends"),
    ("user moved from Berlin to Lisbon", "user visited Berlin last year"),
    ("user lives in Berlin", "actually, user bought new running shoes"),
    ("user likes meetings", "user does not eat meat"),
]


@pytest.fixture()
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


@pytest.mark.parametrize("existing_text,candidate_text", CONTRADICTIONS)
def test_casual_contradiction_supersedes(
    store: MemoryStore, existing_text: str, candidate_text: str
) -> None:
    assert len(CONTRADICTIONS) >= 15
    mem = store.create(user="u", type="fact", text=existing_text)
    existing = store.list(user="u", include_superseded=True)
    assert contradicts(candidate_text, existing_text)
    llm = FakeLLM()
    res = consolidate_candidate(
        llm,
        model="m",
        candidate=CandidateMemory("fact", candidate_text, 0.6),
        existing=existing,
    )
    assert res.action == "SUPERSEDE", candidate_text
    assert res.target_id == mem.id
    assert llm.calls == 0  # deterministic guard: no model call spent


@pytest.mark.parametrize("existing_text,candidate_text", NON_CONTRADICTIONS)
def test_no_false_supersede(store: MemoryStore, existing_text: str, candidate_text: str) -> None:
    store.create(user="u", type="fact", text=existing_text)
    existing = store.list(user="u", include_superseded=True)
    assert not contradicts(candidate_text, existing_text)
    llm = FakeLLM(['{"action": "ADD", "target_id": "", "reason": "new"}'])
    res = consolidate_candidate(
        llm,
        model="m",
        candidate=CandidateMemory("fact", candidate_text, 0.5),
        existing=existing,
    )
    assert res.action == "ADD"
    assert llm.calls == 1  # fell through to the LLM as designed


def test_is_correction_cues() -> None:
    for cue in (
        "actually x",
        "correction: y",
        "update: z",
        "switched to b",
        "moved from a",
        "no longer",
        "my mistake",
        "i meant q",
        "scratch that",
        "rather late",
        "not Berlin",
        "oops here",
    ):
        assert is_correction(cue), cue
    assert not is_correction("user drinks oat latte every morning")
