"""Retrieval tests: ranking, token budget, recency, superseded exclusion, explain."""

from __future__ import annotations

import pytest

from selflearn.memory import Retriever, estimate_tokens, recency_score
from selflearn.store import MemoryStore


@pytest.fixture()
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    s.create(
        user="u", type="preference", text="drinks oat milk latte every morning", importance=0.8
    )
    s.create(user="u", type="fact", text="moved from Berlin to Lisbon in March", importance=0.9)
    s.create(user="u", type="fact", text="has a golden retriever named Biscuit", importance=0.5)
    s.create(user="u", type="fact", text="plays chess on weekends", importance=0.3)
    yield s
    s.close()


def test_relevant_memory_ranked_first(store: MemoryStore) -> None:
    r = Retriever(store, user="u")
    top = r.retrieve("what does the user drink in the morning?")
    assert top[0].memory.text.startswith("drinks oat milk")
    assert top[0].score > 0


def test_components_explain_score(store: MemoryStore) -> None:
    r = Retriever(store, user="u")
    top = r.retrieve("Lisbon Berlin moved")
    assert top[0].memory.text.startswith("moved from Berlin")
    assert set(top[0].components) == {"bm25", "vector", "recency", "importance"}
    for expl in r.explain("Lisbon"):
        assert {"id", "text", "score", "components", "est_tokens"} <= set(expl)


def test_token_budget_respected(store: MemoryStore) -> None:
    r = Retriever(store, user="u")
    _, used = r.build_context("user", top_k=10, max_tokens=12)
    assert sum(estimate_tokens(f"- [{m.memory.type}] {m.memory.text}") for m in used) <= 24
    assert len(used) >= 1  # first memory always fits even over budget


def test_superseded_and_deleted_excluded(store: MemoryStore) -> None:
    old = store.create(user="u", type="fact", text="lives in Berlin apartment")
    new = store.create(user="u", type="fact", text="lives in Lisbon apartment")
    store.supersede(user="u", old_id=old.id, new_id=new.id)
    r = Retriever(store, user="u")
    ids = [x.memory.id for x in r.retrieve("lives in Berlin apartment", top_k=10)]
    assert old.id not in ids and new.id in ids
    store.soft_delete(user="u", memory_id=new.id)
    assert new.id not in [x.memory.id for x in r.retrieve("Lisbon apartment", top_k=10)]


def test_recency_prefers_fresher(store: MemoryStore) -> None:
    store.create(user="u", type="fact", text="duplicate note alpha", importance=0.5)
    fresh = store.create(user="u", type="fact", text="duplicate note alpha", importance=0.5)
    store._conn.execute(
        "UPDATE memories SET updated_at=?, last_accessed_at=? WHERE importance=0.5 AND id!=?",
        ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", fresh.id),
    )
    store._conn.commit()
    r = Retriever(
        store, user="u", weights={"bm25": 0.3, "vector": 0.3, "recency": 0.35, "importance": 0.05}
    )
    top = r.retrieve("duplicate note alpha", top_k=2)
    assert top[0].memory.id == fresh.id


def test_recency_score_shape() -> None:
    assert recency_score("2000-01-01T00:00:00+00:00") < 0.01
    assert recency_score("not-a-date") == 0.5
    from selflearn.store import utcnow

    assert recency_score(utcnow()) == pytest.approx(1.0, abs=0.01)


def test_empty_query_and_other_user(store: MemoryStore) -> None:
    r = Retriever(store, user="u")
    assert r.retrieve("") == []
    assert Retriever(store, user="stranger").retrieve("latte") == []
    ctx, used = Retriever(store, user="stranger").build_context("latte")
    assert ctx == "" and used == []
