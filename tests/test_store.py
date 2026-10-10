"""Store tests: CRUD, history, and per-user/per-namespace isolation."""

from __future__ import annotations

import pytest

from selflearn.store import MemoryStore


@pytest.fixture()
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_crud_roundtrip(store: MemoryStore) -> None:
    mem = store.create(user="u1", type="fact", text="likes latte", source="chat")
    assert mem.version == 1 and mem.deleted is False
    got = store.get(user="u1", memory_id=mem.id)
    assert got is not None and got.text == "likes latte"
    updated = store.update(user="u1", memory_id=mem.id, text="likes oat latte")
    assert updated is not None and updated.version == 2
    assert store.soft_delete(user="u1", memory_id=mem.id) is True
    assert store.list(user="u1") == []
    assert store.list(user="u1", include_deleted=True)[0].deleted is True


def test_rejects_bad_input(store: MemoryStore) -> None:
    with pytest.raises(ValueError):
        store.create(user="u1", type="nope", text="x")
    with pytest.raises(ValueError):
        store.create(user="u1", type="fact", text="   ")


def test_history_audit_trail(store: MemoryStore) -> None:
    mem = store.create(user="u1", type="preference", text="v1")
    store.update(user="u1", memory_id=mem.id, text="v2")
    trail = store.history(user="u1", memory_id=mem.id)
    assert [h.change for h in trail] == ["created", "updated"]
    assert [h.version for h in trail] == [1, 2]


def test_supersede_links(store: MemoryStore) -> None:
    old = store.create(user="u1", type="fact", text="lives in Berlin")
    new = store.create(user="u1", type="fact", text="lives in Lisbon")
    assert store.supersede(user="u1", old_id=old.id, new_id=new.id) is True
    assert store.get(user="u1", memory_id=old.id).superseded_by == new.id  # type: ignore[union-attr]
    # Superseded hidden from default listing.
    assert all(m.id != old.id for m in store.list(user="u1"))
    assert any(m.id == old.id for m in store.list(user="u1", include_superseded=True))


def test_user_isolation(store: MemoryStore) -> None:
    """One user can never read, update or delete another's memories."""
    a = store.create(user="alice", type="fact", text="alice secret")
    store.create(user="bob", type="fact", text="bob secret")
    # Read
    assert store.get(user="bob", memory_id=a.id) is None
    assert store.history(user="bob", memory_id=a.id) == []
    assert all(m.user == "bob" for m in store.list(user="bob"))
    assert all(m.user == "alice" for m in store.list(user="alice"))
    # Update
    assert store.update(user="bob", memory_id=a.id, text="hijack") is None
    assert store.get(user="alice", memory_id=a.id).text == "alice secret"  # type: ignore[union-attr]
    # Delete
    assert store.soft_delete(user="bob", memory_id=a.id) is False
    assert store.supersede(user="bob", old_id=a.id, new_id=a.id) is False
    # Export/delete-user scoped
    assert all(m["user"] == "bob" for m in store.export_user(user="bob"))
    removed = store.delete_user(user="bob")
    assert removed > 0
    assert store.list(user="bob", include_deleted=True) == []
    assert store.get(user="alice", memory_id=a.id) is not None


def test_namespace_isolation(store: MemoryStore) -> None:
    """Namespaces partition one user's memories."""
    store.create(user="u1", type="fact", text="work note", namespace="work")
    store.create(user="u1", type="fact", text="home note", namespace="home")
    assert [m.text for m in store.list(user="u1", namespace="work")] == ["work note"]
    assert [m.text for m in store.list(user="u1", namespace="home")] == ["home note"]
    assert store.list(user="u1") == []  # default namespace is separate
    assert store.get(user="u1", memory_id="x", namespace="work") is None


def test_touch_and_importance(store: MemoryStore) -> None:
    mem = store.create(user="u1", type="fact", text="x", importance=0.5)
    bumped = store.adjust_importance(user="u1", memory_id=mem.id, delta=0.3)
    assert bumped is not None and bumped.importance == pytest.approx(0.8)
    store.touch(user="u1", memory_id=mem.id)  # never raises
    store.touch(user="u1", memory_id="missing")  # never raises
    assert store.adjust_importance(user="u2", memory_id=mem.id, delta=0.1) is None


def test_prune_policy(store: MemoryStore) -> None:
    mem = store.create(user="u1", type="fact", text="old junk", importance=0.05)
    # Fresh memories survive even when unimportant.
    assert store.prune(user="u1", max_age_days=365, min_importance=0.2) == 0
    # Ancient memories do not.
    store._conn.execute(
        "UPDATE memories SET updated_at=? WHERE id=?", ("2000-01-01T00:00:00+00:00", mem.id)
    )
    store._conn.commit()
    assert store.prune(user="u1", max_age_days=365, min_importance=0.2) == 1
    assert store.get(user="u1", memory_id=mem.id) is None
