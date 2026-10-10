"""SQLite memory store. User + namespace scoping enforced in every query."""

from __future__ import annotations

import builtins
import os
import sqlite3
import uuid
from datetime import UTC, datetime

from selflearn.store.migrations import migrate
from selflearn.store.models import VALID_TYPES, Memory, MemoryVersion


def utcnow() -> str:
    """Current UTC time as ISO string."""
    return datetime.now(UTC).isoformat()


class MemoryStore:
    """Per-user isolated memory persistence with full audit history."""

    def __init__(self, path: str = ":memory:") -> None:
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
        self.path = path
        # check_same_thread=False: the REST server answers each request on a
        # worker thread sharing this connection; writes stay serialized by commit.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        migrate(self._conn, now=utcnow())

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()

    # -- mapping ---------------------------------------------------------------
    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            user=row["user_id"],
            namespace=row["namespace"],
            type=row["type"],
            text=row["text"],
            importance=float(row["importance"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"],
            source=row["source"],
            version=int(row["version"]),
            deleted=bool(row["deleted"]),
            superseded_by=row["superseded_by"],
        )

    def _record_history(self, mem: Memory, *, change: str, now: str) -> None:
        self._conn.execute(
            "INSERT INTO memory_history "
            "(memory_id, user_id, version, type, text, importance, changed_at, change) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (mem.id, mem.user, mem.version, mem.type, mem.text, mem.importance, now, change),
        )

    # -- writes ------------------------------------------------------------------
    def create(
        self,
        *,
        user: str,
        type: str,
        text: str,
        importance: float = 0.5,
        source: str = "",
        namespace: str = "default",
    ) -> Memory:
        """Create a memory scoped to (user, namespace)."""
        if type not in VALID_TYPES:
            raise ValueError(f"bad memory type {type!r}; choose from {VALID_TYPES}")
        if not text.strip():
            raise ValueError("memory text must not be empty")
        now = utcnow()
        mem = Memory(
            id=uuid.uuid4().hex[:16],
            user=user,
            namespace=namespace,
            type=type,
            text=text.strip(),
            importance=max(0.0, min(1.0, importance)),
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
            source=source,
            version=1,
        )
        self._conn.execute(
            "INSERT INTO memories (id, user_id, namespace, type, text, importance, "
            "created_at, updated_at, last_accessed_at, source, version, deleted, superseded_by)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, '')",
            (
                mem.id,
                mem.user,
                mem.namespace,
                mem.type,
                mem.text,
                mem.importance,
                mem.created_at,
                mem.updated_at,
                mem.last_accessed_at,
                mem.source,
                mem.version,
            ),
        )
        self._record_history(mem, change="created", now=now)
        self._conn.commit()
        return mem

    def update(
        self,
        *,
        user: str,
        memory_id: str,
        text: str | None = None,
        importance: float | None = None,
        namespace: str = "default",
    ) -> Memory | None:
        """Update text/importance; bumps version; returns None when not found."""
        mem = self.get(user=user, memory_id=memory_id, namespace=namespace)
        if mem is None or mem.deleted:
            return None
        now = utcnow()
        new = Memory(
            id=mem.id,
            user=mem.user,
            namespace=mem.namespace,
            type=mem.type,
            text=text.strip() if text is not None else mem.text,
            importance=max(0.0, min(1.0, importance)) if importance is not None else mem.importance,
            created_at=mem.created_at,
            updated_at=now,
            last_accessed_at=now,
            source=mem.source,
            version=mem.version + 1,
            deleted=False,
            superseded_by=mem.superseded_by,
        )
        self._conn.execute(
            "UPDATE memories SET text=?, importance=?, updated_at=?, "
            "last_accessed_at=?, version=? WHERE id=? AND user_id=? AND namespace=?",
            (
                new.text,
                new.importance,
                new.updated_at,
                new.last_accessed_at,
                new.version,
                memory_id,
                user,
                namespace,
            ),
        )
        self._record_history(new, change="updated", now=now)
        self._conn.commit()
        return new

    def soft_delete(self, *, user: str, memory_id: str, namespace: str = "default") -> bool:
        """Soft-delete. Returns False when the memory is not yours."""
        mem = self.get(user=user, memory_id=memory_id, namespace=namespace)
        if mem is None or mem.deleted:
            return False
        now = utcnow()
        self._conn.execute(
            "UPDATE memories SET deleted=1, updated_at=? WHERE id=? AND user_id=? AND namespace=?",
            (now, memory_id, user, namespace),
        )
        self._record_history(mem, change="deleted", now=now)
        self._conn.commit()
        return True

    def supersede(self, *, user: str, old_id: str, new_id: str, namespace: str = "default") -> bool:
        """Link old memory as superseded by new. Both must belong to (user, namespace)."""
        old = self.get(user=user, memory_id=old_id, namespace=namespace)
        new = self.get(user=user, memory_id=new_id, namespace=namespace)
        if old is None or new is None or old.deleted:
            return False
        now = utcnow()
        self._conn.execute(
            "UPDATE memories SET superseded_by=?, updated_at=? "
            "WHERE id=? AND user_id=? AND namespace=?",
            (new_id, now, old_id, user, namespace),
        )
        self._record_history(old, change="superseded", now=now)
        self._conn.commit()
        return True

    def touch(self, *, user: str, memory_id: str, namespace: str = "default") -> None:
        """Refresh last-accessed timestamp (best effort; never raises)."""
        try:
            self._conn.execute(
                "UPDATE memories SET last_accessed_at=? "
                "WHERE id=? AND user_id=? AND namespace=? AND deleted=0",
                (utcnow(), memory_id, user, namespace),
            )
            self._conn.commit()
        except sqlite3.Error:
            pass

    def adjust_importance(
        self, *, user: str, memory_id: str, delta: float, namespace: str = "default"
    ) -> Memory | None:
        """Shift importance by delta, clamped to [0, 1]. Returns None when not found."""
        mem = self.get(user=user, memory_id=memory_id, namespace=namespace)
        if mem is None or mem.deleted:
            return None
        return self.update(
            user=user,
            memory_id=memory_id,
            importance=mem.importance + delta,
            namespace=namespace,
        )

    # -- reads (always scoped) -----------------------------------------------------
    def get(self, *, user: str, memory_id: str, namespace: str = "default") -> Memory | None:
        """Fetch one memory. None unless it belongs to (user, namespace)."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id=? AND user_id=? AND namespace=?",
            (memory_id, user, namespace),
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def list(
        self,
        *,
        user: str,
        namespace: str = "default",
        include_deleted: bool = False,
        include_superseded: bool = False,
        type: str | None = None,
        limit: int = 200,
    ) -> builtins.list[Memory]:
        """List memories for (user, namespace) only. Soft-deleted hidden by default."""
        query = "SELECT * FROM memories WHERE user_id=? AND namespace=?"
        params: builtins.list[object] = [user, namespace]
        if not include_deleted:
            query += " AND deleted=0"
        if not include_superseded:
            query += " AND superseded_by=''"
        if type is not None:
            query += " AND type=?"
            params.append(type)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def history(self, *, user: str, memory_id: str) -> builtins.list[MemoryVersion]:
        """Audit trail for one memory. Empty unless it belongs to user."""
        owner = self._conn.execute(
            "SELECT user_id FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if owner is None or owner["user_id"] != user:
            return []
        rows = self._conn.execute(
            "SELECT * FROM memory_history WHERE memory_id=? ORDER BY version, rowid",
            (memory_id,),
        ).fetchall()
        return [
            MemoryVersion(
                memory_id=r["memory_id"],
                user=r["user_id"],
                version=int(r["version"]),
                type=r["type"],
                text=r["text"],
                importance=float(r["importance"]),
                changed_at=r["changed_at"],
                change=r["change"],
            )
            for r in rows
        ]

    # -- privacy primitives ----------------------------------------------------------
    def export_user(self, *, user: str) -> builtins.list[dict[str, object]]:
        """Full per-user export across namespaces (includes deleted + history refs)."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE user_id=? ORDER BY created_at", (user,)
        ).fetchall()
        return [self._row_to_memory(r).to_dict() for r in rows]

    def delete_user(self, *, user: str) -> int:
        """Hard-delete ALL of a user's data (memories + history). Returns rows removed."""
        h = self._conn.execute("DELETE FROM memory_history WHERE user_id=?", (user,)).rowcount
        m = self._conn.execute("DELETE FROM memories WHERE user_id=?", (user,)).rowcount
        self._conn.commit()
        return (h or 0) + (m or 0)

    def prune(
        self,
        *,
        user: str,
        namespace: str = "default",
        max_age_days: float = 365.0,
        min_importance: float = 0.2,
    ) -> int:
        """Hard-delete old low-importance memories. Returns count removed."""
        from datetime import timedelta  # local import: only user here

        cutoff = (datetime.now(UTC) - timedelta(days=max_age_days)).isoformat()
        cur = self._conn.execute(
            "DELETE FROM memories WHERE user_id=? AND namespace=? "
            "AND importance < ? AND updated_at < ?",
            (user, namespace, min_importance, cutoff),
        )
        self._conn.commit()
        return cur.rowcount or 0


__all__ = ["MemoryStore", "utcnow"]
