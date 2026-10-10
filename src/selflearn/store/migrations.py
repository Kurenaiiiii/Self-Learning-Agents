"""SQLite schema migrations. Versioned, idempotent, stdlib only."""

from __future__ import annotations

import sqlite3

MIGRATION_001 = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    type TEXT NOT NULL,
    text TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    version INTEGER NOT NULL DEFAULT 1,
    deleted INTEGER NOT NULL DEFAULT 0,
    superseded_by TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_memories_user_ns ON memories(user_id, namespace, deleted);
CREATE TABLE IF NOT EXISTS memory_history (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    type TEXT NOT NULL,
    text TEXT NOT NULL,
    importance REAL NOT NULL,
    changed_at TEXT NOT NULL,
    change TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_mem ON memory_history(memory_id);
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""

MIGRATIONS: tuple[tuple[int, str], ...] = ((1, MIGRATION_001),)


def migrate(conn: sqlite3.Connection, *, now: str) -> int:
    """Apply pending migrations. Returns schema version."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    for version, sql in MIGRATIONS:
        if version not in applied:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, now),
            )
    conn.commit()
    return max((v for v, _ in MIGRATIONS), default=0)


__all__ = ["MIGRATIONS", "migrate"]
