"""Memory record types."""

from __future__ import annotations

from dataclasses import dataclass

VALID_TYPES = ("fact", "preference", "episode", "lesson")


@dataclass(frozen=True)
class Memory:
    """One stored memory. Timestamps are UTC ISO strings."""

    id: str
    user: str
    namespace: str
    type: str
    text: str
    importance: float
    created_at: str
    updated_at: str
    last_accessed_at: str
    source: str
    version: int
    deleted: bool = False
    superseded_by: str = ""

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable view (no secrets; text already redacted at write)."""
        return {
            "id": self.id,
            "user": self.user,
            "namespace": self.namespace,
            "type": self.type,
            "text": self.text,
            "importance": self.importance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "source": self.source,
            "version": self.version,
            "deleted": self.deleted,
            "superseded_by": self.superseded_by,
        }


@dataclass(frozen=True)
class MemoryVersion:
    """One row of the audit history: a past version of a memory."""

    memory_id: str
    user: str
    version: int
    type: str
    text: str
    importance: float
    changed_at: str
    change: str  # created | updated | deleted | superseded | restored


__all__ = ["VALID_TYPES", "Memory", "MemoryVersion"]
