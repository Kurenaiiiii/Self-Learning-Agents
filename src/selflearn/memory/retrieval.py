"""Hybrid retrieval: BM25 + vector + recency + importance, with token budget.

Superseded and soft-deleted memories are excluded by default. Every score
breaks down into explainable components (see explain()).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from selflearn.embeddings.hybrid import Bm25Index, NgramTfidfIndex
from selflearn.store.models import Memory
from selflearn.store.sqlite import MemoryStore

DEFAULT_WEIGHTS = {"bm25": 0.4, "vector": 0.3, "recency": 0.15, "importance": 0.15}
RECENCY_HALF_LIFE_DAYS = 60.0


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars/4). Approximation, documented, stdlib-only."""
    return max(1, len(text) // 4)


def recency_score(updated_at: str, *, half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> float:
    """Exponential decay from updated_at to now: 1.0 fresh, -> 0.0 ancient."""
    try:
        updated = datetime.fromisoformat(updated_at)
    except ValueError:
        return 0.5
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    age_days = (datetime.now(UTC) - updated).total_seconds() / 86400.0
    if age_days < 0:
        return 1.0
    return float(0.5 ** (age_days / half_life_days))


@dataclass(frozen=True)
class RankedMemory:
    """A memory with its final score and per-signal components."""

    memory: Memory
    score: float
    components: dict[str, float] = field(default_factory=dict)


class Retriever:
    """Hybrid ranker over one user's memories in one namespace."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        user: str,
        namespace: str = "default",
        weights: dict[str, float] | None = None,
    ) -> None:
        self.store = store
        self.user = user
        self.namespace = namespace
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    def _candidates(self) -> list[Memory]:
        # ponytail: full scan + temp indexes; fine until 10k+ memories (then persist vectors).
        return self.store.list(user=self.user, namespace=self.namespace)

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RankedMemory]:
        """Rank memories for the query, best first."""
        cands = self._candidates()
        if not cands or not query.strip():
            return []
        bm25 = Bm25Index()
        vec = NgramTfidfIndex()
        for mem in cands:
            bm25.add(mem.id, mem.text)
            vec.add(mem.id, mem.text)
        raw_bm25 = {mem.id: bm25.score(query, mem.id) for mem in cands}
        peak = max(raw_bm25.values(), default=0.0)
        ranked: list[RankedMemory] = []
        for mem in cands:
            comp = {
                "bm25": (raw_bm25[mem.id] / peak) if peak > 0 else 0.0,
                "vector": vec.cosine(query, mem.id),
                "recency": recency_score(mem.updated_at),
                "importance": mem.importance,
            }
            total = sum(comp[k] * self.weights.get(k, 0.0) for k in comp)
            ranked.append(RankedMemory(memory=mem, score=total, components=comp))
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:top_k]

    def build_context(
        self, query: str, *, top_k: int = 5, max_tokens: int = 800
    ) -> tuple[str, list[RankedMemory]]:
        """Render memories as prompt context within a token budget."""
        used: list[RankedMemory] = []
        lines: list[str] = []
        spent = 0
        for ranked in self.retrieve(query, top_k=top_k):
            line = f"- [{ranked.memory.type}] {ranked.memory.text}"
            cost = estimate_tokens(line)
            if used and spent + cost > max_tokens:
                break
            lines.append(line)
            used.append(ranked)
            spent += cost
        # ponytail: touch() best-effort refresh of last-accessed for used memories.
        for ranked in used:
            self.store.touch(user=self.user, memory_id=ranked.memory.id, namespace=self.namespace)
        return "\n".join(lines), used

    def explain(self, query: str, *, top_k: int = 5) -> list[dict[str, object]]:
        """Debug view: why each memory was retrieved."""
        out: list[dict[str, object]] = []
        for ranked in self.retrieve(query, top_k=top_k):
            out.append(
                {
                    "id": ranked.memory.id,
                    "text": ranked.memory.text,
                    "type": ranked.memory.type,
                    "score": round(ranked.score, 4),
                    "components": {k: round(v, 4) for k, v in ranked.components.items()},
                    "est_tokens": estimate_tokens(ranked.memory.text),
                }
            )
        return out


__all__ = [
    "DEFAULT_WEIGHTS",
    "RECENCY_HALF_LIFE_DAYS",
    "RankedMemory",
    "Retriever",
    "estimate_tokens",
    "recency_score",
]
