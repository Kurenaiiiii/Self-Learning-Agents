"""Feedback, reflection, and forgetting: the learning signals."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from selflearn.llm.base import LLMProvider
from selflearn.llm.types import ChatMessage, ChatRequest, LLMError
from selflearn.memory.extraction import CandidateMemory, parse_extraction_output
from selflearn.store.models import Memory
from selflearn.store.sqlite import MemoryStore

FEEDBACK_UP_DELTA = 0.15
FEEDBACK_DOWN_DELTA = -0.2

REFLECTION_PROMPT = (
    "You review how well the assistant served this user. From the transcript, "
    "extract 0-3 LESSON memories: durable notes on how to serve this user better "
    "(tone, format, topics to prefer/avoid). Skip specifics already covered. "
    "Reply with ONLY strict JSON: "
    '{"memories": [{"type": "lesson", "text": "...", "importance": 0.0-1.0}]}'
)


def apply_feedback(
    store: MemoryStore,
    *,
    user: str,
    memory_ids: list[str],
    direction: str,
    namespace: str = "default",
) -> int:
    """Thumbs up/down shifts importance of the memories used. Returns count changed."""
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")
    delta = FEEDBACK_UP_DELTA if direction == "up" else FEEDBACK_DOWN_DELTA
    changed = 0
    for mid in memory_ids:
        if store.adjust_importance(user=user, memory_id=mid, delta=delta, namespace=namespace):
            changed += 1
    return changed


def reflect_session(
    llm: LLMProvider, *, model: str, transcript: str, temperature: float = 0.0
) -> list[CandidateMemory]:
    """Post-session reflection -> lesson candidates. Validated, never raises."""
    try:
        resp = llm.chat(
            ChatRequest(
                messages=(
                    ChatMessage(role="system", content=REFLECTION_PROMPT),
                    ChatMessage(role="user", content=transcript[:12000]),
                ),
                model=model,
                temperature=temperature,
                max_tokens=512,
            )
        )
        raw = resp.text
    except LLMError:
        return []
    lessons: list[CandidateMemory] = []
    for cand in parse_extraction_output(raw):
        if cand.type == "lesson":
            lessons.append(cand)
    return lessons[:3]


def apply_forgetting(
    store: MemoryStore,
    *,
    user: str,
    namespace: str = "default",
    max_age_days: float = 365.0,
    min_importance: float = 0.2,
    decay_days: float = 90.0,
    decay_factor: float = 0.9,
) -> dict[str, int]:
    """Decay importance of long-unaccessed memories, then prune old + unimportant."""
    cutoff = (datetime.now(UTC) - timedelta(days=decay_days)).isoformat()
    decayed = 0
    for mem in store.list(user=user, namespace=namespace, include_superseded=True):
        if mem.last_accessed_at < cutoff and mem.importance > min_importance:
            store.adjust_importance(
                user=user,
                memory_id=mem.id,
                delta=-(mem.importance * (1 - decay_factor)),
                namespace=namespace,
            )
            decayed += 1
    pruned = store.prune(
        user=user,
        namespace=namespace,
        max_age_days=max_age_days,
        min_importance=min_importance,
    )
    return {"decayed": decayed, "pruned": pruned}


def save_lessons(
    store: MemoryStore, *, user: str, lessons: list[CandidateMemory], namespace: str = "default"
) -> list[Memory]:
    """Persist validated lesson candidates."""
    saved: list[Memory] = []
    for lesson in lessons:
        try:
            saved.append(
                store.create(
                    user=user,
                    type="lesson",
                    text=lesson.text,
                    importance=lesson.importance,
                    source="reflection",
                    namespace=namespace,
                )
            )
        except ValueError:
            continue
    return saved


__all__ = [
    "apply_feedback",
    "apply_forgetting",
    "reflect_session",
    "save_lessons",
]
