"""Rolling conversation summary for long sessions (stdlib fallback included)."""

from __future__ import annotations

from selflearn.llm.base import LLMProvider
from selflearn.llm.types import ChatMessage, ChatRequest, LLMError
from selflearn.memory.retrieval import estimate_tokens

SUMMARIZE_PROMPT = (
    "Summarize this conversation so far in 5 sentences or fewer. "
    "Keep user facts, preferences, decisions, and open questions. "
    "Reply with ONLY the summary."
)


class RollingSummary:
    """Keeps recent turns verbatim; summarizes the rest via LLM when over budget."""

    def __init__(self, *, max_tokens: int = 2000, keep_last_turns: int = 6) -> None:
        self.max_tokens = max_tokens
        self.keep_last_turns = keep_last_turns
        self._turns: list[tuple[str, str]] = []
        self.summary: str = ""

    def add_turn(self, role: str, text: str) -> None:
        """Append one turn."""
        self._turns.append((role, text[:4000]))

    def recent_text(self) -> str:
        """Verbatim recent turns."""
        return "\n".join(f"{r}: {t}" for r, t in self._turns[-self.keep_last_turns :])

    def maybe_summarize(self, llm: LLMProvider, *, model: str) -> str:
        """Summarize older turns when over budget. Falls back to truncation."""
        full = "\n".join(f"{r}: {t}" for r, t in self._turns)
        if estimate_tokens(full) <= self.max_tokens:
            self.summary = ""
            return ""
        older = self._turns[: -self.keep_last_turns] or self._turns
        older_text = "\n".join(f"{r}: {t}" for r, t in older)[:12000]
        try:
            resp = llm.chat(
                ChatRequest(
                    messages=(
                        ChatMessage(role="system", content=SUMMARIZE_PROMPT),
                        ChatMessage(role="user", content=older_text),
                    ),
                    model=model,
                    temperature=0.0,
                    max_tokens=400,
                )
            )
            self.summary = resp.text.strip()[:2000] or self._fallback(older_text)
        except LLMError:
            self.summary = self._fallback(older_text)
        return self.summary

    @staticmethod
    def _fallback(older_text: str) -> str:
        # ponytail: truncation beats no summary; newest-first keeps context relevant.
        tail = older_text[-1500:]
        return f"(truncated) ...{tail}"


__all__ = ["RollingSummary", "SUMMARIZE_PROMPT"]
