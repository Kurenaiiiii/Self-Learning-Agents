"""MemoryAgent: chat with injected memories, optional tools, post-turn learning."""

from __future__ import annotations

from dataclasses import dataclass, field

from selflearn.agent.prompts import build_system_prompt
from selflearn.agent.summary import RollingSummary
from selflearn.agent.tools import (
    ToolContext,
    apply_consolidation,
    execute_tool_call,
    memory_tool_defs,
)
from selflearn.llm.base import LLMProvider
from selflearn.llm.types import ChatMessage, ChatRequest, LLMError, ToolCall, Usage
from selflearn.memory.consolidation import consolidate_candidate
from selflearn.memory.extraction import extract_memories
from selflearn.memory.learning import apply_feedback, reflect_session, save_lessons
from selflearn.memory.retrieval import RankedMemory, Retriever, estimate_tokens
from selflearn.store.sqlite import MemoryStore


@dataclass
class ChatResult:
    """Agent reply plus what shaped it."""

    text: str
    memories_used: list[RankedMemory] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class MemoryAgent:
    """Per-user chat agent with long-term memory."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        store: MemoryStore,
        model: str,
        user: str,
        namespace: str = "default",
        memory_budget_tokens: int = 800,
        enable_tools: bool = True,
        learn: bool = True,
    ) -> None:
        self.llm = llm
        self.store = store
        self.model = model
        self.user = user
        self.namespace = namespace
        self.memory_budget_tokens = memory_budget_tokens
        self.enable_tools = enable_tools
        self.learn = learn
        self.history: list[ChatMessage] = []
        self.summary = RollingSummary()
        self.last_result: ChatResult | None = None

    def _system(self, query: str) -> tuple[str, list[RankedMemory]]:
        retriever = Retriever(self.store, user=self.user, namespace=self.namespace)
        context, used = retriever.build_context(
            query, top_k=5, max_tokens=self.memory_budget_tokens
        )
        self.summary.maybe_summarize(self.llm, model=self.model)
        return build_system_prompt(memory_context=context, summary=self.summary.summary), used

    def chat(self, message: str, *, temperature: float = 0.0) -> ChatResult:
        """One chat turn: retrieve, answer (tools optional), then learn."""
        system, used = self._system(message)
        self.history.append(ChatMessage(role="user", content=message))
        self.summary.add_turn("user", message)
        wire: list[ChatMessage] = [ChatMessage(role="system", content=system)] + self.history[-10:]
        tools = memory_tool_defs() if self.enable_tools else ()
        try:
            resp = self.llm.chat(
                ChatRequest(
                    messages=tuple(wire),
                    model=self.model,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=1024,
                )
            )
        except LLMError as e:
            return ChatResult(
                text=f"(error: {e})", memories_used=used, memory_ids=[m.memory.id for m in used]
            )
        tool_calls = list(resp.tool_calls)
        if tool_calls:
            ctx = ToolContext(
                store=self.store,
                user=self.user,
                namespace=self.namespace,
                llm=self.llm,
                model=self.model,
            )
            tool_msgs: list[ChatMessage] = []
            for call in tool_calls:
                try:
                    out = execute_tool_call(ctx, call)
                except Exception as e:  # noqa: BLE001 - tools never break chat
                    out = f"tool {call.name} failed: {e}"
                tool_msgs.append(
                    ChatMessage(role="tool", content=out, name=call.name, tool_call_id=call.id)
                )
            follow = wire + [
                ChatMessage(role="assistant", content=resp.text, tool_calls=tuple(tool_calls)),
                *tool_msgs,
            ]
            try:
                resp2 = self.llm.chat(
                    ChatRequest(
                        messages=tuple(follow),
                        model=self.model,
                        temperature=temperature,
                        max_tokens=1024,
                    )
                )
                text, usage = (
                    resp2.text,
                    Usage(
                        prompt_tokens=resp.usage.prompt_tokens + resp2.usage.prompt_tokens,
                        completion_tokens=resp.usage.completion_tokens
                        + resp2.usage.completion_tokens,
                    ),
                )
            except LLMError:
                text, usage = resp.text, resp.usage
        else:
            text, usage = resp.text, resp.usage
        self.history.append(ChatMessage(role="assistant", content=text))
        self.summary.add_turn("assistant", text)
        result = ChatResult(
            text=text,
            memories_used=used,
            memory_ids=[m.memory.id for m in used],
            tool_calls=tool_calls,
            usage=usage,
        )
        self.last_result = result
        if self.learn:
            self.learn_turn(message, text)
        return result

    def learn_turn(self, user_text: str, assistant_text: str) -> int:
        """Extract + consolidate one turn into the store. Returns memories written."""
        conversation = f"user: {user_text}\nassistant: {assistant_text}"
        cands = extract_memories(self.llm, model=self.model, conversation=conversation)
        if not cands:
            return 0
        ctx = ToolContext(store=self.store, user=self.user, namespace=self.namespace)
        written = 0
        for cand in cands[:5]:
            existing = self.store.list(
                user=self.user, namespace=self.namespace, include_superseded=True
            )
            result = consolidate_candidate(
                self.llm, model=self.model, candidate=cand, existing=existing
            )
            if result.action != "NOOP":
                apply_consolidation(ctx, candidate=cand, result=result)
                written += 1
        return written

    def feedback(self, direction: str) -> int:
        """Thumbs up/down on the last answer's memories."""
        if self.last_result is None or not self.last_result.memory_ids:
            return 0
        return apply_feedback(
            self.store,
            user=self.user,
            memory_ids=self.last_result.memory_ids,
            direction=direction,
            namespace=self.namespace,
        )

    def reflect(self, transcript: str) -> int:
        """Post-session reflection -> saved lesson memories."""
        lessons = reflect_session(self.llm, model=self.model, transcript=transcript)
        return len(
            save_lessons(self.store, user=self.user, lessons=lessons, namespace=self.namespace)
        )

    def explain_last(self, query: str) -> list[dict[str, object]]:
        """Why these memories were retrieved for the query."""
        return Retriever(self.store, user=self.user, namespace=self.namespace).explain(query)

    def token_estimate(self) -> int:
        """Rough prompt size of the next turn (for budget debugging)."""
        system, _ = self._system(self.history[-1].content if self.history else "")
        recent = "\n".join(m.content for m in self.history[-10:])
        return estimate_tokens(system + recent)


__all__ = ["ChatResult", "MemoryAgent"]
