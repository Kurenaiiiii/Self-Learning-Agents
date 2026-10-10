"""Agent layer: chat with memory, tools, rolling summary."""

from selflearn.agent.agent import ChatResult, MemoryAgent
from selflearn.agent.prompts import (
    BASE_INSTRUCTIONS,
    UNTRUSTED_MEMORY_GUARD,
    build_system_prompt,
)
from selflearn.agent.summary import RollingSummary
from selflearn.agent.tools import (
    ToolContext,
    apply_consolidation,
    execute_tool_call,
    memory_tool_defs,
)

__all__ = [
    "BASE_INSTRUCTIONS",
    "UNTRUSTED_MEMORY_GUARD",
    "ChatResult",
    "MemoryAgent",
    "RollingSummary",
    "ToolContext",
    "apply_consolidation",
    "build_system_prompt",
    "execute_tool_call",
    "memory_tool_defs",
]
