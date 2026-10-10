"""Privacy tests: redaction, forget-me, export, retention, write-path redaction."""

from __future__ import annotations

import json

import pytest

from selflearn.agent import MemoryAgent, ToolContext
from selflearn.agent.tools import execute_tool_call
from selflearn.llm.fake import FakeLLM
from selflearn.llm.types import ToolCall
from selflearn.memory import (
    enforce_retention,
    export_user_json,
    forget_me,
    redact_text,
)
from selflearn.store import MemoryStore


@pytest.fixture()
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_redact_key_patterns() -> None:
    cases = [
        "key is sk-abcDEF123456789",
        "anthropic sk-ant-xyz1234567890 here",
        "token gsk_abc1234567890xyz here",
        "Authorization: Bearer tokengoeshere123",
        "api_key=supersecretvalue9",
        "aws AKIAIOSFODNN7EXAMPLE key",
    ]
    for raw in cases:
        out = redact_text(raw)
        assert "[REDACTED]" in out, raw
        assert "sk-" not in out.replace("[REDACTED]", "") or "sk-" not in raw.split()[0]


def test_redact_leaves_normal_text_alone() -> None:
    text = "user drinks oat latte every morning in Lisbon"
    assert redact_text(text) == text
    assert redact_text("") == ""


def test_forget_me_hard_deletes_everything(store: MemoryStore) -> None:
    mem = store.create(user="u", type="fact", text="remember me")
    store.update(user="u", memory_id=mem.id, text="remember me v2")
    assert forget_me(store, user="u") > 0
    assert store.list(user="u", include_deleted=True) == []
    assert store.history(user="u", memory_id=mem.id) == []
    assert store.export_user(user="u") == []


def test_export_json_roundtrip(store: MemoryStore) -> None:
    store.create(user="u", type="preference", text="likes chess")
    data = json.loads(export_user_json(store, user="u"))
    assert len(data) == 1 and data[0]["text"] == "likes chess"
    assert json.loads(export_user_json(store, user="ghost")) == []


def test_retention_prunes_old_junk(store: MemoryStore) -> None:
    old = store.create(user="u", type="fact", text="stale", importance=0.05)
    store._conn.execute(
        "UPDATE memories SET updated_at=? WHERE id=?",
        ("2000-01-01T00:00:00+00:00", old.id),
    )
    store._conn.commit()
    report = enforce_retention(store, user="u")
    assert report["pruned"] == 1


def test_agent_learn_path_redacts_secrets(store: MemoryStore) -> None:
    llm = FakeLLM(
        [
            "ok",
            '{"memories": [{"type": "fact", '
            '"text": "deploy key is sk-abcDEF123456789", "importance": 0.6}]}',
        ]
    )
    agent = MemoryAgent(llm=llm, store=store, model="m", user="u", enable_tools=False)
    agent.chat("my deploy key is sk-abcDEF123456789")
    stored = store.list(user="u")
    assert len(stored) == 1
    assert "[REDACTED]" in stored[0].text
    assert "sk-abcDEF123456789" not in stored[0].text


def test_tool_save_path_redacts_secrets(store: MemoryStore) -> None:
    out = execute_tool_call(
        ToolContext(store=store, user="u"),
        ToolCall(id="c", name="memory_save", arguments={"text": "token gsk_abc1234567890xyz here"}),
    )
    assert "ADD" in out
    assert "gsk_abc1234567890xyz" not in store.list(user="u")[0].text
