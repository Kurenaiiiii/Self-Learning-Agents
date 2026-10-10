"""Extra coverage: error paths, compat delegation, caches, doctor, CLI chat."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from selflearn.agent import MemoryAgent
from selflearn.bench import BenchConfig
from selflearn.bench.harness import _load_or_learn_store, build_report, run_benchmark
from selflearn.bench.scenarios import SCENARIOS
from selflearn.cli import main
from selflearn.config import Config, load_config, redacted
from selflearn.doctor import run_doctor
from selflearn.embeddings import OllamaEmbedder, OpenAICompatEmbedder
from selflearn.llm.cache import DiskCache
from selflearn.llm.fake import FakeLLM
from selflearn.llm.ollama import OllamaProvider
from selflearn.llm.types import ServerError
from selflearn.server import create_app
from selflearn.store import MemoryStore
from tests.mock_servers import MockServer, OpenAIMockHandler


def _cfg(tmp_path: object, **kw: object) -> Config:
    from pathlib import Path

    base: dict[str, object] = {
        "provider": "fake",
        "model": "fake",
        "db_path": str(Path(str(tmp_path)) / "x.db"),
    }
    base.update(kw)
    return Config(**base)  # type: ignore[arg-type]


def test_doctor_reports() -> None:
    bad = run_doctor(Config(provider="openai_compatible", base_url="", model="m"))
    assert "ERROR" in bad
    fake = run_doctor(Config(provider="fake", model="fake"))
    assert "config: OK" in fake and "tiny ping" in fake
    down = run_doctor(Config(provider="ollama", model="m", base_url="http://127.0.0.1:1"))
    assert "ERROR" in down
    assert redacted("secret", keep=4).endswith("...cret)")


def test_load_config_file(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    monkeypatch.chdir(Path(str(tmp_path)))
    for var in (
        "SELFLEARN_PROVIDER",
        "SELFLEARN_MODEL",
        "SELFLEARN_BASE_URL",
        "SELFLEARN_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    Path("c.json").write_text(json.dumps({"provider": "ollama", "model": "llama3"}))
    cfg = load_config("c.json")
    assert cfg.provider == "ollama" and cfg.model == "llama3"
    assert load_config("missing.json").provider == "openai_compatible"


def test_ollama_v1_delegates_to_compat() -> None:
    srv = MockServer(OpenAIMockHandler)
    try:
        p = OllamaProvider(base_url=srv.base_url + "/v1", api_key="k", max_retries=1)
        from selflearn.llm.types import ChatMessage, ChatRequest

        resp = p.chat(
            ChatRequest(messages=(ChatMessage(role="user", content="hi"),), model="mock-model")
        )
        assert resp.text == "hello from mock"
        assert (
            list(
                p.chat_stream(
                    ChatRequest(
                        messages=(ChatMessage(role="user", content="hi"),), model="mock-model"
                    )
                )
            )[-1].done
            is True
        )
    finally:
        srv.stop()


def test_embedder_server_error() -> None:
    srv = MockServer(OpenAIMockHandler)
    try:
        emb = OpenAICompatEmbedder(base_url=srv.base_url + "/v1", api_key="k", model="m")
        srv.behaviors = ["ok"]  # embeddings endpoint ignores behavior queue; force via bad input
        with pytest.raises(Exception):  # noqa: BLE001, PT011 - any typed error
            OllamaEmbedder(base_url="http://127.0.0.1:1", model="m").embed(["a"])
        assert emb.embed(["a"])[0]
    finally:
        srv.stop()


def test_disk_cache_corruption(tmp_path: object) -> None:
    from pathlib import Path

    cache = DiskCache(str(Path(str(tmp_path)) / "c"))
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    assert cache.get("missing") is None
    Path(str(Path(str(tmp_path)) / "c") + "/k.json").write_text("{broken")
    assert cache.get("k") is None


def test_learn_cache_reuse(tmp_path: object) -> None:
    from pathlib import Path

    learn_dir = str(Path(str(tmp_path)) / "learn")
    bench = BenchConfig(
        strategies=("hybrid",), runs=1, limit=1, throttle_s=0, learn_cache_dir=learn_dir
    )
    scn = SCENARIOS[0]
    first = _load_or_learn_store(
        FakeLLM(['{"memories": []}']), model="m", scn=scn, run=1, bench=bench, user="u"
    )
    first.close()
    # Second load hits the sqlite file, no LLM needed (empty queue would fail).
    second = _load_or_learn_store(FakeLLM(), model="m", scn=scn, run=1, bench=bench, user="u")
    assert len(second.list(user="u", namespace=scn.id)) == 0
    second.close()


def test_benchmark_error_row_and_judge() -> None:
    cfg = Config(provider="fake", model="fake")
    bench = BenchConfig(strategies=("no_memory",), runs=1, limit=1, throttle_s=0, judge=True)
    report = run_benchmark(cfg, bench=bench, llm=FakeLLM([ServerError("x", provider="fake")]))
    assert report["rows"]
    assert report["meta"]["judge_mode"] == "llm-judge"
    empty = build_report(
        [], provider="fake", model="fake", date="2026-01-01", options=BenchConfig()
    )
    assert empty["summary"] == {}


def test_server_open_mode_chat_failure(tmp_path: object) -> None:
    app = create_app(_cfg(tmp_path), llm=FakeLLM([ServerError("down", provider="fake")]))
    with TestClient(app) as c:
        r = c.post("/v1/chat", json={"user": "u", "message": "hi"})
        assert r.status_code == 200 and r.json()["reply"].startswith("(error:")
        app.state.store.close()  # break the db -> readiness must report 503, not raise
        assert c.get("/ready").status_code == 503


def test_cli_chat_single_and_memories(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pathlib import Path

    monkeypatch.chdir(Path(str(tmp_path)))
    monkeypatch.setenv("SELFLEARN_PROVIDER", "fake")
    monkeypatch.setenv("SELFLEARN_MODEL", "fake")
    monkeypatch.setenv("SELFLEARN_DB_PATH", "cli.db")
    assert main(["chat", "--user", "u", "--message", "hello", "--no-tools"]) == 0
    assert main(["memories", "search", "--user", "u", "hello"]) == 0
    assert main(["memories", "export", "--user", "u"]) == 0
    assert main(["chat", "--user", "u", "--message", "x", "--no-learn"]) == 0
    capsys.readouterr()


def test_agent_helpers(tmp_path: object) -> None:
    store = MemoryStore(":memory:")
    try:
        agent = MemoryAgent(
            llm=FakeLLM(["hi", "[]"]), store=store, model="m", user="u", enable_tools=False
        )
        agent.chat("hello")
        assert agent.token_estimate() > 0
        assert isinstance(agent.explain_last("hello"), list)
        assert agent.summary.recent_text() != ""
    finally:
        store.close()


def test_fake_helpers() -> None:
    from selflearn.llm.types import ToolCall

    f = FakeLLM()
    assert f.last_request() is None and f.last_user_text() == ""
    assert "fake" in f.as_dict()["provider"]
    resp = f.tool_response([ToolCall(id="1", name="t", arguments={})], text="doing")
    assert resp.tool_calls[0].name == "t"
    f.queue(resp)
    assert (
        f.chat(
            __import__("selflearn.llm.types", fromlist=["ChatRequest"]).ChatRequest(
                messages=(), model="m"
            )
        ).text
        == "doing"
    )
