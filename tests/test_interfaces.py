"""CLI + REST API tests (auth, validation, isolation, rate limit)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from selflearn.cli import build_parser, main
from selflearn.config import Config
from selflearn.llm.fake import FakeLLM
from selflearn.server import create_app
from selflearn.store import MemoryStore


def _cfg(tmp_path: object, **kw: object) -> Config:
    from pathlib import Path

    base: dict[str, object] = {
        "provider": "fake",
        "model": "m",
        "db_path": str(Path(str(tmp_path)) / "t.db"),
    }
    base.update(kw)
    return Config(**base)  # type: ignore[arg-type]


@pytest.fixture()
def client(tmp_path: object) -> TestClient:
    app = create_app(_cfg(tmp_path), llm=FakeLLM(["hello there", "[]"]))
    with TestClient(app) as c:
        yield c


def test_health_and_ready(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["status"] == "ready"


def test_chat_end_to_end(client: TestClient) -> None:
    r = client.post("/v1/chat", json={"user": "u", "message": "hi"})
    assert r.status_code == 200
    assert r.json()["reply"] == "hello there"


def test_memory_crud_and_search(client: TestClient) -> None:
    r = client.post("/v1/memories", json={"user": "u", "text": "likes oat latte"})
    assert r.status_code == 201
    mid = r.json()["id"]
    assert client.get("/v1/memories", params={"user": "u"}).json()["memories"][0]["id"] == mid
    found = client.get("/v1/memories/search", params={"user": "u", "q": "latte"}).json()
    assert found["results"][0]["id"] == mid
    assert client.delete(f"/v1/memories/{mid}", params={"user": "u"}).json() == {"deleted": True}
    assert client.get("/v1/memories", params={"user": "u"}).json() == {"memories": []}


def test_per_user_isolation(client: TestClient) -> None:
    client.post("/v1/memories", json={"user": "alice", "text": "alice secret"})
    assert client.get("/v1/memories", params={"user": "bob"}).json() == {"memories": []}
    assert client.get("/v1/export", params={"user": "bob"}).json() == []
    assert client.delete("/v1/users/bob").json() == {"removed": 0}
    assert len(client.get("/v1/export", params={"user": "alice"}).json()) == 1


def test_validation_errors(client: TestClient) -> None:
    assert client.post("/v1/chat", json={"user": "", "message": "hi"}).status_code == 422
    assert (
        client.post("/v1/memories", json={"user": "u", "text": "x", "type": "nope"}).status_code
        == 400
    )
    assert (
        client.post(
            "/v1/feedback", json={"user": "u", "memory_ids": [], "direction": "x"}
        ).status_code
        == 400
    )


def test_feedback_and_export_delete(client: TestClient) -> None:
    mid = client.post("/v1/memories", json={"user": "u", "text": "likes chess"}).json()["id"]
    assert client.post(
        "/v1/feedback", json={"user": "u", "memory_ids": [mid], "direction": "up"}
    ).json() == {"changed": 1}
    assert client.delete("/v1/users/u").json()["removed"] > 0


def test_bearer_auth_enforced(tmp_path: object) -> None:
    app = create_app(_cfg(tmp_path, api_token="secret123"), llm=FakeLLM())
    with TestClient(app) as c:
        assert c.get("/v1/memories", params={"user": "u"}).status_code == 401
        assert (
            c.get(
                "/v1/memories", params={"user": "u"}, headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        ok = c.get(
            "/v1/memories", params={"user": "u"}, headers={"Authorization": "Bearer secret123"}
        )
        assert ok.status_code == 200


def test_rate_limit(tmp_path: object) -> None:
    app = create_app(_cfg(tmp_path), llm=FakeLLM(), rate_limit_per_min=3)
    with TestClient(app) as c:
        for _ in range(3):
            assert c.get("/health").status_code == 200
        assert c.get("/health").status_code == 429


def test_cli_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    parser = build_parser()
    assert parser.parse_args(["chat", "--user", "u"]).cmd == "chat"
    assert parser.parse_args(["memories", "list", "--user", "u"]).mcmd == "list"
    assert parser.parse_args(["eval", "--runs", "2"]).runs == 2
    assert main([]) == 2
    out = capsys.readouterr().out
    assert "0.1.0" in out


def test_cli_memories_list(
    tmp_path: object, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    db = str(Path(str(tmp_path)) / "cli.db")
    monkeypatch.setenv("SELFLEARN_DB_PATH", db)
    monkeypatch.setenv("SELFLEARN_PROVIDER", "fake")
    store = MemoryStore(db)
    store.create(user="u", type="fact", text="cli note")
    store.close()
    assert main(["memories", "list", "--user", "u"]) == 0
    assert "cli note" in capsys.readouterr().out
