"""REST API (server extra): chat, memory CRUD/search, feedback, export, delete.

Run: uvicorn selflearn.server:app --host 0.0.0.0 --port 8000
Auth: Bearer token from SELFLEARN_API_TOKEN when set (open dev mode otherwise).
"""

from __future__ import annotations

import hmac
import os
import time
from typing import Any

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover - server extra missing
    raise ImportError("server extra needed: pip install 'selflearn[server]'") from e

from selflearn import __version__
from selflearn.agent import MemoryAgent
from selflearn.config import Config, load_config
from selflearn.llm.base import LLMProvider
from selflearn.llm.factory import provider_from_config
from selflearn.llm.fake import FakeLLM
from selflearn.log import get_logger, new_request_id
from selflearn.memory import Retriever, apply_feedback, export_user_json, forget_me, redact_text
from selflearn.store import VALID_TYPES, MemoryStore

log = get_logger("selflearn.server")

RATE_LIMIT_PER_MIN = 120


# -- models ----------------------------------------------------------------------


class ChatIn(BaseModel):
    """Chat request body."""

    user: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=12000)
    namespace: str = Field(default="default", max_length=128)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    learn: bool = True


class MemoryIn(BaseModel):
    """Create-memory request body."""

    user: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=2000)
    type: str = Field(default="fact")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    namespace: str = Field(default="default", max_length=128)
    source: str = Field(default="api", max_length=64)


class FeedbackIn(BaseModel):
    """Feedback request body."""

    user: str = Field(min_length=1, max_length=128)
    memory_ids: list[str] = Field(max_length=50)
    direction: str = "up"
    namespace: str = Field(default="default", max_length=128)


# -- app factory -------------------------------------------------------------------


def create_app(
    cfg: Config | None = None,
    *,
    llm: LLMProvider | None = None,
    rate_limit_per_min: int = RATE_LIMIT_PER_MIN,
) -> FastAPI:
    """Build the FastAPI app. Provider overridable for tests."""
    config = cfg or load_config()
    provider = llm or _build_provider(config)
    store = MemoryStore(config.db_path)
    app = FastAPI(title="Self-Learning Agents", version=__version__)
    app.state.cfg = config
    app.state.llm = provider
    app.state.store = store
    app.state.hits = {}

    origins = [
        o.strip() for o in os.environ.get("SELFLEARN_CORS_ORIGINS", "").split(",") if o.strip()
    ]
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
        )

    @app.middleware("http")
    async def _middleware(request: Request, call_next: Any) -> Any:
        new_request_id()
        if not _check_rate_limit(app, request, rate_limit_per_min):
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
        started = time.monotonic()
        response = await call_next(request)
        log.info(
            "request",
            extra={
                "provider": config.provider,
                "model": config.model,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
            },
        )
        return response

    def _auth(authorization: str | None = Header(default=None)) -> None:
        token = config.api_token
        if not token:
            return  # dev mode: open, logged at startup
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        if not hmac.compare_digest(authorization[7:], token):
            raise HTTPException(status_code=401, detail="invalid token")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        try:
            store.list(user="__readiness__", limit=1)
        except Exception as e:  # noqa: BLE001 - readiness must report, not raise
            raise HTTPException(status_code=503, detail=f"db not ready: {e}") from e
        return {"status": "ready"}

    @app.post("/v1/chat")
    def chat(body: ChatIn, _ok: None = Depends(_auth)) -> dict[str, Any]:
        agent = MemoryAgent(
            llm=provider,
            store=store,
            model=config.model or "fake",
            user=body.user,
            namespace=body.namespace,
            learn=body.learn,
        )
        try:
            result = agent.chat(body.message, temperature=body.temperature)
        except Exception as e:  # noqa: BLE001 - sane 500s, never stack traces
            raise HTTPException(status_code=500, detail=f"chat failed: {e}") from e
        return {
            "reply": result.text,
            "memory_ids": result.memory_ids,
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
            },
        }

    @app.get("/v1/memories")
    def list_memories(
        user: str = Query(min_length=1),
        namespace: str = "default",
        type: str | None = None,  # noqa: A002 - query param name is the API
        limit: int = Query(default=50, ge=1, le=200),
        _ok: None = Depends(_auth),
    ) -> dict[str, Any]:
        if type is not None and type not in VALID_TYPES:
            raise HTTPException(status_code=400, detail=f"bad type {type!r}")
        mems = store.list(user=user, namespace=namespace, type=type, limit=limit)
        return {"memories": [m.to_dict() for m in mems]}

    @app.post("/v1/memories", status_code=201)
    def create_memory(body: MemoryIn, _ok: None = Depends(_auth)) -> dict[str, Any]:
        if body.type not in VALID_TYPES:
            raise HTTPException(status_code=400, detail=f"bad type {body.type!r}")
        try:
            mem = store.create(
                user=body.user,
                type=body.type,
                text=redact_text(body.text.strip()),
                importance=body.importance,
                source=body.source,
                namespace=body.namespace,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return mem.to_dict()

    @app.get("/v1/memories/search")
    def search_memories(
        user: str = Query(min_length=1),
        q: str = Query(min_length=1, max_length=2000),
        namespace: str = "default",
        top_k: int = Query(default=5, ge=1, le=20),
        _ok: None = Depends(_auth),
    ) -> dict[str, Any]:
        return {"results": Retriever(store, user=user, namespace=namespace).explain(q, top_k=top_k)}

    @app.delete("/v1/memories/{memory_id}")
    def delete_memory(
        memory_id: str,
        user: str = Query(min_length=1),
        namespace: str = "default",
        _ok: None = Depends(_auth),
    ) -> dict[str, bool]:
        return {"deleted": store.soft_delete(user=user, memory_id=memory_id, namespace=namespace)}

    @app.post("/v1/feedback")
    def feedback(body: FeedbackIn, _ok: None = Depends(_auth)) -> dict[str, int]:
        if body.direction not in ("up", "down"):
            raise HTTPException(status_code=400, detail="direction must be up|down")
        changed = apply_feedback(
            store,
            user=body.user,
            memory_ids=body.memory_ids,
            direction=body.direction,
            namespace=body.namespace,
        )
        return {"changed": changed}

    @app.get("/v1/export")
    def export(user: str = Query(min_length=1), _ok: None = Depends(_auth)) -> list[Any]:
        import json as _json

        payload: list[Any] = _json.loads(export_user_json(store, user=user))
        return payload

    @app.delete("/v1/users/{user}")
    def delete_user(user: str, _ok: None = Depends(_auth)) -> dict[str, int]:
        return {"removed": forget_me(store, user=user)}

    return app


def _build_provider(config: Config) -> LLMProvider:
    try:
        return provider_from_config(config)
    except ValueError:
        log.warning("no valid LLM config; server chat falls back to FakeLLM")
        return FakeLLM(default_text="(server has no model configured)")


def _check_rate_limit(app: FastAPI, request: Request, per_min: int) -> bool:
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - 60.0
    hits = app.state.hits.get(ip, [])
    hits = [t for t in hits if t > window_start]
    if len(hits) >= per_min:
        app.state.hits[ip] = hits
        return False
    hits.append(now)
    app.state.hits[ip] = hits
    return True


app = create_app(load_config())

__all__ = ["app", "create_app"]
