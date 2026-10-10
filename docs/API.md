# REST API Reference

Base URL `http://localhost:8000`. Interactive docs at `/docs` (OpenAPI).
Every example below is a real call against a local server (FakeLLM backend,
so replies read `fake reply`; memory behavior is identical with real models).

Auth: when `SELFLEARN_API_TOKEN` is set, every `/v1/*` route needs
`Authorization: Bearer <token>` (constant-time compare, 401 otherwise).
Without it the server runs open for local dev and logs a warning.

## Health and readiness

```bash
curl localhost:8000/health
# {"status":"ok","version":"0.1.0"}

curl localhost:8000/ready
# {"status":"ready"}   # 503 when the database is unreachable
```

## Create a memory

```bash
curl -X POST localhost:8000/v1/memories \
  -H 'Content-Type: application/json' \
  -d '{"user":"demo","text":"drinks oat milk latte every morning",
       "type":"preference","importance":0.8}'
# {"id":"5a09855824984748","user":"demo","namespace":"default",
#  "type":"preference","text":"drinks oat milk latte every morning",
#  "importance":0.8,...,"version":1,"deleted":false,"superseded_by":""}
```

Bad `type` → `400 {"detail": "bad type 'nope'"}`. Secret patterns in `text`
are redacted before storage.

## Chat

```bash
curl -X POST localhost:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"user":"demo","message":"what do I drink?"}'
# {"reply":"...","memory_ids":["5a09855824984748"],
#  "usage":{"prompt_tokens":10,"completion_tokens":5}}
```

Optional fields: `namespace` (default `"default"`), `temperature` (0–2),
`learn` (default `true` — post-turn extraction into the store).
Each request is stateless; memory persists in SQLite across calls.

## List memories

```bash
curl "localhost:8000/v1/memories?user=demo&namespace=default&limit=50"
# {"memories":[{...}]}
```

`type` filters to `fact|preference|episode|lesson`. Scoped to the user:
`?user=bob` never returns alice's rows.

## Search with explainable scores

```bash
curl "localhost:8000/v1/memories/search?user=demo&q=latte&top_k=2"
# {"results":[{"id":"5a09855824984748",
#   "text":"drinks oat milk latte every morning","type":"preference",
#   "score":0.7751,
#   "components":{"bm25":1.0,"vector":0.3502,"recency":1.0,"importance":0.8},
#   "est_tokens":8}]}
```

## Feedback (thumbs up/down)

```bash
curl -X POST localhost:8000/v1/feedback \
  -H 'Content-Type: application/json' \
  -d '{"user":"demo","memory_ids":["5a09855824984748"],"direction":"up"}'
# {"changed":1}
```

`up` adds +0.15 importance, `down` −0.2 (clamped to [0, 1]).
Anything else → `400 {"detail": "direction must be up|down"}`.

## Export and forget-me

```bash
curl "localhost:8000/v1/export?user=demo"
# [{...full memory rows, all namespaces, including deleted...}]

curl -X DELETE "localhost:8000/v1/memories/5a09855824984748?user=demo"
# {"deleted":true}          # soft delete; hidden from retrieval

curl -X DELETE localhost:8000/v1/users/demo
# {"removed":4}             # hard delete: memories + history rows
```

## Errors and limits

| Status | Meaning                                              |
| ------ | ---------------------------------------------------- |
| 400    | bad `type`/`direction`, empty text                   |
| 401    | missing/invalid Bearer token (when token configured) |
| 422    | pydantic validation (empty user, oversize fields)   |
| 429    | per-IP rate limit exceeded (120 req/min default)     |
| 500    | chat failure, body `{"detail": "chat failed: ..."}` |
| 503    | `/ready` when the database is unreachable            |

Rate limiting is an in-memory per-IP sliding window; CORS is off unless
`SELFLEARN_CORS_ORIGINS` lists origins. Validation errors never leak stack
traces.
