# Changelog

All notable changes to this project are documented here. Versioning follows
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-10

Initial public release.

### Added

- Provider-agnostic LLM layer: `openai_compatible`, `anthropic`, `ollama`,
  `fake` behind one interface (chat, tools, streaming, usage, retries with
  backoff/jitter + `Retry-After`, opt-in disk cache). Shared contract tests
  run against faithful mock servers for every provider.
- Pluggable embeddings: from-scratch BM25 + hashed char n-gram TF-IDF hybrid
  (no API), optional OpenAI-compatible and Ollama embeddings endpoints.
- SQLite memory store (stdlib): migrations, per-user/per-namespace isolation
  at the storage layer, version history audit trail, soft delete, supersede
  links, forgetting/pruning primitives.
- Extraction (validated JSON with repair retry + safe fallback) and
  consolidation (`ADD`/`UPDATE`/`SUPERSEDE`/`NOOP` with exact + near-duplicate
  guards).
- Hybrid retrieval (BM25 + vector + recency decay + importance) with token
  budgets and an explainable debug view.
- Memory agent: memory injection within budget, `memory_search`/`memory_save`
  tools, rolling summary, prompt-injection defense, feedback-driven
  importance, post-session reflection lessons, forgetting.
- Privacy: secret redaction at write time, `forget me` hard delete, per-user
  JSON export, configurable retention.
- Interfaces: Python API, `selflearn` CLI (`chat`, `memories`, `eval`,
  `doctor`), FastAPI REST API (Bearer auth, validation, rate limiting, CORS,
  health/readiness, OpenAPI docs).
- Benchmark: 40 synthetic scenarios across 7 categories, 6 strategies,
  deterministic rubric grading (+ optional LLM judge), tokens/latency/stale
  metrics, response + learned-store caches, throttling.
- Production polish: multi-stage Dockerfile (non-root, healthcheck),
  docker-compose, Makefile, pre-commit, Dependabot, CI with 85% coverage
  gate, mypy strict, ruff.
- Docs: README, `docs/ARCHITECTURE.md`, `docs/LEARNING.md`, `docs/API.md`,
  CONTRIBUTING, issue/PR templates.
