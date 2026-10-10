# Self-Learning Agents

Production-grade long-term memory for LLM agents: extraction, consolidation,
hybrid retrieval, feedback and forgetting, with a CLI, a REST API, privacy
controls, and a reproducible benchmark. Works with OpenAI-compatible APIs,
Anthropic, and local Ollama models.

> **Credit / independence note.** Inspired by Dave Ebbelaar's video *How to
> Build Self-Learning AI Agents* and the memory-layer approach popularized by
> Mem0. **No code was copied** — every line in this repository was written
> from scratch for this project.

## Contents

- [What it is](#what-it-is)
- [Demo (real model, real recall + correction)](#demo-real-model-real-recall--correction)
- [Features](#features)
- [Providers](#providers)
- [Quick start](#quick-start)
- [Usage (Python, CLI, REST)](#usage-python-cli-rest)
- [Architecture](#architecture)
- [Benchmark](#benchmark)
- [Security and privacy](#security-and-privacy)
- [Configuration](#configuration)
- [Docker deployment](#docker-deployment)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Project structure](#project-structure)
- [Documentation](#documentation)
- [License](#license)

## What it is

LLMs forget everything between sessions. This library gives an agent durable,
per-user memory: after each turn it extracts atomic facts into SQLite,
curates them (`ADD` / `UPDATE` / `SUPERSEDE` / `NOOP`), and retrieves the
relevant ones into the next prompt within a token budget. Corrections win
over stale facts; feedback tunes importance; forgetting prunes the rest.
One interface, three providers, stdlib-only core.

## Demo (real model, real recall + correction)

Real transcript, model `openai/gpt-oss-20b` through an OpenAI-compatible
endpoint, 2026-10-10, temperature 0 (lightly formatted; full text verbatim):

```
you> Hi! I just moved from Berlin to Lisbon. I drink oat milk latte every morning.
ai>  Welcome to Lisbon! ... How are you finding the city so far? And that
     oat-milk latte — do you have a favorite café ...?

you> I play chess on weekends, by the way.
ai>  That's a great way to spend a weekend! Do you have a favorite opening ...?

you> Small correction: I actually moved from Munich, not Berlin. Berlin was years ago.
ai>  Got it — Munich to Lisbon, not Berlin. Thanks for the update! ...

you> What city did I move from, and what do I drink in the morning?
ai>  You moved from **Munich**, and you start your mornings with an **oat-milk latte**.
```

Stored memories after the session (abridged):

```
- [fact] User moved from Munich to Lisbon
- [fact] User moved from Berlin to Lisbon        <- stale, supersede link missed
- [preference] drinks oat milk latte every morning
- [fact] plays chess on weekends
```

Honest note: the final answer is correct, but consolidation created the
Munich memory without `SUPERSEDE`-linking the Berlin one — retrieval ranking
carried the answer. Contradiction linking works in the benchmark's explicit
update scenarios but is not perfect on casual phrasing; see
[Benchmark](#benchmark) and [Limitations](#limitations).

## Features

- **Provider-agnostic LLM layer** — OpenAI-compatible, Anthropic (native
  Messages API), Ollama (native API), plus `FakeLLM`. Normalized chat, tool
  calls, streaming, token usage, and errors; retries with backoff + jitter
  and `Retry-After`; opt-in disk cache.
- **Pluggable embeddings** — from-scratch BM25 + hashed char n-gram TF-IDF
  hybrid by default (no API); optional OpenAI-compatible / Ollama endpoints.
- **SQLite store** — migrations, per-user + per-namespace isolation enforced
  in SQL, version history audit trail, soft delete, supersede links.
- **Extraction + consolidation** — validated JSON with repair retry and safe
  fallback; deterministic dedup guards before any LLM decision.
- **Hybrid retrieval** — BM25 + vector + recency decay + importance, token
  budget, explainable scores.
- **Agent** — memory injection, `memory_search` / `memory_save` tools, rolling
  summary, prompt-injection defense, feedback, reflection lessons, forgetting.
- **Privacy** — secret redaction at write time, `forget me`, JSON export,
  retention policy.
- **Interfaces** — Python API, `selflearn` CLI, FastAPI REST (auth,
  validation, rate limiting, CORS, health/readiness, OpenAPI).
- **Benchmark** — 40 scenarios × 6 strategies × 3 runs, deterministic grading,
  labeled reports.

## Providers

| Provider             | Wire format                              | Verification status |
| -------------------- | ---------------------------------------- | ------------------- |
| `openai_compatible`  | OpenAI Chat Completions + `GET /models`  | **Verified live** by the author (chat + model list, 2026-10-10) |
| `anthropic`          | Native Messages API (own headers, system field, tool-use blocks, SSE events) | Verified against a faithful mock server + shared contract tests; **not** tested against the real API (author has no key). Opt-in live tests included — run `SELFLEARN_LIVE_ANTHROPIC=1 pytest tests/test_live.py` with your own key. |
| `ollama`             | Native `/api/chat` + `/api/tags` (or `/v1` OpenAI-compatible endpoint) | Verified against a faithful mock server + shared contract tests; **not** tested against a real server. Opt-in: `SELFLEARN_LIVE_OLLAMA=1 pytest tests/test_live.py`. |

No model names appear in code — the model always comes from config.
Anthropic has no embeddings API and no list-models endpoint; both are
documented, not worked around.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env   # then fill in your provider (placeholders only in repo)
```

**OpenAI-compatible endpoint** (works with OpenAI and any compatible
service or local server — set your own URL, key, model):

```bash
export SELFLEARN_PROVIDER=openai_compatible
export SELFLEARN_BASE_URL=https://YOUR_ENDPOINT/v1
export SELFLEARN_API_KEY=your-key
export SELFLEARN_MODEL=your-model
selflearn doctor
selflearn chat --user alice
```

**Anthropic (native API):**

```bash
export SELFLEARN_PROVIDER=anthropic
export SELFLEARN_API_KEY=your-key        # or ANTHROPIC_API_KEY
export SELFLEARN_MODEL=your-model
selflearn doctor
selflearn chat --user alice
```

**Local Ollama:**

```bash
export SELFLEARN_PROVIDER=ollama
export SELFLEARN_BASE_URL=http://localhost:11434
export SELFLEARN_MODEL=your-local-model
selflearn doctor
selflearn chat --user alice
```

## Usage (Python, CLI, REST)

Python:

```python
from selflearn.agent import MemoryAgent
from selflearn.config import load_config
from selflearn.llm.factory import provider_from_config
from selflearn.store import MemoryStore

cfg = load_config()
agent = MemoryAgent(llm=provider_from_config(cfg), store=MemoryStore(cfg.db_path),
                    model=cfg.model, user="alice")
print(agent.chat("I moved from Berlin to Lisbon.").text)
print(agent.chat("Where do I live?").text)
agent.feedback("up")   # thumbs up tunes the memories just used
```

CLI:

```bash
selflearn chat --user alice --message "Where do I live?"
selflearn memories list --user alice
selflearn memories search --user alice "Lisbon"
selflearn memories export --user alice --out alice.json
selflearn memories forget --user alice --yes
selflearn eval --strategies hybrid --runs 3 --out results.json
selflearn doctor
```

REST (see [docs/API.md](docs/API.md) for the full reference with real outputs):

```bash
uvicorn selflearn.server:app --port 8000
curl -X POST localhost:8000/v1/chat -H 'Content-Type: application/json' \
  -d '{"user":"alice","message":"Where do I live?"}'
```

## Architecture

```
 user turn ──► EXTRACTION (LLM→JSON, repair+fallback) ──► CONSOLIDATION ──► STORE (SQLite+history)
   ADD / UPDATE / SUPERSEDE / NOOP ──► dedup guards first, LLM decides ties
 question ──► RETRIEVAL (0.4 BM25 + 0.3 vector + 0.15 recency + 0.15 importance)
          ──► system prompt += memories (token budget, fenced as untrusted data)
          ──► tools: memory_search / memory_save ──► answer ──► learn, feedback, reflect, forget
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (worked example,
scoring explained simply, contradictions, injection defense, trade-offs).

## Benchmark

<!-- BENCH:TABLES -->

Honest commentary (read before the numbers):

- Memory strategies beat no-memory on recall/update by a wide margin; the gap
  narrows on preference questions answerable from the question alone.
- `full_history` is competitive on short scenarios but costs far more tokens
  per query and degrades with distractors; it cannot scale past the window.
- The main failure mode is consolidation missing casual contradictions (see
  demo note): the stale row survives and retrieval must outrank it. Explicit
  "Update: X" phrasing links correctly far more often.
- Keyword grading is strict and can undercount good paraphrases; absolute
  accuracy understates real usefulness. Relative ordering is the signal.
- Synthetic data is cooperative and crisp; expect lower absolutes on messy
  real conversations.

## Security and privacy

- Secrets (`sk-`, `sk-ant-`, `gsk_`, `Bearer`, `api_key=…`, AWS keys) are
  redacted before memories are stored; logs/traces redact the same shapes.
- Stored memories are untrusted data: fenced in prompts, labeled in tool
  results, never obeyed; hostile-memory tests enforce this.
- `forget me` hard-deletes all rows + history; export returns everything.
- Sent to the provider: chat messages, retrieval queries,
  extraction/reflection transcripts, injected memory text. See
  [SECURITY.md](SECURITY.md).

## Configuration

| Variable | Meaning | Default |
| -------- | ------- | ------- |
| `SELFLEARN_PROVIDER` | `openai_compatible` \| `anthropic` \| `ollama` \| `fake` | `openai_compatible` |
| `SELFLEARN_BASE_URL` | endpoint base URL | provider default |
| `SELFLEARN_API_KEY` | key (falls back to `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) | empty |
| `SELFLEARN_MODEL` | model name (required except `fake`) | empty |
| `SELFLEARN_DB_PATH` | SQLite path | `./data/selflearn.db` |
| `SELFLEARN_API_TOKEN` | REST Bearer token (unset = open dev mode) | empty |
| `SELFLEARN_CORS_ORIGINS` | comma-separated CORS origins | empty (off) |
| `SELFLEARN_LOG_LEVEL` | log level | `INFO` |

Plus an optional JSON config file (`--config`). Validation fails fast with
clear errors. `.env` is git-ignored from the first commit; `.env.example`
holds placeholders only.

## Docker deployment

```bash
docker build -t selflearn:latest .
docker run -p 8000:8000 --env-file .env selflearn:latest
# or: docker compose up --build
```

Non-root user, healthcheck on `/health`, data volume at `/app/data`.

## Limitations

- Contradiction linking misses casual phrasing (see demo); retrieval usually
  still ranks the newer memory first, but stale rows accumulate.
- Token counting is a chars/4 estimate, not a real tokenizer.
- Retrieval scans all of a user's memories per query (fine to ~10k; persisted
  vectors would come next).
- Keyword grading undercounts paraphrases; LLM-judge mode is separate.
- Anthropic and Ollama providers are mock-verified only (no author key/server).

## Roadmap

Persisted vectors past 10k memories · weight tuning from benchmark data ·
streaming chat endpoint · multi-user benchmark scenarios · optional real
tokenizer · richer redaction (custom patterns).

## Project structure

```
src/selflearn/   llm/ (providers)  embeddings/  store/ (sqlite)
                 memory/ (extract, consolidate, retrieve, learn, privacy)
                 agent/ (agent, tools, summary, prompts)
                 bench/ (scenarios, harness, grading)
                 cli.py  server.py  doctor.py  config.py  log.py
tests/           contract tests + mocks, store/agent/privacy/API/bench tests
docs/            ARCHITECTURE.md  LEARNING.md  API.md
benchmarks/      results/ (git-ignored JSON)  .cache/ (git-ignored)
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how it works, plain English
- [docs/LEARNING.md](docs/LEARNING.md) — glossary + 20 interview Q&As
- [docs/API.md](docs/API.md) — REST reference with real call outputs
- [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md),
  [CHANGELOG.md](CHANGELOG.md)

## License

MIT — see [LICENSE](LICENSE).
