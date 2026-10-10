# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Open a GitHub Security Advisory for this repository, or email the maintainer
listed in `pyproject.toml`. Include steps to reproduce, impact, and any
suggested fix. Expect an acknowledgement within 7 days.

## Secrets and Data Handling

- Never commit API keys, tokens, or `.env` contents. `.env` is git-ignored
  from the first commit; CI runs with no secrets.
- The library redacts common secret patterns (`sk-`, `sk-ant-`, `gsk_`,
  `Bearer` tokens, `api_key=...`, AWS keys) before storing memories.
- Stored memories are treated as untrusted data and are never executed as
  instructions (see `docs/ARCHITECTURE.md`).
- `forget me` (`DELETE /v1/users/{user}`) hard-deletes all of a user's
  memories and history rows.
- Data sent to the LLM provider: chat messages, retrieval queries, and
  extraction/reflection transcripts. Memory text is injected into prompts
  within a token budget. See `docs/ARCHITECTURE.md` for details.

## Dependency and Supply Chain

- Core library: standard library only (no supply-chain surface).
- Server extra: FastAPI/uvicorn/pydantic, pinned weekly by Dependabot.
- Builds are reproducible from source (`python -m build`).
