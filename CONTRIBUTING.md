# Contributing to Self-Learning Agents

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Workflow

1. Pick a small issue. Keep diffs small and conventional commits:
   `feat:`, `fix:`, `test:`, `docs:`, `chore:`.
2. Before pushing, run the full gate:

```bash
make lint && make type && make test
```

3. Coverage must stay at or above 85% (`make cov`). New behavior needs tests,
   including error paths and isolation checks where relevant.
4. mypy strict must be clean; ruff must be clean.

## Rules

- Core library stays standard-library only. New dependencies belong in
  `server` or `dev` extras with justification.
- Never hard-code model names; models come from config.
- Never commit secrets. Tests use `FakeLLM` and mock servers; live tests are
  opt-in (`SELFLEARN_LIVE=1`) and never run in CI.
- Benchmark numbers in docs must come from real runs, labeled with provider
  type, model, date, runs, and temperature.
- No TODOs in merged code; no dead code; no silently swallowed exceptions.

## Pull Requests

Fill in `.github/PULL_REQUEST_TEMPLATE.md`. Small PRs merge faster.
