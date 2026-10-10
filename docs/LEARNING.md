# Learning Guide

## Glossary

- **Context window**: how much text a model sees at once (prompt + reply).
  Anything outside it might as well not exist. Our token budget exists to
  respect it.
- **Long-term memory**: facts that survive across sessions, stored in SQLite
  here — not in the model's weights. The model reads them from the prompt.
- **Embedding**: text turned into a list of numbers (a vector) so similarity
  becomes arithmetic. Our default builds vectors from character n-grams with
  no API.
- **BM25**: a keyword ranking formula. Rare matching words score high; long
  documents are penalized. Great for exact terms like names and places.
- **Cosine similarity**: the angle between two vectors, 0 (unrelated) to 1
  (identical direction). Used for our n-gram vectors.
- **Consolidation**: the per-candidate decision (`ADD`/`UPDATE`/`SUPERSEDE`/
  `NOOP`) that keeps the store coherent instead of a pile of duplicates.
- **Supersede**: link an outdated memory to its replacement. The old row
  stays in history (audit) but leaves retrieval.
- **Prompt injection**: smuggling instructions into data the model reads
  (here: a hostile stored memory). Defended by labeling memories untrusted
  and fencing them as data.
- **Token budget**: a cap on how much memory text enters the prompt
  (default 800 estimated tokens). Forces ranking to matter.
- **Rate limiting / backoff**: when an API says "slow down" (HTTP 429), wait
  — honoring its `Retry-After` header — then retry with growing, jittered
  delays instead of hammering it.
- **Idempotency**: doing an operation twice has the same effect as once.
  Our migrations and cache writes are safe to rerun.
- **Multi-tenancy**: one deployment serving many users. Every store query
  carries `(user, namespace)` so tenants never see each other's memories —
  proven by isolation tests.
- **Provider abstraction**: one `LLMProvider` interface; OpenAI-compatible,
  Anthropic, and Ollama hide their wire differences behind it. No model
  names in code; config picks the model.

## Interview Q&A (about this codebase)

1. **Why does the agent need a memory layer at all?**
   The model forgets everything between sessions. Choices are stuffing
   history (blows the window/budget), summaries (lose detail), or a curated
   store — we built the third, with retrieval scoped to a token budget.

2. **Walk me through one memory's lifecycle.**
   Chat turn → extraction LLM call → validated candidate → dedup guards →
   LLM consolidation decision → `ADD`/`UPDATE`/`SUPERSEDE` write with history
   row → ranked at query time → injected into the prompt → feedback adjusts
   importance → eventually decays or is pruned.

3. **How do you handle "I moved from Berlin to Lisbon"?**
   Consolidation sees the contradiction, creates the Lisbon memory, and
   `SUPERSEDE`-links Berlin. Berlin stays in history; retrieval excludes it;
   recency also favors the newer row.

4. **Why both BM25 and vectors?**
   BM25 wins on exact rare terms (names, places); char n-gram vectors win on
   paraphrase and morphology. Either alone has blind spots; the hybrid plus
   recency/importance beats both in our benchmark.

5. **How do you dedup without an LLM call?**
   Exact match (normalized) and pairwise char n-gram TF cosine ≥ 0.85 —
   IDF-free, because IDF is degenerate on tiny corpora (unseen grams get max
   weight and crush similarity; found during testing).

6. **What stops a stored "ignore your instructions" from working?**
   System-prompt guard, fenced memory block labeled untrusted, labeled tool
   results, and tests that plant hostile memories and assert normal replies.

7. **How do retries work?**
   Retryable errors (429, 5xx, timeouts, network) retry up to `max_retries`
   with `base * 2^attempt + jitter`, capped; 429s honor `Retry-After`. Auth
   and validation errors never retry.

8. **Why urllib instead of httpx or vendor SDKs?**
   Zero dependencies in core: smaller supply chain, no SDK drift across three
   providers, and the wire formats are simple enough to own.

9. **How is token usage tracked?**
   Each provider normalizes its usage shape (OpenAI `usage`, Anthropic
   `input/output_tokens`, Ollama `prompt_eval/eval_count`, stream tail
   events) into one `Usage` type, summed across tool-loop turns.

10. **What data reaches the LLM provider?**
    Chat messages, retrieval queries, extraction/reflection transcripts, and
    injected memory text. Secrets are redacted at storage; live conversation
    necessarily goes to the provider. Documented in `SECURITY.md`.

11. **How do you prove tenant isolation?**
    `user_id` + `namespace` appear in every `WHERE` clause; tests assert one
    user cannot read, update, delete, link, export, or list another's rows —
    including via history and forget-me paths.

12. **What does the benchmark actually prove?**
    That the memory layer answers recall/update questions better than stuffing
    history or summarizing, with fewer tokens, on 40 synthetic scenarios × 6
    strategies × 3 runs — plus where it fails (reported honestly).

13. **Why deterministic grading instead of an LLM judge?**
    Reproducibility and cost. Keyword rubrics are strict (may undercount good
    paraphrases) but stable; the LLM judge exists as a separate labeled mode.

14. **How do you keep benchmark costs sane?**
    Disk cache for the learning phase (identical requests hash-hit),
    shared learned stores across strategies, throttling, and always-live
    final answers so spread stays real.

15. **What happens on malformed model output?**
    One repair retry, then safe fallback (extraction → `[]`, consolidation →
    `ADD`, reflection → `[]`, judge → `None`). The chat path never raises
    because of model output.

16. **How does streaming work across providers?**
    Each provider parses its own stream (SSE deltas, Anthropic events,
    Ollama NDJSON) into identical `ChatChunk` deltas plus a final usage
    chunk. Retries cover connection setup, not mid-stream resumes.

17. **Why SQLite?**
    Serverless, stdlib, transactional, enough for per-user memory scale;
    migrations are versioned and idempotent. The 10k+ path (persisted
    vectors) is documented, not built — YAGNI.

18. **How does forgetting work?**
    `last_accessed_at` refreshes on retrieval (`touch`); long-unaccessed
    memories decay in importance; old + unimportant rows are pruned by
    policy. Feedback pushes the other way.

19. **How is the REST API secured?**
    Bearer token from env (constant-time compare), per-user scoping on every
    endpoint, pydantic validation with 400s (never stack traces), per-IP
    rate limiting with 429s, configurable CORS, health/readiness probes.

20. **What would you build next?**
    Persisted vectors past 10k memories, per-strategy weight tuning from
    benchmark data, streaming chat endpoint, and multi-user benchmark
    scenarios measuring isolation under load.
