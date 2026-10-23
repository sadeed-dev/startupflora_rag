# Minimum-call chat mode

The backend `.env` and deployment template `server.env` now enable `MINIMUM_CALLS=true`.
This is an opt-in mode; installations without that setting retain the previous orchestration.

For a substantive question, the backend resolves simple follow-up references in code,
checks the existing caches, runs one retrieval using Qdrant similarity + MySQL keyword
ranking + RRF, and makes one answer-generation invocation. An uncached search needs
at most one OpenAI query embedding invocation. Cached search results or embeddings
avoid that paid call. Exact greetings, unresolved references, empty evidence and
completed request replays can finish without generation. The currently configured
answer provider is Gemini.

There are no model calls for rewriting, summaries, ranking, routing, language detection,
verification or output repair in this mode. No model-selected search rounds are exposed.
Both chat and embedding SDK clients have **zero automatic network retries** in this mode;
a failed network attempt therefore cannot multiply the call count. A completed retry
with the same request ID returns the stored result. A new request after a failure is
a separate attempt and can incur new calls.

## Changed/new files

| File | Change |
|---|---|
| `rag/minimum.py` (new) | Single retrieval/generation path; question-focused grounding instructions; 30–60 word target; conservative follow-up resolution; deduplication; whole-pair/history and evidence trimming; safe invalid/truncated output handling. |
| `rag/agent.py` | Selects the new mode; skips summaries, query rewriting and model verification; preserves existing ownership, rate limits, leases, saved retries and response fields. Includes the new module in answer-cache fingerprints. |
| `rag/config.py` | Adds `MINIMUM_CALLS` and `MAX_OUTPUT_TOKENS`; permits the requested 8,000-token input budget while retaining consistency checks. |
| `rag/llm.py` | Disables automatic retries in minimum-call mode; applies its output limit; guards serialized messages and schemas plus estimated protocol overhead. |
| `rag/retrieval.py` | Prevents model reranking in minimum-call mode, including when legacy reranking flags are enabled. Existing embedding cache, vector search, keyword ranking and RRF remain in use. |
| `rag/text.py` | History fitting accepts only complete user/assistant pairs. |
| `rag/timing.py` | Adds model-call purpose and automatic retry limit to privacy-safe usage logs; request totals remain separate from each call's provider-reported usage. |
| `.env`, `server.env` | Enable minimum-call mode and apply the settings below. Existing credentials, embedding dimensions and collection names are retained. |
| `tests/test_minimum_calls.py` (new) | Offline regression tests for counts, caches, follow-ups, history, input/output budgets, provider wire parameters, retry limits, citations and access protections. |
| `MINIMUM_CALLS.md` (new) | Configuration, restart commands, validation and limitations. |

## Environment settings

```dotenv
MINIMUM_CALLS=true
MODEL_INPUT_TOKENS=8000
CONTEXT_TOKENS=2500
HISTORY_TOKENS=1000
FINAL_CHUNKS=3
RETRIEVAL_CANDIDATES=8
RECENT_MESSAGES=6
ENABLE_RERANK=false
MAX_TOOL_CALLS=0
VERIFY_ANSWERS=false
MAX_OUTPUT_TOKENS=400
```

`MAX_OUTPUT_TOKENS` includes structured-answer overhead, not just displayed prose.
The installed OpenAI adapter serializes it as `max_completion_tokens`; the existing
Gemini compatibility adapter sends `max_tokens`. `ANSWER_MAX_TOKENS` remains the
legacy-mode setting. The mode overrides legacy utility flags for its call guarantees.
History and evidence have separate limits within the full model-input budget.
The original question and essential instructions are retained; older history pairs
and lower-priority evidence are removed before the final guard. JSON is never sliced.
Local counts are estimates; provider-reported usage is recorded separately when available.

## Run/restart (PowerShell)

Stop the existing backend with **Ctrl+C**, then run:

```powershell
Set-Location C:\projects\startupflora\startupflora_rag
.\.venv\Scripts\python.exe -m uvicorn rag.api:app --host 127.0.0.1 --port 8000 --workers 1
```

For development with Python reload:

```powershell
Set-Location C:\projects\startupflora\startupflora_rag
.\.venv\Scripts\python.exe -m uvicorn rag.api:app --host 127.0.0.1 --port 8000 --workers 1 --reload --reload-dir rag
```

Restart after changing `.env`; Python-only reload does not watch `.env`. `server.env`
is the deployment template; the local loader reads `.env`, with process environment
variables taking precedence. Apply these settings to the deployed service environment
and restart that service when deploying. The frontend has no changes requiring restart.
Its existing contact form and submission route use application/database code and make
zero AI calls.

## Validation and limitations

Run from `startupflora_rag`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check rag/minimum.py rag/agent.py rag/config.py rag/llm.py rag/retrieval.py rag/text.py rag/timing.py tests/test_minimum_calls.py
```

Validation: **116 tests passed**; Ruff passed for all changed Python files. Effective
local settings were loaded and checked without contacting providers. Tests use offline
models, fake HTTP, temporary SQLite storage and in-memory Qdrant. They cover one fresh
embedding/generation, cached reuse, no utility calls, saved request replay, complete
history, safe output rejection, duplicate evidence, budgets, ownership, concurrency,
rate limits and OpenAI/Gemini output parameters with no HTTP retry after a mock failure.

No paid APIs or live MySQL/Qdrant service were used for validation. No production history,
stored vectors or collection definitions were changed, and no ingestion was run.
Actual answer quality and response speed need representative deployment evaluation;
there is no claim of live verification or a guaranteed response time. Citation checks
validate IDs, matching labels and URLs, but cannot prove every factual claim without
semantic verification. Unsupported or truncated output produces a short safe response,
never an automatic repair call. Follow-up resolution deliberately asks for clarification
when recent user text does not identify one recognized service/topic reliably.
