# StartupFlora RAG v2 — LangChain edition

A source-grounded chatbot backend with modular LangChain integrations, Qdrant,
BM25 keyword retrieval, strict custom tools and persistent per-user history.
Every production function includes a purpose docstring. The code is split by
responsibility so you can read, test and modify one part at a time.

This is a runnable starter to configure and evaluate. It has not been deployed
to startupflora.com, and it does not include your website content or API keys.

## Start here: Windows, local mode, no Docker

Extract into a **fresh folder**, for example `C:\projects\startupflora-rag-v2`.
Read [MIGRATION.md](MIGRATION.md) before reusing an earlier project's data or .env.

```powershell
cd C:\projects\startupflora-rag-v2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set your real `OPENAI_API_KEY`. Keep `QDRANT_MODE=local`. Terminal indexing/chat
needs no `BACKEND_API_KEY`, Docker daemon or running Qdrant server.
If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of
`python`; no activation is required with that interpreter path.

```powershell
python main.py doctor
python main.py ingest-web --max-pages 5
python main.py ask "What startup services do you offer?"
python main.py chat
```

`doctor` prints safe diagnostics: .env filename, active storage mode/path, package
versions and whether a key is configured. It does not reveal keys or contact the
provider. Environment variables override .env. Old `$env:QDRANT_PATH` settings in
the same PowerShell session still apply; inspect doctor output or start a fresh
PowerShell window after changing configuration.

To process the site's discovered pages rather than just the first five:

```powershell
python main.py ingest-web --max-pages 0
python main.py status
```

`0` means process discovered pages up to `MAX_CRAWL_URLS` (default 10,000), not an
unbounded crawl. Discovery uses robots sitemap hints, common sitemap paths, nested
XML/gzip sitemap indexes, and same-host internal links. An explicit sitemap is optional:

```powershell
python main.py ingest-web --sitemap https://startupflora.com/YOUR-ACTUAL-SITEMAP.xml --max-pages 0
```

Replace that placeholder with an actual sitemap. It is not a verified sitemap URL.

## How this version works

**Indexing:** sitemap/internal URLs -> robots and URL checks -> conditional HTML
fetch -> meaningful text -> LangChain recursive splitting -> cached OpenAI
embeddings -> LangChain QdrantVectorStore + active SQLite chunks/FTS5 index.

**Answering:** authenticated user -> owned conversation -> recent messages and
older summary -> standalone query -> mandatory custom search tool -> dense/BM25
rank fusion -> optional reranking -> bounded extra tools -> strict Answer schema
-> citation validation and optional evidence check -> atomic history write.

All indexed pages participate in retrieval. `FINAL_CHUNKS=5` selects five useful
chunks from that whole corpus; it does not restrict the database to five pages.
The model does not receive the whole website for every question.

## What improved

| Area | Implementation |
|---|---|
| LangChain integration | ChatOpenAI, OpenAIEmbeddings, QdrantVectorStore, recursive text splitter, prompts/messages and custom tools |
| Website coverage | Sitemap indexes + internal links + explicit crawl report with outcomes and pending URLs |
| Easier local testing | Explicit local mode by default; paths resolve relative to the project |
| Incremental refresh | HTTP ETag/Last-Modified requests, content hashes and deterministic chunk IDs |
| Embedding cost | Batch missing vectors, deduplicate exact text and reuse model-specific cache entries |
| Retrieval quality | Semantic + FTS5 BM25 search, reciprocal-rank fusion, model relevance selection |
| Repeated searches | Cache ranked chunk IDs by query/config/corpus revision; content changes invalidate results |
| Bounded context | Token-estimated evidence/history budgets and a pre-call input guard |
| Custom tools | LangChain @tool functions with strict schemas and a per-request evidence ledger |
| Conversation continuity | Full SQLite history, recent complete turns, rolling summary, resume UUID and conversation listing |
| Retry/concurrency behavior | Idempotent request IDs, atomic complete-turn writes and renewed DB leases |
| Reliability | Validated citations, optional fail-closed model evidence check and explicit error statuses |
| Maintenance | Cache cleanup, confirmed-deletion removal, explicit source removal and orphan-vector cleanup |
| Reviewability | Purpose docstrings, clear modules, tests and retrieval evaluation harness |

The integration uses the modular `langchain-core`, `langchain-openai`,
`langchain-qdrant` and `langchain-text-splitters` packages. The storage and auth
rules remain application code; a framework is not an authorization boundary.

## Full-site indexing and refresh behavior

Indexing writes `data/v2/crawl-report.json` and prints progress. Review:

- `indexed` / `unchanged` / `removed` / `failed` counts;
- every processed source and its outcome;
- sitemap warnings and failed URL exception types;
- `limited`, `remaining`, and the first 200 pending URLs.

A run with `failed > 0`, discovery warnings or `limited=true` is not proof of full
coverage. Pages absent from both links and sitemaps need explicit seeds. The
crawler reads static HTML; JavaScript-only content needs a server-rendered export
or an additional approved renderer. PDFs are imported explicitly. Private/account
pages, query-string variants and binary files are excluded from HTML discovery.
Configured hosts default to startupflora.com and www.startupflora.com.

Rerun `ingest-web` to refresh. A 304 response avoids reparsing/embedding; an unchanged
content hash avoids embedding even without HTTP validators. Failed fetches preserve
the previous active version. Confirmed HTTP 404/410 responses unpublish known
sources. A URL merely disappearing from a sitemap is not automatically deleted:
that could be a partial crawl, so review inventory and use explicit removal.

```powershell
python main.py remove-source https://startupflora.com/AN-OBSOLETE-REVIEWED-PATH
python main.py cleanup-cache --days 30
python main.py cleanup-vectors
```

`cleanup-vectors` removes vector IDs absent from the active SQLite corpus, and
shares a publication lease with indexing. Use a dedicated collection for this
application. It never cleans unrelated collections. In local mode run one process
at a time: finish CLI indexing/chat before starting the API, and stop the API
before another CLI process opens its Qdrant directory.

There is no installed scheduler. For production, run the refresh command from
an admin job runner on a cadence appropriate to your content. Review current
scheme documents before publishing; do not treat a fetch date as a scheme's
publication, effective date or proof of current eligibility/deadlines.

## PDF ingestion

```powershell
python main.py ingest-pdf .\docs\approved-guide.pdf --source-url https://YOUR-APPROVED-PUBLIC-SOURCE/guide.pdf
```

The URL is citation metadata, not a fetch operation. Import only public approved
content into this shared corpus. Limits: 25 MB and 300 pages. Scanned PDFs require
OCR first. Page labels are preserved in text; complex table/layout reconstruction
is not implemented by this basic PDF extractor.

## Models, budgets and caching

Defaults keep the original `gpt-4.1-mini` chat/utility model and
`text-embedding-3-small` with 1536 dimensions. Models are configurable and require
access on your own account. No latest-model selection is assumed.

| Setting | Default | Meaning |
|---|---:|---|
| CHUNK_SIZE / CHUNK_OVERLAP | 1600 / 200 | Character-based recursive splitting with overlap |
| RETRIEVAL_CANDIDATES | 16 | Candidate passages after rank fusion |
| FINAL_CHUNKS | 5 | Maximum passages selected by a search |
| ENABLE_RERANK | true | Model-based relevance selection over bounded complete passages |
| VERIFY_ANSWERS | true | Separate model check of answer claims against cited passages |
| SEARCH_CACHE_SECONDS | 900 | Cache rankings, not personalized generated answers |
| CONTEXT_TOKENS | 6000 | Distinct evidence budget per request |
| HISTORY_TOKENS | 2500 | Budget for recent complete history turns |
| MODEL_INPUT_TOKENS | 18000 | Guard serialized input before provider calls |
| RECENT_MESSAGES | 12 | Upper bound before history-budget trimming |
| MAX_TOOL_CALLS | 2 | Extra retrieval tools after the mandatory initial search |

Token budgeting uses cl100k_base, not the billing tokenizer of every possible
model. If that tokenizer's initial download fails, conservative UTF-8 byte counts
keep the app bounded. This may admit fewer passages/history messages; doctor
still works offline. Chunk boundaries remain character-based with LangChain.

A first substantive question usually invokes embedding, reranking, answer
formatting, and verification. Follow-ups may add query rewrite, and older histories
occasionally add summarization. Extra tools add calls. Exact greetings use no
model/embedding call. Repeated search results and embeddings are reused, but a
personalized final answer is not shared from a cross-user cache.

Keep verification/reranking on while measuring quality. Disable them only after
reviewing the accuracy/latency tradeoff on your actual questions. Reranking and
verification here are model judgments, not calibrated confidence or factual
guarantees. Similarity and RRF values are not confidence probabilities.

## Persistent chat and tools

Interactive `python main.py chat` keeps the same conversation until `/exit`.
It prints a UUID that can be resumed:

```powershell
python main.py chat --user local-demo --conversation YOUR-CONVERSATION-UUID
python main.py ask "What documents are needed for that?" --user local-demo --conversation YOUR-CONVERSATION-UUID
```

All turns stay in SQLite across restarts. Only recent complete turns plus a compact
user-context summary go to the model. A new conversation starts a fresh context;
conversation listing lets the frontend resume an older one explicitly. Summary
compaction can omit details, so full message history is the audit record.

`rag/tools.py` defines two custom Python functions with LangChain's @tool:

- `search_knowledge(query)` searches the approved corpus.
- `read_chunk_context(chunk_id)` expands an already exposed passage with neighbors.

The server forces initial retrieval. The model can use at most the configured
extra tool count, then must return a typed `Answer`. No tool executes shell/SQL,
fetches arbitrary URLs, books meetings, creates leads or reads private records.
Business action tools require your actual API, trusted caller identity, argument
validation, consent for writes and retry handling before they can be registered.

System prompts, hypothetical examples and relevance/memory instructions are in
`rag/prompts.py`. Old messages, summaries and document text are contextual data,
not instructions or trusted evidence. Source IDs are generated by the server;
the model cannot invent clickable URLs. The answer checker reduces unsupported
claims, but manual evaluations are still required.

## Website integration

Generate a server-only secret when you are ready for the API:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set its value in `BACKEND_API_KEY` on both the RAG server and your website backend.
Then start:

```powershell
python -m uvicorn rag.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Mount `examples/express-proxy.mjs` after your existing authentication middleware.
It reads `req.user.id`; adapt that field to your own login/session system. The
browser calls only your website backend (`examples/browser-client.js`). Neither
the OpenAI key nor the backend secret belongs in React, VITE_* or NEXT_PUBLIC_*.
For anonymous visitors, your backend must issue a signed server-generated session
identity. That session issuance is not included. Preserve existing CSRF protection
for cookie-authenticated routes.

| Method | Route | Purpose |
|---|---|---|
| POST | /conversations | Create a user-owned conversation |
| GET | /conversations?limit=20&before=CURSOR | List that user's prior chats |
| POST | /chat | conversation_id, request_id, message |
| GET | /conversations/{id}/messages?after=0&limit=50 | Read owned history and saved source metadata |
| DELETE | /conversations/{id} | Delete owned app conversation data |
| GET | /health | Process liveness only |

Your trusted backend supplies `Authorization: Bearer ...` and `X-User-ID`.
It must derive identity from authentication, never from browser JSON/headers.
A new message gets a new request UUID; retries of the exact same message reuse it.
Concurrent processing returns 409. A completed retry returns the saved reply.
Reusing a request UUID with different text returns 409; provider failures return
503 without a partial history turn. HTTP 429 is the per-user request limiter.

Render answer text safely and source links separately; use a sanitized Markdown
renderer if needed. Clear cached conversation data on logout/account changes.
Use private networking and TLS when the backend and RAG service are on different
machines. This version targets one API instance/worker with SQLite. Larger
multi-instance deployments should move history/coordination to PostgreSQL/Redis
and ingestion to a job queue with operational monitoring.

## Testing and evaluation

Install optional development tools:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python -m ruff check .
python -m ruff format --check .
```

The tests run real local Qdrant, SQLite and LangChain tools. Mock HTTP transports
exercise actual LangChain/OpenAI structured-output, embedding and function-call
wire formats. Tests cover indexing updates, ranking cache invalidation, sitemap
discovery, robots/DNS restrictions, 304 refreshes, citation checks, user isolation,
history restart/migration, retry behavior and atomic turn writes. No paid OpenAI
calls are made. The authoring runtime is Python 3.12 on Linux; Windows instructions
are provided, but Windows and Docker container execution were not performed here.

`requirements.txt` pins direct packages. `requirements.lock.txt` captures the
complete tested dependency set, including dev tools, for reproducibility.

Copy `examples/evaluation.example.jsonl` into your own case file and replace its
placeholder URLs with manually reviewed expected sources. Then:

```powershell
python evaluate.py --cases .\my-reviewed-cases.jsonl
```

This explicitly calls your configured provider and writes a retrieval report.
It measures source recall, unrelated-query abstention and elapsed time. It does
not measure final-answer factual accuracy. Warm caches affect timings.

Before production, review at least 50 realistic English/Hindi/Hinglish questions,
ambiguous follow-ups, wrong eligibility assumptions, unsupported prices/deadlines,
conflicting/outdated sources and malicious source text. Inspect both retrieved
passages and final answers. Measure actual provider usage, latency and quality;
no live-quality benchmark has been claimed for this project.

## Data handling and current limits

`data/v2` contains private chat history and cache data. Protect and back it up
with the Qdrant volume/directory; set a retention policy for your users. The app's
DELETE endpoint removes its own stored conversation data, not independent backups
or provider-held data. OpenAI chat calls use `store=False`. Messages/evidence go
to the configured chat provider (OpenAI or Google Gemini), and embedding inputs
go to OpenAI. Gemini requests follow Google's provider/account policies; the
OpenAI `store` parameter is not sent to Gemini. LangSmith
tracing is not enabled by the application; enabling it yourself sends trace data
to that service. Application error logs omit chat text, keys and provider bodies.

There is no streaming UI, live government-portal lookup, OCR pipeline, scheduled
refresh, payment/lead tool or production monitoring dashboard in this package.
Website pages that cannot be discovered or parsed need an explicit ingestion path.
No framework choice can guarantee coverage or eliminate hallucinations.

Official documentation used:
- [LangChain ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai)
- [LangChain Qdrant](https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant)
- [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools)
- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [Sitemap protocol](https://www.sitemaps.org/protocol.html)

## Response performance

The Python backend reads these settings from `startupflora_rag/.env` (restart
the server after environment changes):

```env
DB_BACKEND=mysql
MYSQL_POOL_SIZE=5
MYSQL_POOL_TIMEOUT=10
FINAL_CHUNKS=3
ANSWER_MAX_TOKENS=1000
VERIFY_ANSWERS=true
ENABLE_RERANK=true
```

Each API process maintains a bounded pool of exclusive MySQL connections.
Session settings are applied once per physical connection. Transactions commit
or roll back before the connection returns to the pool; stale sockets are
replaced before a transaction starts. Queries are never automatically retried
inside a transaction. Shutdown closes idle and returned connections. The pool
limit applies per process, so account for the worker count when sizing MySQL.

Ordinary answers target 60 to 100 words, with a 1000-token model output limit.
Three final passages are selected per search. Verification and reranking remain
enabled; complex questions can still require additional model calls.

Every `/chat` request emits a `rag_request_timing` JSON record through the
Uvicorn logger, including its request ID, total duration, stage durations, and
call counts. Stages include `retrieval`, `generation`, `verification`,
`persistence`, `database_acquire`, `database_query`, `database_commit`, and
`database_rollback` when applicable. Utility model calls are labelled
`model_ranking`, `model_query`, `model_memory`, or `model_grounding`.
Retrieval also records `retrieval_setup`, `embedding`, `vector_search`, and
`keyword_search` for work actually executed on a cache miss.
Parent stages include their child timings, so do not add all stages together.
Cache hits and greetings omit model stages they do not execute. Timings include
failed requests and exclude prompts, answers, credentials, and submitted contact
details. With the local hidden server launch, records are in
`data/rag-mysql.stderr.log`; foreground launches print them to the terminal.

## Gemini answers with existing OpenAI embeddings

The hybrid setup uses Google's documented
[OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai)
through the already-installed LangChain client. No Google SDK is required.
Configure the Python backend's `.env`:

```env
CHAT_PROVIDER=gemini
GEMINI_API_KEY=your-google-key
GEMINI_CHAT_MODEL=gemini-3.8-flash
GEMINI_UTILITY_MODEL=gemini-3.8-flash
MODEL_TIMEOUT_SECONDS=90
GEMINI_REASONING_EFFORT=low
OPENAI_API_KEY=your-existing-openai-key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
VERIFY_ANSWERS=true
```

Gemini handles answers, reranking, query rewriting, summaries, and grounding
verification. OpenAI handles document/query embeddings using the original cache,
model, dimensions, and Qdrant collection. Switching the chat provider requires no
re-embedding or SQL migration. Keep the existing Qdrant target/collection and
embedding settings. MySQL conversation and lead storage are unchanged.

The Gemini adapter preserves Google's opaque tool-call signatures across
follow-up calls and uses JSON schemas plus application-side validation. Search
and answer caches include the active provider/model so results are not confused
between providers. Both provider keys are required; authentication failures do
not silently fall back to another provider. Keys stay in the backend environment
and are excluded from diagnostic output and answer-cache fingerprints.

Restart the RAG process after changing `.env`. `python main.py doctor` reports
the effective chat/utility provider and models without displaying keys.
`server.env` has the same provider settings but an empty Gemini-key placeholder;
set that key separately in the server environment when deploying. `CHAT_MODEL`
and `UTILITY_MODEL` are used only when `CHAT_PROVIDER=openai`.

For a 20-second website response target, use
`gemini-3.1-flash-lite`, `FAST_MODE=true`,
`MODEL_TIMEOUT_SECONDS=10`, `RESPONSE_TIMEOUT_SECONDS=18`, and
`ANSWER_MAX_TOKENS=600`. Fast mode disables provider retries, model reranking,
query rewriting, memory summarization, additional retrieval tools and answer
repair. Follow-up retrieval includes recent user messages; generation still sees
conversation history. Existing citation checks remain mandatory.
`VERIFY_ANSWERS` is respected separately: enabling it adds a verification call.
Answer caching works with verification enabled or disabled. Cache keys include
the verification setting, so an unchecked answer cannot satisfy a verified request.

The current local configuration and backend deployment template allow 25 seconds
per model call and a 35-second API processing budget. The website uses
`RAG_TIMEOUT_MS=40000` (40 seconds), configurable in `startupflora-next/.env`
up to 55 seconds within the routes' 60-second runtime budget. For the shorter
profile above, also set `RAG_TIMEOUT_MS=19000`.
The API returns HTTP 504 after its processing budget. A slow provider can therefore
produce a timeout message rather than an answer. These are wait limits, not a
guarantee of successful answers within 20 seconds. Workers finish their current
blocking operation after HTTP timeout, then reject late answers before saving.
Restart both backend and website after these changes.

Local Qdrant connections use IPv4 (`127.0.0.1`) when the configured host is
`localhost`, avoiding repeated Windows IPv6 connection delays. The stored
namespace still uses the configured URL, so this needs no data migration.

`ANSWER_CACHE_SECONDS=-1` (the default and local setting) keeps accepted answers
without time-based expiry. Use `86400` for one-day reuse or `0` to disable caching.
Only successful answers with accepted citations are cached. Self-contained
questions, including "give me count of total success stories", can reuse the same user's saved answer after unrelated turns,
in another conversation, or after a server restart. Case and extra whitespace
do not affect matching. Questions that depend on earlier context, such as
"What are its fees?" or "DPIIT for my company?", retain conversation handling.
Cache hits make no retrieval, embedding, generation, rewrite, or verification
provider calls; chat turns are still saved. Inspect `answer_cache: "hit"` and
empty `token_usage` in request timings to confirm reuse. CLI `--debug` deliberately
bypasses answer/search caches and should not be used to test cache hits.
Corpus changes, model/settings changes and code changes trigger a fresh answer.
The existing `verified_answer_cache` table name is retained for compatibility;
no database migration is needed. Entries are no longer deleted automatically
after seven days, so storage grows with distinct cached questions and revisions.
Bypassed questions now include `answer_cache_reason` in the timing log: `disabled`
or `context_dependent`. Cache eligibility no longer requires a named service.

Each provider call logs `rag_token_usage` with provider, model, operation,
request ID and reported input/output/total tokens. These entries also appear in
`rag_request_timing.token_usage`, alongside `answer_cache` (`hit`, `miss`,
`bypass` or `request_replay`). A cache hit uses no generation or embedding API
calls. Embedding counts come from the OpenAI response before LangChain discards
its usage metadata. Missing counts and failed calls use `usage_reported=false`
and null counts; they do not assert zero cost. Logs contain no prompts, answers
or keys, and counts are not a currency/credit estimate. Provider retries can
include usage that the final response does not report.

### Timing and token tracking table

Authenticated, validated `POST /chat` attempts are stored in one table,
`rag_request_timings`, in the configured MySQL database (or SQLite database).
The table and indexes are created automatically at startup; restart the backend
after updating the code. No extra database or AI calls are needed to view it.

Each HTTP attempt has a unique `attempt_id`; retries can share a `request_id`
and still have separate rows. Columns include `conversation_id`, UTC
`created_at`, `status_code`, `outcome`, `answer_cache`, `total_ms`,
`provider_calls`, `input_tokens`, `output_tokens`, `total_tokens`,
`unreported_usage_calls`, and `usage_complete`. `stages_ms`, `calls` and
`token_usage` store the detailed JSON breakdown from the terminal log.
Questions, answers, user identities and credentials are not copied into this table.

Writes run after the HTTP response and do not count toward `total_ms`.
If a request times out while its worker is still running, `usage_complete=0`;
when the worker finishes, the same row is updated with late usage and
`usage_complete=1`, while the HTTP status remains 504. Token totals sum known
counts only: inspect `unreported_usage_calls` before treating totals as complete.
Tracking-write failures are logged without changing the chat response.
Requests rejected before the chat handler (authentication/input validation) are
not included. Historical terminal logs are not imported automatically.

Recent requests:

```sql
SELECT created_at, request_id, status_code, outcome, answer_cache,
       ROUND(total_ms / 1000, 2) AS seconds, provider_calls,
       input_tokens, output_tokens, total_tokens,
       unreported_usage_calls, usage_complete
FROM rag_request_timings
ORDER BY created_at DESC, attempt_id DESC
LIMIT 100;
```

Daily usage (UTC):

```sql
SELECT DATE(created_at) AS day, COUNT(*) AS attempts,
       SUM(CASE WHEN answer_cache = 'hit' THEN 1 ELSE 0 END) AS cache_hits,
       SUM(provider_calls) AS provider_calls,
       SUM(total_tokens) AS reported_tokens,
       SUM(unreported_usage_calls) AS calls_without_usage,
       ROUND(AVG(total_ms) / 1000, 2) AS avg_seconds
FROM rag_request_timings
GROUP BY DATE(created_at)
ORDER BY day DESC;
```
