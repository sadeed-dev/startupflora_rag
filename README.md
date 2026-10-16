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
or provider-held data. Model calls use `store=False`, but messages/evidence still
travel to OpenAI and remain subject to your provider/account policies. LangSmith
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
