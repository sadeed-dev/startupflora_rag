# Verification performed for this deliverable

Authoring environment: Python 3.12.14 on Linux, 2026-10-14.

- 42 automated checks passed.
- Ruff syntax/import checks passed.
- Ruff formatting checks passed.
- All production Python modules compiled successfully.
- Express and browser helper JavaScript passed syntax checks.
- Every production Python function has a purpose docstring.

The tests use real LangChain tools, prompt/message objects, QdrantVectorStore,
local Qdrant and SQLite. Actual LangChain/OpenAI request serialization is tested
with in-process HTTP mocks for structured output, embeddings and function calls.
No paid provider calls, live site crawl, Windows execution, Docker server run,
production deployment or final-answer quality benchmark was performed.

Important checked behaviors include user isolation, durable history, legacy
history import, idempotent retries, atomic failed turns, bounded tools, source ID
validation, verifier failures, model/namespace mismatches, cache invalidation,
conditional requests, sitemap discovery, robots restrictions, public DNS checks,
confirmed deleted sources and orphan-vector cleanup.

The included evaluate.py can measure retrieval on your reviewed cases using your
own provider account. Its scores are retrieval metrics, not a guarantee of answer
correctness. Review full answers and real content coverage before production.
