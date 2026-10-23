# Debug an unanswered question

Run from `startupflora_rag` using the same environment as the backend:

```powershell
python main.py doctor
python main.py ask "I need a loan" --user debug-loan --debug
```

The second command performs a real retrieval and generation and stores the chat
turn. It bypasses answer/search caches for that invocation; cached embeddings may
still be reused. It prints the response and request timings. With minimum-call mode,
`debug.answer_diagnostic` includes the decision, provider finish reason, limits,
retrieved chunk IDs, and IDs actually supplied after budgeting. It does not print
raw passage text or model output rejected by validation.

Use the returned conversation ID with `--conversation ID` to reproduce a follow-up.
Use a new invocation without `--conversation` to test a fresh question. Restart the
backend after code or `.env` changes; existing processes retain loaded settings.

For a website request, find its request ID in the browser Network panel's chat
response. Search the backend output for that ID and `rag_answer_diagnostic` or
`rag_request_timing`. Diagnostic logs remain available even if MySQL audit writes
fail. Existing saved requests replay their old result; send a new request ID to
exercise the updated code.

| Reason | What to inspect or change |
| --- | --- |
| `no_retrieved_chunks` | Check the active corpus, collection, and indexed loan pages. |
| `context_budget_exhausted` | Retrieved chunks could not fit `CONTEXT_TOKENS`; inspect chunk sizes and budget. |
| `model_input_budget_exhausted` | Evidence was removed to fit `MODEL_INPUT_TOKENS`; inspect history and serialized input size. |
| `generation_truncated` | Increase `MAX_OUTPUT_TOKENS` cautiously and repeat; the structured answer was cut off. |
| `invalid_tool_arguments` / `invalid_answer_schema` / `missing_or_unexpected_answer_tool` | Inspect provider tool/schema support; the output was not a usable Answer call. |
| `invalid_citations_or_format` | The generated answer failed citation/link/format validation. Keep citation checks; correct the generation path. |
| `model_insufficient_evidence` | Inspect the supplied passages for relevance and support; consider query wording, ranking, or missing content. |
| `model_refusal` / `generation_filtered` | The provider refused or filtered generation. |

For `invalid_citations_or_format`, inspect `answer_diagnostic.validation.errors`:
`unknown_source_ids`, `inline_source_ids_mismatch`, `model_generated_url`,
`empty_answer`, `answer_too_long`, or `answered_without_sources`. The same object
includes inline, declared, and available source IDs, without the rejected answer
text. A clarification without factual claims should use status `clarification`
and empty `source_ids`; status `answered` requires cited evidence.

Inspect exact passage text locally with a bound, read-only query for the chunk IDs:

```sql
SELECT id, title, source, text FROM chunks WHERE id IN ('CHUNK_ID');
```

Minimum-call failures also use the existing MySQL audit table when available:

```sql
SELECT created_at, request_id, question, search_query, failure_reason, retrieved_pages
FROM rag_unanswered_questions
ORDER BY created_at DESC
LIMIT 20;
```

Bare requests such as "I need a loan" now ask whether this is for a new business,
an existing business, or a personal loan without provider calls. Specific questions
about loan schemes, amounts, rates, or eligibility still require retrieval. For
minimum-call answers with valid inline citations, missing or inconsistent link
metadata is derived from those citations and logged as
`citation_metadata_reconciled: true`. Unknown IDs and uncited answers still fail;
the server never adds citations to uncited claims.

`retrieved_pages` lists retrieval candidates, while the runtime diagnostic identifies
which chunks survived the context/input budgets. Audit records contain user questions;
normal diagnostic logs contain IDs and decisions only. If the diagnostic reports
`accepted`, inspect the response status (`answered` or `clarification`) before
assuming the model provided factual guidance. Provider/network exceptions continue
through the existing API error handling rather than being disguised as no evidence.
