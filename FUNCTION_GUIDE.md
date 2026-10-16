# Where to read and customize the code

Every production function has a docstring stating its purpose. Start with these
entry points rather than reading the whole project at once.

| File / function | Purpose |
|---|---|
| main.py / main | Route terminal commands and close storage on exit |
| main.py / doctor | Inspect configuration without displaying secrets or calling the provider |
| rag/config.py / load_settings | Read .env/environment values and validate paths/budgets |
| rag/runtime.py / runtime | Construct one shared set of application services |
| rag/crawler.py / Website.sitemap_urls | Discover nested sitemap pages with URL and resource limits |
| rag/crawler.py / Website.fetch | Apply robots, DNS, pacing, redirect and bounded retry rules |
| rag/crawler.py / Website.page | Fetch conditionally and extract meaningful HTML/internal links |
| rag/text.py / split_text | Use LangChain's recursive character splitter with overlap |
| rag/ingest.py / Ingestor.document | Skip unchanged sources and publish versioned chunks through LangChain |
| rag/ingest.py / Ingestor.crawl | Crawl approved sources and record actual coverage/outcomes |
| rag/ingest.py / Ingestor.remove_source | Unpublish obsolete content safely before vector cleanup |
| rag/llm.py / CachedEmbeddings | Implement LangChain embeddings with model-specific deduplicated caching |
| rag/llm.py / Models.structured | Parse strict typed utility outputs and enforce model input budgets |
| rag/retrieval.py / Retriever.search | Combine semantic/keyword results, rerank and cache rankings |
| rag/retrieval.py / Evidence | Track only the passages exposed during the current request |
| rag/tools.py / build_tools | Create read-only custom LangChain tools scoped to one request |
| rag/agent.py / Agent.memory | Combine recent complete turns with a rolling older-context summary |
| rag/agent.py / Agent.generate | Run mandatory retrieval and bounded strict tool calls |
| rag/agent.py / Agent.validate_answer | Check source IDs/links and optionally verify factual support |
| rag/agent.py / Agent.respond | Enforce ownership/idempotency and persist a complete turn atomically |
| rag/store.py / Store | Own active chunks, caches, conversations, rate limits and leases |
| rag/api.py | Expose authenticated private endpoints to your website backend |
| evaluate.py / evaluate | Measure retrieval against manually reviewed expected sources |

## Common customizations

- Change tone and source rules in rag/prompts.py.
- Change models, budgets, crawl hosts and storage mode in .env.
- Add a tool in rag/tools.py, then test its arguments, ownership checks and retry
  behavior. Read-only retrieval tools can return evidence; write tools need your
  real API and explicit consent handling before execution.
- Keep server-owned source metadata authoritative. Do not replace source IDs with
  free-form model-generated links.
- Keep history persistence separate from model context size. Adding more history
  to every prompt increases cost and can distract retrieval.
- Private customer records require a separate authenticated tool or tenant-filtered
  store; never ingest them into the shared public website corpus.
