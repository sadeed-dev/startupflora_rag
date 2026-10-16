"""Application-owned LangChain tools. No arbitrary execution, SQL or URL fetching."""

from langchain_core.tools import tool

from .schemas import ContextInput, SearchInput


def build_tools(retriever, store, evidence):
    """Create tools scoped to ONE request so source IDs never leak between concurrent users."""

    @tool(args_schema=SearchInput)
    def search_knowledge(query: str) -> dict:
        """Search the full approved corpus using semantic search, BM25 and reranking.

        Purpose: find missing evidence or investigate a second topic. Use a
        standalone question. Read-only; no live website or private-record access.
        """
        if not 1 <= len(query.strip()) <= 2000:
            return {"error": "query must contain 1-2000 characters"}
        return evidence.tool_result(retriever.search(query.strip()))

    @tool(args_schema=ContextInput)
    def read_chunk_context(chunk_id: str) -> dict:
        """Read an already retrieved chunk and its adjacent sections from the same source.

        Purpose: clarify context without another semantic search or embedding
        call. Only a chunk ID exposed in this request is permitted.
        """
        if chunk_id not in evidence.by_chunk:
            return {"error": "Only chunk IDs retrieved in this request are allowed"}
        return evidence.tool_result(store.neighbors(chunk_id))

    return {t.name: t for t in (search_knowledge, read_chunk_context)}
