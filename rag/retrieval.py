"""LangChain Qdrant retrieval plus keyword BM25, rank fusion and optional reranking."""

import json
import logging
from urllib.parse import urlsplit

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from .prompts import RERANK_PROMPT
from .schemas import Ranking
from .text import digest, rrf, token_count
from .timing import stage, timed

log = logging.getLogger(__name__)


def qdrant_connection_url(url):
    """Avoid Windows localhost IPv6 connection delays without changing corpus identity."""
    parts = urlsplit(url)
    if parts.hostname == "localhost" and parts.username is None:
        authority = "127.0.0.1" + (f":{parts.port}" if parts.port is not None else "")
        return parts._replace(netloc=authority).geturl()
    return url


class Retriever:
    """Search the complete active corpus; candidate limits do not limit indexed website pages."""

    def __init__(self, settings, store, models, qdrant=None):
        """Select explicit local/server storage so a stopped Docker engine cannot affect local tests."""
        self.s, self.store, self.models = settings, store, models
        self.q = qdrant or (
            QdrantClient(path=settings.qdrant_path)
            if settings.qdrant_mode == "local"
            else QdrantClient(
                url=qdrant_connection_url(settings.qdrant_url), api_key=settings.qdrant_key or None, timeout=15
            )
        )
        target = settings.qdrant_path if settings.qdrant_mode == "local" else settings.qdrant_url
        try:
            self.store.bind_namespace(
                f"langchain-v2|{target}|{settings.collection}|{models.signature}"
            )
        except BaseException:
            self.q.close()
            raise
        self.vector_store = None

    def ensure_collection(self, *, signature_checked=False):
        """Create/validate the vector collection and lazily attach its LangChain adapter."""
        if not signature_checked:
            self.store.assert_embedding_signature(self.models.signature)
        if self.vector_store is not None:
            return
        if not self.q.collection_exists(self.s.collection):
            self.q.create_collection(
                self.s.collection,
                vectors_config=qm.VectorParams(size=self.s.dimensions, distance=qm.Distance.COSINE),
            )
        info = self.q.get_collection(self.s.collection)
        config = info.config.params.vectors
        if (
            not isinstance(config, qm.VectorParams)
            or config.size != self.s.dimensions
            or config.distance != qm.Distance.COSINE
        ):
            raise ValueError("Qdrant dimensions differ; choose a new DB/collection and reindex")
        if self.s.qdrant_mode == "server":
            self.q.create_payload_index(
                self.s.collection,
                "metadata.source",
                qm.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        if self.vector_store is None:
            self.vector_store = QdrantVectorStore(
                client=self.q,
                collection_name=self.s.collection,
                embedding=self.models.embeddings,
                validate_collection_config=False,
            )

    @timed("retrieval")
    def search(self, query: str, rerank=True):
        """Fuse semantic and literal-term candidates, validate active IDs, and cache only rankings."""
        with stage("retrieval_setup"):
            self.store.assert_embedding_signature(self.models.signature)
            if not self.store.document_count():
                return []
            self.ensure_collection(signature_checked=True)
        revision = self.store.revision()
        fingerprint = {
            "query": query,
            "revision": revision,
            "embedding": self.models.signature,
            "candidates": self.s.candidates,
            "final": self.s.final_chunks,
            "rerank": bool(rerank and self.s.rerank and not self.s.fast_mode and not self.s.minimum_calls),
            "rank_provider": self.s.chat_provider,
            "rank_model": self.s.effective_utility_model,
            "version": 2,
        }
        cache_key = digest(json.dumps(fingerprint, sort_keys=True, ensure_ascii=False))
        if self.s.search_cache_seconds:
            cached = self.store.search_cached(cache_key, self.s.search_cache_seconds)
            if cached is not None:
                return self.store.get_chunks(cached)
        with stage("embedding"):
            vector = self.models.embeddings.embed_query(query)
        with stage("vector_search"):
            documents = self.vector_store.similarity_search_by_vector(vector, k=self.s.candidates * 2)
        ids = [str(d.metadata.get("chunk_id", d.metadata.get("_id", ""))) for d in documents]
        dense = [c["id"] for c in self.store.get_chunks(ids)][: self.s.candidates]
        with stage("keyword_search"):
            keyword = self.store.keyword_search(query, self.s.candidates)
        fused = rrf([dense, keyword])[: self.s.candidates]
        found = self.store.get_chunks(fused)
        chosen = found[: self.s.final_chunks]
        cacheable = True
        if found and rerank and self.s.rerank and not self.s.fast_mode and not self.s.minimum_calls:
            try:
                passages, count = [], 0
                for chunk in found:
                    item = {
                        "chunk_id": chunk["id"],
                        "title": chunk["title"],
                        "text": chunk["text"],
                    }
                    cost = token_count(json.dumps(item, ensure_ascii=False))
                    if count + cost > self.s.model_input_tokens - token_count(query) - 2000:
                        break
                    passages.append(item)
                    count += cost
                ranked = self.models.structured(
                    Ranking,
                    RERANK_PROMPT,
                    {"query": query, "passages": passages},
                    max_tokens=650,
                )
                seen = {p["chunk_id"] for p in passages}
                allowed = {c["id"]: c for c in found if c["id"] in seen}
                ranked_ids = list(dict.fromkeys(ranked.chunk_ids))
                if any(id_ not in allowed for id_ in ranked_ids):
                    raise ValueError("Unknown rerank ID")
                chosen = [allowed[id_] for id_ in ranked_ids[: self.s.final_chunks]]
            except Exception:
                log.warning("Reranking unavailable; using fused retrieval order")
                cacheable = False
        # Do not reuse results if content changed concurrently during retrieval.
        if cacheable and self.s.search_cache_seconds and revision == self.store.revision():
            self.store.save_search(cache_key, [c["id"] for c in chosen])
        return chosen

    def close(self):
        """Release local file locks and remote HTTP clients before another CLI process runs."""
        self.q.close()


class Evidence:
    """Track exactly which passages a single request exposed to the model."""

    def __init__(self, budget):
        """Initialize a token-bounded ledger with server-owned source IDs."""
        self.budget, self.used = budget, 0
        self.items, self.by_chunk = {}, {}

    def add(self, chunks):
        """Register complete passages once and return references for repeated passages."""
        result = []
        for chunk in chunks:
            if chunk["id"] in self.by_chunk:
                result.append(self.items[self.by_chunk[chunk["id"]]])
                continue
            sid = f"S{len(self.items) + 1}"
            entry = {
                "source_id": sid,
                "chunk_id": chunk["id"],
                "title": chunk["title"],
                "source": chunk["source"],
                "indexed_at": chunk["indexed_at"],
                "text": chunk["text"],
            }
            cost = token_count(json.dumps(entry, ensure_ascii=False)) + 20
            if self.used + cost > self.budget:
                continue
            self.items[sid], self.by_chunk[chunk["id"]] = entry, sid
            self.used += cost
            result.append(entry)
        return result

    def tool_result(self, chunks):
        """Avoid resending text already supplied earlier in this request's tool conversation."""
        before = set(self.items)
        found = self.add(chunks)
        return {
            "evidence": [c for c in found if c["source_id"] not in before],
            "already_supplied_source_ids": [
                c["source_id"] for c in found if c["source_id"] in before
            ],
        }

    def sources(self, ids):
        """Construct clickable citations from validated indexed metadata, never model-invented URLs."""
        return [
            {
                "id": sid,
                "title": self.items[sid]["title"],
                "url": self.items[sid]["source"],
                "indexed_at": self.items[sid]["indexed_at"],
            }
            for sid in dict.fromkeys(ids)
        ]
