"""Admin-only incremental ingestion: full-site discovery is separate from answer retrieval limits."""

import json
import logging
import uuid
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

from langchain_core.documents import Document
from pypdf import PdfReader

from .crawler import Website
from .text import canonical_url, digest, normalize, public_page, split_text

log = logging.getLogger(__name__)


class Ingestor:
    """Publish versioned documents through LangChain and keep SQLite's active index authoritative."""

    def __init__(self, settings, store, models, retriever):
        """Inject storage/model adapters so indexing can be tested against local Qdrant."""
        self.s, self.store, self.models, self.r = settings, store, models, retriever

    def document(self, source, title, text):
        """Skip unchanged sources, split content with LangChain, and replace only changed generations."""
        text = normalize(text)
        title = normalize(title)[:300]
        if len(text) < 60:
            raise ValueError(
                "Too little extracted text; export JavaScript-only pages or OCR scanned PDFs"
            )
        self.r.ensure_collection()
        signature = f"lc-split-v2:{self.s.chunk_size}:{self.s.chunk_overlap}"
        content_hash = digest(signature + "\n" + title + "\n" + text)
        with self.store.lease("publication"):
            old = self.store.document(source)
            if old and old["hash"] == content_hash:
                return {"source": source, "status": "unchanged"}
            pieces = split_text(text, self.s.chunk_size, self.s.chunk_overlap)
            entries = [
                {
                    "id": str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            source + "|" + content_hash + "|" + str(i),
                        )
                    ),
                    "position": i,
                    "text": piece,
                }
                for i, piece in enumerate(pieces)
            ]
            documents = [
                Document(
                    page_content=title + "\n\n" + c["text"],
                    metadata={
                        "source": source,
                        "title": title,
                        "chunk_id": c["id"],
                        "position": c["position"],
                        "schema_version": 2,
                    },
                )
                for c in entries
            ]
            # Publish vectors first. Until SQLite commits, new IDs cannot become
            # evidence; old IDs remain active if embedding/vector upload fails.
            for start in range(0, len(entries), 32):
                self.r.vector_store.add_documents(
                    documents[start : start + 32],
                    ids=[c["id"] for c in entries[start : start + 32]],
                )
            old_ids = self.store.replace_document(
                source, title, content_hash, self.models.signature, entries
            )
            obsolete = sorted(set(old_ids) - {c["id"] for c in entries})
            self._delete_vectors(obsolete)
            return {"source": source, "status": "indexed", "chunks": len(entries)}

    def _delete_vectors(self, ids):
        """Remove obsolete vectors in bounded batches; stale IDs are never trusted by retrieval."""
        for start in range(0, len(ids), 256):
            self.r.vector_store.delete(ids[start : start + 256])

    def remove_source(self, source):
        """Unpublish a confirmed removed/admin-selected source before best-effort vector cleanup."""
        self.r.ensure_collection()
        with self.store.lease("publication"):
            ids = self.store.remove_document(source)
            try:
                self._delete_vectors(ids)
            except Exception:
                log.warning("Vector cleanup pending; inactive IDs remain excluded by SQLite")
        return {"source": source, "status": "removed", "chunks": len(ids)}

    def crawl(self, start, max_pages=0, sitemap=None, website=None):
        """Discover sitemap/internal URLs, refresh conditionally, and report actual coverage/errors."""
        self.r.ensure_collection()
        start = canonical_url(start, self.s.crawl_hosts)
        limit = min(max_pages or self.s.max_crawl_urls, self.s.max_crawl_urls)
        website = website or Website(self.s)
        queue, offered, visited = deque(), set(), set()
        report = {
            "discovered_urls": 0,
            "attempted": 0,
            "indexed": 0,
            "unchanged": 0,
            "removed": 0,
            "failed": 0,
            "errors": [],
            "sitemap_warnings": [],
            "limited": False,
            "remaining": 0,
            "outcomes": [],
        }

        def offer(url):
            """Enqueue a public URL once without allowing unbounded link growth."""
            try:
                url = canonical_url(url, self.s.crawl_hosts)
            except ValueError:
                return
            if url in offered or not public_page(url):
                return
            if len(offered) >= self.s.max_crawl_urls:
                report["limited"] = True
                return
            offered.add(url)
            queue.append(url)

        try:
            offer(start)
            for url in website.sitemap_urls(start, sitemap):
                offer(url)
            report["sitemap_warnings"] = website.discovery_warnings
            report["limited"] |= website.discovery_limited
            while queue and len(visited) < limit:
                url = queue.popleft()
                visited.add(url)
                report["attempted"] += 1
                try:
                    previous = self.store.http_state(url)
                    if previous and not self.store.document(previous["canonical_source"]):
                        previous = None
                    page = website.page(url, previous)
                    if page.status == 304:
                        if not previous:
                            raise ValueError("Unexpected 304 response without a stored source")
                        result = {
                            "source": previous["canonical_source"],
                            "status": "unchanged",
                        }
                    elif page.status in (404, 410):
                        source = previous["canonical_source"] if previous else url
                        result = (
                            self.remove_source(source)
                            if self.store.document(source)
                            else {"source": source, "status": "removed", "chunks": 0}
                        )
                    else:
                        result = self.document(page.url, page.title, page.text)
                    if page.status not in (404, 410):
                        canonical = previous["canonical_source"] if page.status == 304 else page.url
                        self.store.save_http_state(
                            url,
                            canonical,
                            page.headers.get("etag") or (previous or {}).get("etag"),
                            page.headers.get("last-modified")
                            or (previous or {}).get("last_modified"),
                            page.links,
                        )
                        for link in page.links:
                            offer(link)
                    report[result["status"]] += 1
                    report["outcomes"].append(result)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                except Exception as exc:
                    error = {"source": url, "error_type": type(exc).__name__}
                    report["errors"].append(error)
                    report["failed"] += 1
                    print(json.dumps({**error, "status": "failed"}), flush=True)
            report["discovered_urls"] = len(offered)
            report["remaining"] = len(queue)
            report["pending_urls_first_200"] = list(queue)[:200]
            report["limited"] |= bool(queue)
            return report
        finally:
            website.close()

    def pdf(self, path, source_url):
        """Import a reviewed public text PDF; preserve page labels and reject scans/oversized files."""
        p = Path(path)
        url = urlsplit(source_url)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query:
            raise ValueError(
                "Provide an approved public HTTPS citation URL without credentials/query strings"
            )
        if p.stat().st_size > 25_000_000:
            raise ValueError("PDF must be under 25 MB")
        reader = PdfReader(p)
        if len(reader.pages) > 300:
            raise ValueError("PDF must contain at most 300 pages; split larger documents")
        pages = [
            (i + 1, (page.extract_text() or "").strip()) for i, page in enumerate(reader.pages)
        ]
        if sum(len(text) for _, text in pages) < 60:
            raise ValueError("PDF needs OCR before indexing")
        text = "\n\n".join(f"Page {number}\n{text}" for number, text in pages if text)
        return self.document(source_url, p.stem, text)

    def cleanup_vectors(self):
        """Delete only vector IDs absent from the authoritative active corpus in this dedicated collection."""
        self.r.ensure_collection()
        removed, offset = 0, None
        with self.store.lease("publication"):
            while True:
                points, offset = self.r.q.scroll(
                    self.s.collection,
                    limit=256,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                ids = [str(point.id) for point in points]
                active = {c["id"] for c in self.store.get_chunks(ids)}
                stale = [id_ for id_ in ids if id_ not in active]
                self._delete_vectors(stale)
                removed += len(stale)
                if offset is None:
                    break
        return {"orphan_vectors_removed": removed}
