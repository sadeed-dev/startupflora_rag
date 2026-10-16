"""SQLite is authoritative for active chunks, ownership and durable history."""

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .text import keyword_expression


def create_store(settings):
    """Purpose: DB_BACKEND ke hisaab se SQLite ya MySQL store choose karo."""
    if settings.db_backend == "mysql":
        # Lazy import: SQLite users ko MySQL driver install karna zaroori nahi.
        from .mysql_store import MySQLStore

        return MySQLStore(settings)
    return Store(settings.db_path)


class BusyConversation(Exception):
    """Another request currently owns the conversation lease."""


class RateLimited(Exception):
    """The user exceeded the configured request allowance."""


class RequestConflict(Exception):
    """A request UUID was reused for a different message."""


class Store:
    def __init__(self, path: str):
        """Create durable tables and indexes; never load conversation data into global memory."""
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS documents (
              source TEXT PRIMARY KEY, title TEXT NOT NULL, hash TEXT NOT NULL,
              indexed_at REAL NOT NULL, embedding_signature TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY, source TEXT NOT NULL, position INTEGER NOT NULL,
              title TEXT NOT NULL, text TEXT NOT NULL, indexed_at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source,position);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id UNINDEXED,title,text,
              tokenize='unicode61');
            CREATE TABLE IF NOT EXISTS embedding_cache (
              key TEXT PRIMARY KEY, vector TEXT NOT NULL, created_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS conversations (
              id TEXT PRIMARY KEY, user_id TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '',
              summarized_seq INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS conversations_user_created ON conversations(user_id,created_at DESC,id DESC);
            CREATE TABLE IF NOT EXISTS messages (
              conversation_id TEXT NOT NULL, seq INTEGER NOT NULL, role TEXT NOT NULL,
              content TEXT NOT NULL, created_at REAL NOT NULL, metadata TEXT NOT NULL DEFAULT '{}',
              PRIMARY KEY(conversation_id,seq));
            CREATE TABLE IF NOT EXISTS leases (
              conversation_id TEXT PRIMARY KEY, token TEXT NOT NULL, expires REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (user_id TEXT NOT NULL, at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS requests_user_time ON requests(user_id,at);
            CREATE TABLE IF NOT EXISTS corpus_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL);
            INSERT OR IGNORE INTO corpus_meta VALUES ('revision','0');
            CREATE TABLE IF NOT EXISTS search_cache (
              key TEXT PRIMARY KEY, chunk_ids TEXT NOT NULL, created_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS http_cache (
              source TEXT PRIMARY KEY, canonical_source TEXT NOT NULL, etag TEXT,
              last_modified TEXT, links TEXT NOT NULL, checked_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS answers (
              conversation_id TEXT NOT NULL, request_id TEXT NOT NULL, answer TEXT NOT NULL,
              PRIMARY KEY(conversation_id,request_id));
            """)

    @contextmanager
    def connect(self):
        """Commit successful operations, roll back failures, and always close the SQLite connection."""
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def document(self, source):
        """Read one source fingerprint to detect unchanged content before embedding."""
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM documents WHERE source=?", (source,)
            ).fetchone()
            return dict(row) if row else None

    def replace_document(self, source, title, hash_, signature, entries):
        """Atomically publish the new active chunk generation and keyword index."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = [
                r[0]
                for r in db.execute("SELECT id FROM chunks WHERE source=?", (source,))
            ]
            db.execute(
                "DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE source=?)",
                (source,),
            )
            db.execute("DELETE FROM chunks WHERE source=?", (source,))
            now = time.time()
            for item in entries:
                db.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                    (item["id"], source, item["position"], title, item["text"], now),
                )
                db.execute(
                    "INSERT INTO chunks_fts VALUES (?,?,?)",
                    (item["id"], title, item["text"]),
                )
            db.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?)",
                (source, title, hash_, now, signature),
            )
            db.execute(
                "UPDATE corpus_meta SET value=CAST(value AS INTEGER)+1 WHERE key='revision'"
            )
            db.execute("DELETE FROM search_cache")
            return old

    def get_chunks(self, ids):
        """Resolve only active chunk IDs in the requested order; discard stale vector hits."""
        if not ids:
            return []
        with self.connect() as db:
            marks = ",".join("?" for _ in ids)
            found = {
                r["id"]: dict(r)
                for r in db.execute(f"SELECT * FROM chunks WHERE id IN ({marks})", ids)
            }
            return [found[i] for i in ids if i in found]

    def neighbors(self, id_):
        """Expand an active chunk with its adjacent sections from the same source."""
        found = self.get_chunks([id_])
        if not found:
            return []
        c = found[0]
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM chunks WHERE source=? AND position BETWEEN ? AND ? ORDER BY position",
                    (c["source"], c["position"] - 1, c["position"] + 1),
                )
            ]

    def keyword_search(self, query, limit):
        """Rank literal query matches with FTS5 BM25 and a bounded result count."""
        expr = keyword_expression(query)
        if not expr:
            return []
        with self.connect() as db:
            return [
                r[0]
                for r in db.execute(
                    "SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts,0,2,1) LIMIT ?",
                    (expr, limit),
                )
            ]

    def cached_vector(self, key):
        """Read an exact-text embedding cached under its model-specific hash."""
        with self.connect() as db:
            row = db.execute(
                "SELECT vector FROM embedding_cache WHERE key=?", (key,)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def cache_vectors(self, values):
        """Persist computed vectors so repeated chunks and queries avoid paid embedding calls."""
        with self.connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO embedding_cache VALUES (?,?,?)",
                [(k, json.dumps(v), time.time()) for k, v in values],
            )

    def conversation(self, user_id, conversation_id=None):
        """Create a user-owned conversation or enforce ownership when retrieving one."""
        with self.connect() as db:
            if conversation_id:
                row = db.execute(
                    "SELECT * FROM conversations WHERE id=? AND user_id=?",
                    (conversation_id, user_id),
                ).fetchone()
                if not row:
                    raise LookupError("Conversation not found")
                return dict(row)
            cid = str(uuid.uuid4())
            db.execute(
                "INSERT INTO conversations(id,user_id,created_at) VALUES (?,?,?)",
                (cid, user_id, time.time()),
            )
            return {"id": cid, "user_id": user_id, "summary": "", "summarized_seq": 0}

    def recent(self, cid, limit):
        """Read only a bounded suffix of durable conversation messages."""
        with self.connect() as db:
            rows = db.execute(
                "SELECT seq,role,content FROM messages WHERE conversation_id=? ORDER BY seq DESC LIMIT ?",
                (cid, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def unsummarized(self, cid, after, before, limit=24):
        """Read older turns in sequence for incremental memory compaction."""
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT seq,role,content FROM messages WHERE conversation_id=? AND seq>? AND seq<? ORDER BY seq LIMIT ?",
                    (cid, after, before, limit),
                )
            ]

    def update_summary(self, cid, summary, seq):
        """Save the rolling user-context summary and its processed sequence cursor."""
        with self.connect() as db:
            db.execute(
                "UPDATE conversations SET summary=?,summarized_seq=? WHERE id=?",
                (summary, seq, cid),
            )

    def saved_answer(self, cid, request_id):
        """Read an idempotent reply for this exact conversation and request UUID."""
        with self.connect() as db:
            row = db.execute(
                "SELECT answer FROM answers WHERE conversation_id=? AND request_id=?",
                (cid, request_id),
            ).fetchone()
            return json.loads(row[0]) if row else None

    def save_turn(self, cid, request_id, user_text, result):
        # Persist the pair atomically: failed generations leave no half-turn.
        """Commit the user message, assistant answer, citations and retry record together."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            seq = db.execute(
                "SELECT COALESCE(MAX(seq),0) FROM messages WHERE conversation_id=?",
                (cid,),
            ).fetchone()[0]
            db.executemany(
                "INSERT INTO messages VALUES (?,?,?,?,?,?)",
                [
                    (cid, seq + 1, "user", user_text, time.time(), "{}"),
                    (
                        cid,
                        seq + 2,
                        "assistant",
                        result["answer"],
                        time.time(),
                        json.dumps(
                            {"sources": result["sources"], "status": result["status"]}
                        ),
                    ),
                ],
            )
            db.execute(
                "INSERT INTO answers VALUES (?,?,?)",
                (cid, request_id, json.dumps(result)),
            )

    @contextmanager
    def lease(self, cid):
        """Serialize turns for a conversation and release the lease even after a failure."""
        token, now = str(uuid.uuid4()), time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM leases WHERE expires<?", (now,))
            try:
                db.execute("INSERT INTO leases VALUES (?,?,?)", (cid, token, now + 180))
            except sqlite3.IntegrityError:
                raise BusyConversation("Another message is being processed")
        stop = threading.Event()

        def renew():
            """Renew long model/indexing operations without keeping a SQLite transaction open."""
            while not stop.wait(30):
                try:
                    with self.connect() as db:
                        db.execute(
                            "UPDATE leases SET expires=? WHERE conversation_id=? AND token=?",
                            (time.time() + 180, cid, token),
                        )
                except sqlite3.Error:
                    logging.getLogger(__name__).warning(
                        "Lease heartbeat could not reach storage"
                    )

        heartbeat = threading.Thread(target=renew, daemon=True)
        heartbeat.start()
        try:
            yield
        finally:
            stop.set()
            heartbeat.join(timeout=1)
            with self.connect() as db:
                db.execute(
                    "DELETE FROM leases WHERE conversation_id=? AND token=?",
                    (cid, token),
                )

    def check_rate(self, user_id, rpm):
        """Apply a transactional per-user request limit shared by service threads."""
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM requests WHERE at<?", (now - 60,))
            count = db.execute(
                "SELECT COUNT(*) FROM requests WHERE user_id=?", (user_id,)
            ).fetchone()[0]
            if count >= rpm:
                raise RateLimited("Too many requests; try again shortly")
            db.execute("INSERT INTO requests VALUES (?,?)", (user_id, now))

    def history_page(self, cid, after=0, limit=50):
        """Return a paginated owned-history page including saved source links."""
        with self.connect() as db:
            rows = [
                dict(r)
                for r in db.execute(
                    "SELECT seq,role,content,created_at,metadata FROM messages WHERE conversation_id=? AND seq>? ORDER BY seq LIMIT ?",
                    (cid, after, limit),
                )
            ]
            for row in rows:
                row.update(json.loads(row.pop("metadata")))
            return rows

    def delete_conversation(self, cid):
        """Delete conversation content, memory and retry data from this application DB."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for table in ("messages", "answers", "leases"):
                db.execute(f"DELETE FROM {table} WHERE conversation_id=?", (cid,))
            db.execute("DELETE FROM conversations WHERE id=?", (cid,))

    def document_count(self):
        """Report how many sources are currently active in the corpus."""
        with self.connect() as db:
            return db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def assert_embedding_signature(self, signature):
        """Reject searches that mix vectors from incompatible embedding configurations."""
        with self.connect() as db:
            mismatch = db.execute(
                "SELECT 1 FROM documents WHERE embedding_signature<>? LIMIT 1",
                (signature,),
            ).fetchone()
            if mismatch:
                raise ValueError(
                       "Embedding model/dimensions changed. Use a new DB and Qdrant collection and reindex."
                )

    def revision(self):
        """Return a monotonic corpus generation used to invalidate retrieval caches."""
        with self.connect() as db:
            return int(
                db.execute(
                    "SELECT value FROM corpus_meta WHERE key='revision'"
                ).fetchone()[0]
            )

    def bind_namespace(self, namespace):
        """Prevent one SQLite corpus from silently pointing at a different Qdrant collection."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT value FROM corpus_meta WHERE key='namespace'"
            ).fetchone()
            if row and row[0] != namespace:
                raise ValueError(
                    "Storage namespace changed. Use a new DB_PATH and reindex; migrate-history can copy chats."
                )
            if not row:
                if db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]:
                    raise ValueError(
                        "Legacy corpus detected. Use the v2 data paths and reindex; migrate-history can copy chats."
                    )
                db.execute(
                    "INSERT INTO corpus_meta VALUES ('namespace',?)", (namespace,)
                )

    def search_cached(self, key, ttl):
        """Return recently ranked active chunk IDs; distinguish an empty hit from a cache miss."""
        with self.connect() as db:
            row = db.execute(
                "SELECT chunk_ids FROM search_cache WHERE key=? AND created_at>?",
                (key, time.time() - ttl),
            ).fetchone()
            return json.loads(row[0]) if row else None

    def save_search(self, key, chunk_ids):
        """Persist ranking results rather than generated answers to avoid cross-user answer leakage."""
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO search_cache VALUES (?,?,?)",
                (key, json.dumps(chunk_ids), time.time()),
            )

    def http_state(self, source):
        """Read fetch validators and discovered links for conditional page refreshes."""
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM http_cache WHERE source=?", (source,)
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            result["links"] = json.loads(result["links"])
            return result

    def save_http_state(self, source, canonical_source, etag, last_modified, links):
        """Remember validators only after the source was successfully processed."""
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO http_cache VALUES (?,?,?,?,?,?)",
                (
                    source,
                    canonical_source,
                    etag,
                    last_modified,
                    json.dumps(links),
                    time.time(),
                ),
            )

    def remove_document(self, source):
        """Unpublish obsolete content atomically before vector cleanup, invalidating cached searches."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            ids = [
                r[0]
                for r in db.execute("SELECT id FROM chunks WHERE source=?", (source,))
            ]
            db.execute(
                "DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE source=?)",
                (source,),
            )
            db.execute("DELETE FROM chunks WHERE source=?", (source,))
            db.execute("DELETE FROM documents WHERE source=?", (source,))
            db.execute(
                "DELETE FROM http_cache WHERE source=? OR canonical_source=?",
                (source, source),
            )
            db.execute(
                "UPDATE corpus_meta SET value=CAST(value AS INTEGER)+1 WHERE key='revision'"
            )
            db.execute("DELETE FROM search_cache")
            return ids

    def inventory(self):
        """Expose indexing coverage and fetch dates without dumping page or conversation text."""
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT d.source,d.title,d.indexed_at,COUNT(c.id) AS chunks FROM documents d LEFT JOIN chunks c ON c.source=d.source GROUP BY d.source ORDER BY d.source"
                )
            ]

    def cleanup_cache(self, days):
        """Bound cache growth while leaving indexed content and conversation history intact."""
        with self.connect() as db:
            counts = {}
            for table in ("embedding_cache", "search_cache"):
                cursor = db.execute(
                    f"DELETE FROM {table} WHERE created_at<?",
                    (time.time() - days * 86400,),
                )
                counts[table] = cursor.rowcount
            return counts

    def import_history(self, source_path):
        """Copy non-conflicting owned conversations from an older local DB without importing its corpus."""
        source_path = Path(source_path).resolve()
        if source_path == Path(self.path).resolve() or not source_path.is_file():
            raise ValueError("Choose a different existing SQLite history file")
        source = sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)
        source.row_factory = sqlite3.Row
        source.execute(
            "BEGIN"
        )  # Read a consistent snapshot even if the old app is still active.
        count = 0
        try:
            conversations = source.execute("SELECT * FROM conversations").fetchall()
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                for conv in conversations:
                    if db.execute(
                        "SELECT 1 FROM conversations WHERE id=?", (conv["id"],)
                    ).fetchone():
                        continue
                    db.execute(
                        "INSERT INTO conversations VALUES (?,?,?,?,?)",
                        tuple(
                            conv[k]
                            for k in (
                                "id",
                                "user_id",
                                "summary",
                                "summarized_seq",
                                "created_at",
                            )
                        ),
                    )
                    for row in source.execute(
                        "SELECT * FROM messages WHERE conversation_id=?", (conv["id"],)
                    ):
                        meta = row["metadata"] if "metadata" in row.keys() else "{}"
                        db.execute(
                            "INSERT INTO messages VALUES (?,?,?,?,?,?)",
                            tuple(
                                row[k]
                                for k in (
                                    "conversation_id",
                                    "seq",
                                    "role",
                                    "content",
                                    "created_at",
                                )
                            )
                            + (meta,),
                        )
                    # Old answer caches may refer to a different corpus. Do not reuse them.
                    count += 1
        finally:
            source.close()
        return count

    def list_conversations(self, user_id, limit=20, cursor=None):
        """List only one user's conversations using a stable keyset cursor for resumable chat UIs."""
        import base64

        before = None
        if cursor:
            try:
                before = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
                if (
                    not isinstance(before, list)
                    or len(before) != 2
                    or not isinstance(before[0], (float, int))
                ):
                    raise ValueError("Invalid conversation cursor")
                uuid.UUID(before[1])
            except Exception as exc:
                raise ValueError("Invalid conversation cursor") from exc
        with self.connect() as db:
            query = "SELECT id,created_at FROM conversations WHERE user_id=?"
            params = [user_id]
            if before:
                query += " AND (created_at,id)<(?,?)"
                params.extend(before)
            query += " ORDER BY created_at DESC,id DESC LIMIT ?"
            rows = [dict(r) for r in db.execute(query, params + [limit + 1])]
        more = len(rows) > limit
        rows = rows[:limit]
        next_cursor = (
            base64.urlsafe_b64encode(
                json.dumps([rows[-1]["created_at"], rows[-1]["id"]]).encode()
            ).decode()
            if more
            else None
        )
        return {"conversations": rows, "next_cursor": next_cursor}
