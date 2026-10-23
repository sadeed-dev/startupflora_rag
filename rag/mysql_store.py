"""MySQL/MariaDB store: same RAG interface, InnoDB transactions and FULLTEXT search.

Purpose: SQLite ke records ko MySQL mein store karo; Qdrant vectors stay unchanged.
Transactions borrow exclusive connections from a bounded, thread-safe pool.
"""

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, LifoQueue

import pymysql
from pymysql.cursors import DictCursor

from .answer_cache import AnswerCache
from .request_tracking import RequestTrackingStore
from .store import BusyConversation, RateLimited
from .text import keyword_expression
from .timing import stage


class _Row(dict):
    """Support both named fields and positional access used by the existing store interface."""

    def __getitem__(self, key):
        """Allow row[0] for counts and row['id'] for named columns."""
        return (
            tuple(self.values())[key]
            if isinstance(key, int)
            else super().__getitem__(key)
        )


class _Result:
    """Buffer one bounded result so later cursor executions cannot invalidate earlier rows."""

    def __init__(self, cursor):
        """Copy fetched rows and affected-row count before the cursor is closed."""
        self.rowcount = cursor.rowcount
        self.rows = (
            [_Row(row) for row in cursor.fetchall()] if cursor.description else []
        )
        self.position = 0

    def fetchone(self):
        """Read the next row, returning None when the result is exhausted."""
        if self.position >= len(self.rows):
            return None
        row = self.rows[self.position]
        self.position += 1
        return row

    def fetchall(self):
        """Read remaining rows without retaining an open MySQL cursor."""
        rows = self.rows[self.position :]
        self.position = len(self.rows)
        return rows

    def __iter__(self):
        """Allow existing for-row loops to iterate the remaining result."""
        return iter(self.fetchall())


class _Session:
    """Small parameterized-query interface shared by all methods in this file."""

    def __init__(self, connection):
        """Keep the transaction's connection private to its current caller/thread."""
        self.connection = connection

    def execute(self, sql, params=None):
        """Execute a MySQL statement with bound parameters; never interpolate user text."""
        with stage("database_query"), self.connection.cursor() as cursor:
            cursor.execute(sql, params)
            return _Result(cursor)

    def executemany(self, sql, rows):
        """Insert/update a batch using the driver's parameter binding."""
        with stage("database_query"), self.connection.cursor() as cursor:
            cursor.executemany(sql, rows)
            return _Result(cursor)


class MySQLStore(AnswerCache, RequestTrackingStore):
    """Store active chunks, ownership, history and caches in a dedicated InnoDB database."""

    def document(self, source):
        """Read one source fingerprint to detect unchanged content before embedding."""
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM documents WHERE source=%s", (source,)
            ).fetchone()
            return dict(row) if row else None

    def get_chunks(self, ids):
        """Resolve only active chunk IDs in the requested order; discard stale vector hits."""
        if not ids:
            return []
        with self.connect() as db:
            marks = ",".join(("%s" for _ in ids))
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
                    "SELECT * FROM chunks WHERE source=%s AND `position` BETWEEN %s AND %s ORDER BY `position`",
                    (c["source"], c["position"] - 1, c["position"] + 1),
                )
            ]

    def cached_vector(self, key):
        """Read an exact-text embedding cached under its model-specific hash."""
        with self.connect() as db:
            row = db.execute(
                "SELECT vector FROM embedding_cache WHERE `key`=%s", (key,)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def cache_vectors(self, values):
        """Persist computed vectors so repeated chunks and queries avoid paid embedding calls."""
        with self.connect() as db:
            db.executemany(
                "INSERT INTO embedding_cache (`key`,`vector`,`created_at`) VALUES (%s,%s,%s) ON DUPLICATE KEY UPDATE `vector`=VALUES(`vector`),`created_at`=VALUES(`created_at`)",
                [(k, json.dumps(v), time.time()) for k, v in values],
            )

    def conversation(self, user_id, conversation_id=None):
        """Create a user-owned conversation or enforce ownership when retrieving one."""
        with self.connect() as db:
            if conversation_id:
                row = db.execute(
                    "SELECT * FROM conversations WHERE id=%s AND user_id=%s",
                    (conversation_id, user_id),
                ).fetchone()
                if not row:
                    raise LookupError("Conversation not found")
                return dict(row)
            cid = str(uuid.uuid4())
            db.execute(
                "INSERT INTO conversations(id,user_id,summary,created_at) VALUES (%s,%s,%s,%s)",
                (cid, user_id, "", time.time()),
            )
            return {"id": cid, "user_id": user_id, "summary": "", "summarized_seq": 0}

    def recent(self, cid, limit):
        """Read only a bounded suffix of durable conversation messages."""
        with self.connect() as db:
            rows = db.execute(
                "SELECT seq,role,content FROM messages WHERE conversation_id=%s ORDER BY seq DESC LIMIT %s",
                (cid, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def unsummarized(self, cid, after, before, limit=24):
        """Read older turns in sequence for incremental memory compaction."""
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT seq,role,content FROM messages WHERE conversation_id=%s AND seq>%s AND seq<%s ORDER BY seq LIMIT %s",
                    (cid, after, before, limit),
                )
            ]

    def update_summary(self, cid, summary, seq):
        """Save the rolling user-context summary and its processed sequence cursor."""
        with self.connect() as db:
            db.execute(
                "UPDATE conversations SET summary=%s,summarized_seq=%s WHERE id=%s",
                (summary, seq, cid),
            )

    def saved_answer(self, cid, request_id):
        """Read an idempotent reply for this exact conversation and request UUID."""
        with self.connect() as db:
            row = db.execute(
                "SELECT answer FROM answers WHERE conversation_id=%s AND request_id=%s",
                (cid, request_id),
            ).fetchone()
            return json.loads(row[0]) if row else None

    def save_turn(self, cid, request_id, user_text, result):
        """Commit the user message, assistant answer, citations and retry record together."""
        with self.connect() as db:
            db.execute("START TRANSACTION")
            db.execute(
                "SELECT id FROM conversations WHERE id=%s FOR UPDATE", (cid,)
            ).fetchone()
            seq = db.execute(
                "SELECT COALESCE(MAX(seq),0) FROM messages WHERE conversation_id=%s",
                (cid,),
            ).fetchone()[0]
            db.executemany(
                "INSERT INTO messages VALUES (%s,%s,%s,%s,%s,%s)",
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
                "INSERT INTO answers VALUES (%s,%s,%s)",
                (cid, request_id, json.dumps(result)),
            )

    def history_page(self, cid, after=0, limit=50):
        """Return a paginated owned-history page including saved source links."""
        with self.connect() as db:
            rows = [
                dict(r)
                for r in db.execute(
                    "SELECT seq,role,content,created_at,metadata FROM messages WHERE conversation_id=%s AND seq>%s ORDER BY seq LIMIT %s",
                    (cid, after, limit),
                )
            ]
            for row in rows:
                row.update(json.loads(row.pop("metadata")))
            return rows

    def delete_conversation(self, cid):
        """Delete conversation content, memory and retry data from this application DB."""
        with self.connect() as db:
            db.execute("START TRANSACTION")
            for table in ("messages", "answers", "leases"):
                db.execute(f"DELETE FROM {table} WHERE conversation_id=%s", (cid,))
            db.execute("DELETE FROM conversations WHERE id=%s", (cid,))

    def document_count(self):
        """Report how many sources are currently active in the corpus."""
        with self.connect() as db:
            return db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def assert_embedding_signature(self, signature):
        """Reject searches that mix vectors from incompatible embedding configurations."""
        with self.connect() as db:
            mismatch = db.execute(
                "SELECT 1 FROM documents WHERE embedding_signature<>%s LIMIT 1",
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
                    "SELECT value FROM corpus_meta WHERE `key`='revision'"
                ).fetchone()[0]
            )

    def search_cached(self, key, ttl):
        """Return recently ranked active chunk IDs; distinguish an empty hit from a cache miss."""
        with self.connect() as db:
            row = db.execute(
                "SELECT chunk_ids FROM search_cache WHERE `key`=%s AND created_at>%s",
                (key, time.time() - ttl),
            ).fetchone()
            return json.loads(row[0]) if row else None

    def save_search(self, key, chunk_ids):
        """Persist ranking results rather than generated answers to avoid cross-user answer leakage."""
        with self.connect() as db:
            db.execute(
                "INSERT INTO search_cache (`key`,`chunk_ids`,`created_at`) VALUES (%s,%s,%s) ON DUPLICATE KEY UPDATE `chunk_ids`=VALUES(`chunk_ids`),`created_at`=VALUES(`created_at`)",
                (key, json.dumps(chunk_ids), time.time()),
            )

    def http_state(self, source):
        """Read fetch validators and discovered links for conditional page refreshes."""
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM http_cache WHERE source=%s", (source,)
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
                "INSERT INTO http_cache (`source`,`canonical_source`,`etag`,`last_modified`,`links`,`checked_at`) VALUES (%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE `canonical_source`=VALUES(`canonical_source`),`etag`=VALUES(`etag`),`last_modified`=VALUES(`last_modified`),`links`=VALUES(`links`),`checked_at`=VALUES(`checked_at`)",
                (
                    source,
                    canonical_source,
                    etag,
                    last_modified,
                    json.dumps(links),
                    time.time(),
                ),
            )

    def cleanup_cache(self, days):
        """Bound cache growth while leaving indexed content and conversation history intact."""
        with self.connect() as db:
            counts = {}
            for table in ("embedding_cache", "search_cache"):
                cursor = db.execute(
                    f"DELETE FROM {table} WHERE created_at<%s",
                    (time.time() - days * 86400,),
                )
                counts[table] = cursor.rowcount
            return counts

    def import_history(self, source_path):
        """Copy non-conflicting owned conversations from an older local DB without importing its corpus."""
        source_path = Path(source_path).resolve()
        if not source_path.is_file():
            raise ValueError("Choose an existing SQLite history file")
        source = sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)
        source.row_factory = sqlite3.Row
        source.execute("BEGIN")
        count = 0
        try:
            conversations = source.execute("SELECT * FROM conversations").fetchall()
            with self.connect() as db:
                db.execute("START TRANSACTION")
                for conv in conversations:
                    if db.execute(
                        "SELECT 1 FROM conversations WHERE id=%s", (conv["id"],)
                    ).fetchone():
                        continue
                    db.execute(
                        "INSERT INTO conversations VALUES (%s,%s,%s,%s,%s)",
                        tuple(
                            (
                                conv[k]
                                for k in (
                                    "id",
                                    "user_id",
                                    "summary",
                                    "summarized_seq",
                                    "created_at",
                                )
                            )
                        ),
                    )
                    for row in source.execute(
                        "SELECT * FROM messages WHERE conversation_id=?", (conv["id"],)
                    ):
                        meta = row["metadata"] if "metadata" in row.keys() else "{}"
                        db.execute(
                            "INSERT INTO messages VALUES (%s,%s,%s,%s,%s,%s)",
                            tuple(
                                (
                                    row[k]
                                    for k in (
                                        "conversation_id",
                                        "seq",
                                        "role",
                                        "content",
                                        "created_at",
                                    )
                                )
                            )
                            + (meta,),
                        )
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
                    or (not isinstance(before[0], (float, int)))
                ):
                    raise ValueError("Invalid conversation cursor")
                uuid.UUID(before[1])
            except Exception as exc:
                raise ValueError("Invalid conversation cursor") from exc
        with self.connect() as db:
            query = "SELECT id,created_at FROM conversations WHERE user_id=%s"
            params = [user_id]
            if before:
                query += " AND (created_at,id)<(%s,%s)"
                params.extend(before)
            query += " ORDER BY created_at DESC,id DESC LIMIT %s"
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

    def __init__(self, settings):
        """Purpose: dedicated MySQL tables ready karo; existing rows never reset hote hain."""
        self.s = settings
        self.path = (
            f"mysql:{settings.mysql_database}"  # Safe label; never include password.
        )
        self._pool = LifoQueue(maxsize=settings.mysql_pool_size)
        self._pool_slots = threading.BoundedSemaphore(settings.mysql_pool_size)
        self._pool_guard = threading.Lock()
        self._pool_closed = False
        # DDL commits implicitly in MySQL. Run before any migration/data transaction.
        try:
            self._initialize_schema()
        except BaseException:
            self.close()
            raise

    def _initialize_schema(self):
        with self.connect() as db:
            for statement in SCHEMA:
                db.execute(statement)
            self.request_timing_table(db)
            db.execute(
                "INSERT IGNORE INTO corpus_meta (`key`,value) VALUES ('revision','0')"
            )
            db.execute(
                "INSERT IGNORE INTO schema_meta (`key`,value) VALUES ('version','1')"
            )
            version = db.execute(
                "SELECT value FROM schema_meta WHERE `key`='version'"
            ).fetchone()[0]
            if version != "1":
                raise ValueError("Unsupported MySQL RAG schema version")

    def _new_connection(self):
        """Configure each new physical connection once, before it enters the pool."""
        tls = {}
        if self.s.mysql_ssl_ca:
            tls = dict(
                ssl_ca=self.s.mysql_ssl_ca,
                ssl_verify_cert=True,
                ssl_verify_identity=True,
            )
        try:
            connection = pymysql.connect(
                host=self.s.mysql_host,
                port=self.s.mysql_port,
                user=self.s.mysql_user,
                password=self.s.mysql_password,
                database=self.s.mysql_database,
                charset="utf8mb4",
                cursorclass=DictCursor,
                autocommit=False,
                connect_timeout=10,
                read_timeout=30,
                write_timeout=30,
                **tls,
            )
        except pymysql.Error:
            raise ValueError(
                "MySQL connection failed. Check MYSQL_HOST/PORT/DATABASE/USER/PASSWORD and server access."
            ) from None
        try:
            with connection.cursor() as cursor:
                # READ COMMITTED avoids stale snapshots across lease/row-lock operations.
                cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED")
                cursor.execute(
                    "SET SESSION sql_mode='STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION,ONLY_FULL_GROUP_BY'"
                )
                cursor.execute("SET SESSION innodb_lock_wait_timeout=15")
            return connection
        except BaseException:
            connection.close()
            raise

    def _checkout(self):
        if not self._pool_slots.acquire(timeout=self.s.mysql_pool_timeout):
            raise ValueError("MySQL connection pool is busy; retry shortly")
        try:
            with self._pool_guard:
                if self._pool_closed:
                    raise ValueError("MySQL connection pool is closed")
            try:
                connection = self._pool.get_nowait()
            except Empty:
                return self._new_connection()
            try:
                # Do not transparently reconnect inside an active transaction.
                connection.ping(reconnect=False)
                return connection
            except pymysql.Error:
                connection.close()
                return self._new_connection()
        except BaseException:
            self._pool_slots.release()
            raise

    @contextmanager
    def connect(self):
        """Borrow one connection per transaction; never share it between threads."""
        with stage("database_acquire"):
            connection = self._checkout()
        reusable = True
        try:
            yield _Session(connection)
            with stage("database_commit"):
                connection.commit()
        except BaseException:
            try:
                with stage("database_rollback"):
                    connection.rollback()
            except BaseException:
                reusable = False
            raise
        finally:
            try:
                with self._pool_guard:
                    if reusable and not self._pool_closed and connection.open:
                        self._pool.put_nowait(connection)
                    else:
                        connection.close()
            finally:
                self._pool_slots.release()

    def close(self):
        """Close idle sockets; outstanding borrowers close their sockets on return."""
        with self._pool_guard:
            self._pool_closed = True
            while True:
                try:
                    self._pool.get_nowait().close()
                except Empty:
                    break

    def replace_document(self, source, title, hash_, signature, entries):
        """Purpose: new active chunks/page fingerprint ek transaction mein publish karo."""
        with self.connect() as db:
            old = [
                r[0]
                for r in db.execute("SELECT id FROM chunks WHERE source=%s", (source,))
            ]
            db.execute("DELETE FROM chunks WHERE source=%s", (source,))
            now = time.time()
            db.executemany(
                "INSERT INTO chunks (id,source,`position`,title,`text`,indexed_at) VALUES (%s,%s,%s,%s,%s,%s)",
                [
                    (c["id"], source, c["position"], title, c["text"], now)
                    for c in entries
                ],
            )
            db.execute(
                "INSERT INTO documents (source,title,hash,indexed_at,embedding_signature) VALUES (%s,%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE title=VALUES(title),hash=VALUES(hash),indexed_at=VALUES(indexed_at),embedding_signature=VALUES(embedding_signature)",
                (source, title, hash_, now, signature),
            )
            db.execute(
                "UPDATE corpus_meta SET value=CAST(value AS SIGNED)+1 WHERE `key`='revision'"
            )
            db.execute("DELETE FROM search_cache")
            # InnoDB maintains chunks FULLTEXT index automatically on commit.
            return old

    def keyword_search(self, query, limit):
        """Purpose: SQLite FTS5 ki jagah MySQL FULLTEXT se lexical candidates rank karo."""
        # keyword_expression already extracts literal words and removes operators.
        # Boolean mode's unprefixed terms mean OR; quotes keep user operators inert.
        words = re.findall(r'"([^\"]+)"', keyword_expression(query))
        if not words:
            return []
        expression = " ".join('"' + word + '"' for word in words)
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,MATCH(title,`text`) AGAINST (%s IN BOOLEAN MODE) AS score "
                "FROM chunks WHERE MATCH(title,`text`) AGAINST (%s IN BOOLEAN MODE) "
                "ORDER BY score DESC,id LIMIT %s",
                (expression, expression, int(limit)),
            )
            return [row["id"] for row in rows]

    def remove_document(self, source):
        """Purpose: removed page ko active corpus se hatao; old IDs cleanup ke liye do."""
        with self.connect() as db:
            ids = [
                r[0]
                for r in db.execute("SELECT id FROM chunks WHERE source=%s", (source,))
            ]
            db.execute("DELETE FROM chunks WHERE source=%s", (source,))
            db.execute("DELETE FROM documents WHERE source=%s", (source,))
            db.execute(
                "DELETE FROM http_cache WHERE source=%s OR canonical_source=%s",
                (source, source),
            )
            db.execute(
                "UPDATE corpus_meta SET value=CAST(value AS SIGNED)+1 WHERE `key`='revision'"
            )
            db.execute("DELETE FROM search_cache")
            return ids

    def inventory(self):
        """Purpose: active page list/counts do; strict MySQL GROUP BY rules follow hon."""
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT d.source,d.title,d.indexed_at,COUNT(c.id) AS chunks FROM documents d "
                    "LEFT JOIN chunks c ON c.source=d.source "
                    "GROUP BY d.source,d.title,d.indexed_at ORDER BY d.source"
                )
            ]

    def bind_namespace(self, namespace):
        """Purpose: MySQL corpus ko existing Qdrant target/model identity se match karo."""
        with self.connect() as db:
            # Revision row always exists; lock it to serialize first namespace creation.
            db.execute(
                "SELECT value FROM corpus_meta WHERE `key`='revision' FOR UPDATE"
            )
            row = db.execute(
                "SELECT value FROM corpus_meta WHERE `key`='namespace'"
            ).fetchone()
            if row and row[0] != namespace:
                raise ValueError(
                    "Storage namespace changed. Keep the same Qdrant target, collection, model and dimensions after migration."
                )
            if not row:
                if db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]:
                    raise ValueError(
                        "Existing corpus has no namespace; migrate the full v2 SQLite database."
                    )
                db.execute(
                    "INSERT INTO corpus_meta (`key`,value) VALUES ('namespace',%s)",
                    (namespace,),
                )

    @contextmanager
    def lease(self, cid):
        """Purpose: conversation/publication ek owner process kare; lease heartbeat renew ho."""
        token, now = str(uuid.uuid4()), time.time()
        with self.connect() as db:
            # Delete only this expired key, avoiding a global range lock on leases.
            db.execute(
                "DELETE FROM leases WHERE conversation_id=%s AND expires<%s", (cid, now)
            )
            try:
                db.execute(
                    "INSERT INTO leases VALUES (%s,%s,%s)", (cid, token, now + 180)
                )
            except pymysql.IntegrityError as exc:
                if exc.args[0] != 1062:
                    raise
                raise BusyConversation(
                    "Another message or ingestion publication is being processed"
                ) from None
        stop = threading.Event()

        def renew():
            """Purpose: long model/upload task mein lease ki expiry aage badhao."""
            while not stop.wait(30):
                try:
                    with self.connect() as db:
                        db.execute(
                            "UPDATE leases SET expires=%s WHERE conversation_id=%s AND token=%s",
                            (time.time() + 180, cid, token),
                        )
                except (pymysql.Error, ValueError):
                    logging.getLogger(__name__).warning(
                        "MySQL lease heartbeat could not reach storage"
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
                    "DELETE FROM leases WHERE conversation_id=%s AND token=%s",
                    (cid, token),
                )

    def check_rate(self, user_id, rpm):
        """Purpose: same user ki parallel requests serialize karke rate limit enforce karo."""
        now = time.time()
        with self.connect() as db:
            # This per-user InnoDB row acts as a transaction lock, not an expiring lease.
            db.execute(
                "INSERT INTO rate_locks (user_id,touched) VALUES (%s,%s) ON DUPLICATE KEY UPDATE touched=VALUES(touched)",
                (user_id, now),
            )
            db.execute(
                "SELECT user_id FROM rate_locks WHERE user_id=%s FOR UPDATE", (user_id,)
            )
            db.execute(
                "DELETE FROM requests WHERE user_id=%s AND `at`<%s", (user_id, now - 60)
            )
            count = db.execute(
                "SELECT COUNT(*) FROM requests WHERE user_id=%s AND `at`>=%s",
                (user_id, now - 60),
            ).fetchone()[0]
            if count >= rpm:
                raise RateLimited("Too many requests; try again shortly")
            db.execute(
                "INSERT INTO requests (user_id,`at`) VALUES (%s,%s)", (user_id, now)
            )


# Purpose: MySQL 8.0+/MariaDB 10.11+ InnoDB tables; no SQLite FTS shadow tables.
# Long URLs keep full text; a generated SHA256 unique key avoids prefix-key collisions.
# IDs/ownership use binary collation, while FULLTEXT text uses Unicode case folding.
SCHEMA = (
    """CREATE TABLE IF NOT EXISTS schema_meta (`key` VARCHAR(64) PRIMARY KEY,value TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS documents (
        row_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        source VARCHAR(3000) NOT NULL,
        source_hash BINARY(32) GENERATED ALWAYS AS (UNHEX(SHA2(source,256))) STORED,
        title TEXT NOT NULL,hash CHAR(64) NOT NULL,indexed_at DOUBLE NOT NULL,
        embedding_signature VARCHAR(255) NOT NULL,UNIQUE KEY documents_source(source_hash)
    )""",
    """CREATE TABLE IF NOT EXISTS chunks (
        id CHAR(36) PRIMARY KEY,source VARCHAR(3000) NOT NULL,`position` INT NOT NULL,
        title TEXT COLLATE utf8mb4_unicode_ci NOT NULL,
        `text` LONGTEXT COLLATE utf8mb4_unicode_ci NOT NULL,indexed_at DOUBLE NOT NULL,
        KEY chunks_source(source(512),`position`),FULLTEXT KEY chunks_fulltext(title,`text`)
    )""",
    """CREATE TABLE IF NOT EXISTS embedding_cache (
        `key` CHAR(64) PRIMARY KEY,vector LONGTEXT NOT NULL,created_at DOUBLE NOT NULL,
        KEY embedding_age(created_at)
    )""",
    """CREATE TABLE IF NOT EXISTS conversations (
        id CHAR(36) PRIMARY KEY,user_id VARCHAR(255) NOT NULL,summary LONGTEXT NOT NULL,
        summarized_seq BIGINT NOT NULL DEFAULT 0,created_at DOUBLE NOT NULL,
        KEY conversations_user_created(user_id,created_at,id)
    )""",
    """CREATE TABLE IF NOT EXISTS messages (
        conversation_id CHAR(36) NOT NULL,seq BIGINT NOT NULL,role VARCHAR(16) NOT NULL,
        content LONGTEXT NOT NULL,created_at DOUBLE NOT NULL,metadata LONGTEXT NOT NULL,
        PRIMARY KEY(conversation_id,seq)
    )""",
    """CREATE TABLE IF NOT EXISTS leases (
        conversation_id VARCHAR(255) PRIMARY KEY,token CHAR(36) NOT NULL,expires DOUBLE NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS requests (
        user_id VARCHAR(255) NOT NULL,`at` DOUBLE NOT NULL,KEY requests_user_time(user_id,`at`)
    )""",
    """CREATE TABLE IF NOT EXISTS rate_locks (
        user_id VARCHAR(255) PRIMARY KEY,touched DOUBLE NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS corpus_meta (`key` VARCHAR(64) PRIMARY KEY,value LONGTEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS search_cache (
        `key` CHAR(64) PRIMARY KEY,chunk_ids LONGTEXT NOT NULL,created_at DOUBLE NOT NULL,
        KEY search_age(created_at)
    )""",
    """CREATE TABLE IF NOT EXISTS http_cache (
        row_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        source VARCHAR(3000) NOT NULL,
        source_hash BINARY(32) GENERATED ALWAYS AS (UNHEX(SHA2(source,256))) STORED,
        canonical_source VARCHAR(3000) NOT NULL,etag TEXT,last_modified TEXT,
        links LONGTEXT NOT NULL,checked_at DOUBLE NOT NULL,
        UNIQUE KEY http_source(source_hash),KEY http_canonical(canonical_source(512))
    )""",
    """CREATE TABLE IF NOT EXISTS answers (
        conversation_id CHAR(36) NOT NULL,request_id CHAR(36) NOT NULL,answer LONGTEXT NOT NULL,
        PRIMARY KEY(conversation_id,request_id)
    )""",
)
SCHEMA = tuple(
    sql
    + " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin ROW_FORMAT=DYNAMIC"
    for sql in SCHEMA
)
