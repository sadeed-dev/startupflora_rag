"""Copy the existing v2 SQLite corpus/history/cache to EMPTY MySQL tables.

Purpose: embeddings dobara banaye bina MySQL use karo. Stop chat/API/ingestion
first. Original SQLite file and Qdrant vectors are never modified. This script
never calls OpenAI or writes .env. Migration is one atomic data transaction.
"""

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from rag.config import load_settings
from rag.mysql_store import MySQLStore

# Exact application columns only: do not copy SQLite FTS shadow tables or MySQL
# auto-generated row_id/source_hash columns. FULLTEXT is maintained by InnoDB.
TABLES = {
    "documents": ("source", "title", "hash", "indexed_at", "embedding_signature"),
    "chunks": ("id", "source", "position", "title", "text", "indexed_at"),
    "embedding_cache": ("key", "vector", "created_at"),
    "conversations": ("id", "user_id", "summary", "summarized_seq", "created_at"),
    "messages": ("conversation_id", "seq", "role", "content", "created_at", "metadata"),
    "http_cache": (
        "source",
        "canonical_source",
        "etag",
        "last_modified",
        "links",
        "checked_at",
    ),
    "answers": ("conversation_id", "request_id", "answer"),
    "corpus_meta": ("key", "value"),
}
ORDER = {
    "documents": "source",
    "chunks": "id",
    "embedding_cache": "key",
    "conversations": "id",
    "messages": "conversation_id,seq",
    "http_cache": "source",
    "answers": "conversation_id,request_id",
    "corpus_meta": "key",
}


def row_hash(rows):
    """Purpose: rows ka streaming fingerprint; long vectors ko dobara generate nahi karna."""
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        digest.update(
            json.dumps(list(row), ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def quoted_columns(columns):
    """Purpose: fixed trusted column names quote karo; reserved key/text safe rahein."""
    return ",".join("`" + column + "`" for column in columns)


def migrate(source_path, settings):
    """Purpose: empty MySQL target mein full copy + row hashes verify, phir commit karo."""
    source_path = Path(source_path).resolve()
    if not source_path.is_file():
        raise ValueError("Existing SQLite file not found. Check --from-db and DB_PATH.")
    with closing(
        sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)
    ) as source:
        source.execute(
            "BEGIN"
        )  # One consistent source snapshot, including committed WAL data.
        namespace = source.execute(
            "SELECT value FROM corpus_meta WHERE key='namespace'"
        ).fetchone()
        target = (
            settings.qdrant_path
            if settings.qdrant_mode == "local"
            else settings.qdrant_url
        )
        expected = f"langchain-v2|{target}|{settings.collection}|{settings.embedding_model}:{settings.dimensions}"
        if not namespace or namespace[0] != expected:
            raise ValueError(
                "Source namespace differs from .env. Keep the ORIGINAL Qdrant target/collection/model settings while migrating."
            )
        signatures = source.execute(
            "SELECT DISTINCT embedding_signature FROM documents"
        ).fetchall()
        if any(
            row[0] != f"{settings.embedding_model}:{settings.dimensions}"
            for row in signatures
        ):
            raise ValueError("SQLite embedding model/dimensions differ from .env.")
        mysql = MySQLStore(settings)
        copied = {}
        with mysql.connect() as db:
            # Serialize competing migrations. This row exists before the transaction.
            db.execute("SELECT value FROM schema_meta WHERE `key`='version' FOR UPDATE")
            for table in TABLES:
                if table == "corpus_meta":
                    occupied = db.execute(
                        "SELECT COUNT(*) FROM corpus_meta WHERE `key`<>'revision' OR value<>'0'"
                    ).fetchone()[0]
                else:
                    occupied = db.execute(f"SELECT COUNT(*) FROM `{table}`").fetchone()[
                        0
                    ]
                if occupied:
                    raise ValueError(
                        "MySQL target contains RAG data. Migration refuses to overwrite/merge it; use a new empty database."
                    )
            for table in ("leases", "requests", "rate_locks", "search_cache"):
                if db.execute(f"SELECT COUNT(*) FROM `{table}`").fetchone()[0]:
                    raise ValueError(
                        "MySQL target is already active. Stop processes and choose an empty database."
                    )
            for table, columns in TABLES.items():
                columns_sql = quoted_columns(columns)
                query = f"SELECT {columns_sql} FROM `{table}` ORDER BY {quoted_columns(ORDER[table].split(','))}"
                insert = f"INSERT INTO `{table}` ({columns_sql}) VALUES ({','.join('%s' for _ in columns)})"
                if table == "corpus_meta":
                    insert += " ON DUPLICATE KEY UPDATE value=VALUES(value)"
                rows = source.execute(query)
                copied[table] = 0
                while True:
                    batch = rows.fetchmany(
                        64
                    )  # Bounded memory and packet sizes, including vectors.
                    if not batch:
                        break
                    db.executemany(insert, batch)
                    copied[table] += len(batch)
                # Verify content, not just counts. Read exact copied columns in key order.
                expected_hash = row_hash(source.execute(query))
                actual_hash = row_hash(
                    tuple(row[column] for column in columns)
                    for row in db.execute(query)
                )
                if expected_hash != actual_hash:
                    raise ValueError(
                        f"Migration verification failed for {table}; MySQL data transaction rolled back."
                    )
                print(f"Verified {table}: {copied[table]} rows", flush=True)
            # Search ranks belong to SQLite FTS5 and must not be reused for MySQL FULLTEXT.
            # Active leases/rate windows are transient, so start fresh in the stopped app.
        return copied


def main():
    """Purpose: source path read karo, migration run karo; no model calls and no .env rewrite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-db", help="Existing SQLite file; defaults to DB_PATH from .env"
    )
    args = parser.parse_args()
    settings = load_settings()
    try:
        result = migrate(args.from_db or settings.db_path, settings)
    except (ValueError, sqlite3.Error) as exc:
        raise SystemExit(str(exc)) from None
    print(
        json.dumps(
            {
                "migration": "complete",
                "rows": result,
                "next": "Set DB_BACKEND=mysql, then run python main.py status",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
