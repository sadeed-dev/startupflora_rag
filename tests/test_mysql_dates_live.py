"""Optional regression against an explicitly selected disposable local MySQL instance."""

import os
import uuid
from datetime import datetime, timedelta

import pymysql
import pytest

from rag.answer_cache import AnswerCache
from rag.config import Settings
from rag.diagnostics import SCHEMA as DIAGNOSTICS
from rag.mysql_dates import DATE_COLUMNS, IST, inspect_columns, inspect_indexes, migration_statements, to_storage
from rag.mysql_store import MySQLStore, SCHEMA
from rag.request_tracking import RequestTrackingStore
from rag.request_tracking import RequestTimingWriter
from rag.store import BusyConversation, RateLimited
from rag.timing import request_timings


@pytest.mark.skipif(not os.environ.get("RAG_MYSQL_TEST_PORT"), reason="No disposable local MySQL instance selected")
@pytest.mark.parametrize("layout", ["legacy", "paired", "simple"])
def test_simple_dates_preserve_old_data_cache_expiry_pagination_and_limits(layout, monkeypatch):
    port = int(os.environ["RAG_MYSQL_TEST_PORT"])
    password = "flora-isolated-test-only"
    database = "date_test_" + uuid.uuid4().hex
    raw = pymysql.connect(host="127.0.0.1", port=port, user="root", password=password, autocommit=True)
    class Capture:
        s = True
        def __init__(self):
            self.statements = []
        def execute(self, sql):
            self.statements.append(sql)
    capture = Capture()
    AnswerCache.answer_cache_table(capture, capture)
    RequestTrackingStore.request_timing_table(capture, capture)
    epoch = 1792739594.887433
    with raw.cursor() as cursor:
        cursor.execute(f"CREATE DATABASE `{database}`")
        cursor.execute(f"USE `{database}`")
        cursor.execute("SET time_zone='+00:00'")
        for statement in (*SCHEMA, *capture.statements, DIAGNOSTICS):
            cursor.execute(statement)
        if layout != "simple":
            for table, fields in DATE_COLUMNS.items():
                for column, kind in fields.items():
                    definition = "DOUBLE NOT NULL" if kind == "epoch" else "DATETIME NOT NULL"
                    cursor.execute(f"ALTER TABLE `{table}` MODIFY COLUMN `{column}` {definition}")
        inserted_date = to_storage(epoch, "epoch") if layout == "simple" else epoch
        cursor.execute("INSERT INTO conversations(id,user_id,summary,created_at) VALUES ('old','alice','',%s)", (inserted_date,))
        cursor.execute("INSERT INTO messages VALUES ('old',1,'user','Old row',%s,'{}')", (inserted_date,))
        cursor.execute("INSERT INTO rag_unanswered_questions (conversation_id,request_id,question,search_query,retrieved_pages,failure_reason,created_at) "
                       "VALUES ('old','rid','Question','Query','[]','test',%s)",
                       ("2026-10-23 12:43:14" if layout == "simple" else "2026-10-23 07:13:14",))
        if layout == "paired":
            for table, fields in DATE_COLUMNS.items():
                for column, kind in fields.items():
                    backing = f"{column}_{kind}"
                    definition = "DOUBLE NOT NULL" if kind == "epoch" else "DATETIME NOT NULL DEFAULT (UTC_TIMESTAMP())"
                    value = f"DATE_ADD('1970-01-01 00:00:00', INTERVAL FLOOR(`{backing}`) SECOND)" if kind == "epoch" else f"`{backing}`"
                    cursor.execute(f"ALTER TABLE `{table}` CHANGE COLUMN `{column}` `{backing}` {definition}, "
                                   f"ADD COLUMN `{column}` DATETIME GENERATED ALWAYS AS (DATE_ADD({value}, INTERVAL 330 MINUTE)) VIRTUAL")
        cursor.execute("CREATE VIEW messages_report AS SELECT created_at FROM messages")
    settings = Settings(db_backend="mysql", mysql_host="127.0.0.1", mysql_port=port,
                        mysql_user="root", mysql_password=password, mysql_database=database)
    store = MySQLStore(settings)
    try:
        with store.connect() as db:
            columns = inspect_columns(db)
            assert migration_statements(columns, inspect_indexes(db)) == []
            assert not any(column.endswith(("_epoch", "_utc")) or column.startswith("__migrate_") for _, column in columns)
            assert len([1 for table, fields in DATE_COLUMNS.items() for column in fields if (table, column) in columns]) == len(DATE_COLUMNS)
        with raw.cursor() as cursor:
            cursor.execute("SELECT created_at FROM messages")
            assert cursor.fetchone()[0] == datetime(2026, 10, 23, 12, 43, 14, 887433)
            cursor.execute("SELECT created_at FROM rag_unanswered_questions")
            assert cursor.fetchone()[0] == datetime(2026, 10, 23, 12, 43, 14)
            cursor.execute("SHOW INDEX FROM conversations WHERE Key_name='conversations_user_created'")
            assert [row[4] for row in cursor.fetchall()] == ["user_id", "created_at", "id"]
            cursor.execute("SELECT created_at FROM messages_report")
            assert cursor.fetchone()[0] == datetime(2026, 10, 23, 12, 43, 14, 887433)
        assert store.conversation("alice", "old")["created_at"] == pytest.approx(epoch, abs=1e-6)
        clock = [epoch + 100]
        monkeypatch.setattr("time.time", lambda: clock[0])
        store.save_turn("old", "new", "Hello", {"answer": "Hi", "sources": [], "status": "conversational"})
        assert store.history_page("old")[-1]["created_at"] == pytest.approx(clock[0], abs=1e-6)
        store.cache_answer("key", {"answer": "Saved"})
        assert store.cached_answer("key", 10) == {"answer": "Saved"}
        clock[0] += 20
        assert store.cached_answer("key", 10) is None
        assert store.cached_answer("key", -1) == {"answer": "Saved"}
        ids = []
        for _ in range(3):
            clock[0] += .1
            ids.append(store.conversation("pagination")["id"])
        first = store.list_conversations("pagination", limit=2)
        second = store.list_conversations("pagination", limit=2, cursor=first["next_cursor"])
        assert [row["id"] for row in first["conversations"] + second["conversations"]] == ids[::-1]
        store.check_rate("visitor", 1)
        with pytest.raises(RateLimited):
            store.check_rate("visitor", 1)
        clock[0] += 61
        store.check_rate("visitor", 1)
        with store.lease("lock"):
            with pytest.raises(BusyConversation):
                with store.lease("lock"):
                    pass
        with store.lease("lock"):
            pass
        store.save_search("search-key", ["chunk"])
        assert store.search_cached("search-key", 10) == ["chunk"]
        store.save_http_state("https://example.com", "https://example.com", None, None, [])
        assert store.http_state("https://example.com")["checked_at"] == pytest.approx(clock[0], abs=1e-6)
        chunk_id = str(uuid.uuid4())
        store.replace_document("https://example.com/guide", "Guide", "a" * 64, "test:3", [
            {"id": chunk_id, "position": 0, "text": "An indexed guide"},
        ])
        assert store.document("https://example.com/guide")["indexed_at"] == pytest.approx(clock[0], abs=1e-6)
        assert store.get_chunks([chunk_id])[0]["indexed_at"] == pytest.approx(clock[0], abs=1e-6)
        store.cache_vectors([("b" * 64, [0.1, 0.2])])
        assert store.cached_vector("b" * 64) == [0.1, 0.2]
        clock[0] += 2 * 86400
        assert store.cleanup_cache(1)["embedding_cache"] == 1
        with request_timings("date-timing") as trace:
            writer = RequestTimingWriter(store, trace, "old")
            trace.update(status_code=200, outcome="answered")
        writer.worker_finished()
        writer.http_finished()
        writer.save()
        with raw.cursor() as cursor:
            cursor.execute("SELECT created_at FROM rag_request_timings WHERE request_id='date-timing'")
            assert cursor.fetchone()[0] == datetime.fromisoformat(writer.identity["created_at"]) + timedelta(minutes=330)
            cursor.execute("SET time_zone='-07:00'")
            cursor.execute("INSERT INTO rag_unanswered_questions (conversation_id,request_id,question,search_query,retrieved_pages,failure_reason) "
                           "VALUES ('old','future','Question','Query','[]','test')")
            cursor.execute("SELECT ABS(TIMESTAMPDIFF(SECOND,created_at,DATE_ADD(UTC_TIMESTAMP(6),INTERVAL 330 MINUTE)))<5 FROM rag_unanswered_questions WHERE request_id='future'")
            assert cursor.fetchone()[0] == 1
    finally:
        store.close()
        raw.close()
    reopened = MySQLStore(settings)
    reopened.close()  # A restart must not shift dates a second time.
