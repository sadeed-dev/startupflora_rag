"""One durable timing/usage row per HTTP attempt, including late timeout work."""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone

from .timing import without_request_timings

log = logging.getLogger("uvicorn.error")


class RequestTrackingStore:
    """Share the tracking schema and bound writes across SQLite and MySQL."""

    def request_timing_table(self, db):
        """Create only the tracking table and indexes, preserving existing data."""
        mysql = hasattr(self, "s")
        payload_type = "LONGTEXT" if mysql else "TEXT"
        indexes = ", INDEX timing_request_id (request_id), INDEX timing_created_at (created_at)" if mysql else ""
        db.execute(f"""CREATE TABLE IF NOT EXISTS rag_request_timings (
            attempt_id VARCHAR(36) PRIMARY KEY,
            request_id VARCHAR(64) NOT NULL,
            conversation_id VARCHAR(36) NOT NULL,
            created_at DATETIME NOT NULL,
            status_code INTEGER NOT NULL,
            outcome VARCHAR(32) NOT NULL,
            answer_cache VARCHAR(24) NOT NULL,
            total_ms DOUBLE NOT NULL,
            provider_calls INTEGER NOT NULL,
            input_tokens BIGINT NOT NULL,
            output_tokens BIGINT NOT NULL,
            total_tokens BIGINT NOT NULL,
            unreported_usage_calls INTEGER NOT NULL,
            usage_complete INTEGER NOT NULL,
            stages_ms {payload_type} NOT NULL,
            calls {payload_type} NOT NULL,
            token_usage {payload_type} NOT NULL
            {indexes}
        )""")
        if not mysql:
            db.execute("CREATE INDEX IF NOT EXISTS timing_request_id ON rag_request_timings(request_id)")
            db.execute("CREATE INDEX IF NOT EXISTS timing_created_at ON rag_request_timings(created_at)")

    def save_request_timing(self, snapshot):
        """Upsert one attempt; totals sum only known counts and flag missing usage."""
        usage = snapshot["token_usage"]
        fields = (
            "attempt_id", "request_id", "conversation_id", "created_at", "status_code",
            "outcome", "answer_cache", "total_ms", "provider_calls", "input_tokens",
            "output_tokens", "total_tokens", "unreported_usage_calls", "usage_complete",
            "stages_ms", "calls", "token_usage",
        )
        values = (
            snapshot["attempt_id"], snapshot["request_id"], snapshot["conversation_id"],
            snapshot["created_at"], snapshot["status_code"], snapshot["outcome"],
            snapshot["answer_cache"], snapshot["total_ms"], len(usage),
            *(sum(entry.get(name) or 0 for entry in usage)
              for name in ("input_tokens", "output_tokens", "total_tokens")),
            sum(not entry["usage_reported"] for entry in usage), int(snapshot["usage_complete"]),
            *(json.dumps(snapshot[name], ensure_ascii=False) for name in ("stages_ms", "calls", "token_usage")),
        )
        mysql = hasattr(self, "s")
        marker = "%s" if mysql else "?"
        sql = f"INSERT INTO rag_request_timings ({','.join(fields)}) VALUES ({','.join([marker] * len(fields))})"
        if mysql:
            sql += " ON DUPLICATE KEY UPDATE " + ",".join(f"{name}=VALUES({name})" for name in fields[1:])
        else:
            sql += " ON CONFLICT(attempt_id) DO UPDATE SET " + ",".join(f"{name}=excluded.{name}" for name in fields[1:])
        with self.connect() as db:
            db.execute(sql, values)


class RequestTimingWriter:
    """Serialize initial and late snapshots without delaying the HTTP response."""

    def __init__(self, store, trace, conversation_id):
        self.store, self.trace = store, trace
        self.lock = threading.Lock()
        self.worker_done = False
        self.http_done = False
        self.identity = {
            "attempt_id": str(uuid.uuid4()), "conversation_id": str(conversation_id),
            # Keep storage in UTC; rag_request_timings_report displays all rows in IST.
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }

    def worker_finished(self):
        """Update late usage after an abandoned synchronous worker finishes."""
        with self.lock:
            self.worker_done = True
            http_done = self.http_done
        if http_done:
            self.save()

    def http_finished(self):
        """Mark the finalized HTTP duration/status ready for persistence."""
        with self.lock:
            self.http_done = True

    def save(self):
        """Keep telemetry failures separate from chat outcomes; never log raw errors."""
        with self.lock:
            snapshot = json.loads(json.dumps(self.trace))
            snapshot.update(self.identity, usage_complete=self.worker_done)
            snapshot["stages_ms"] = {key: round(value, 1) for key, value in snapshot["stages_ms"].items()}
            try:
                with without_request_timings():
                    self.store.save_request_timing(snapshot)
            except Exception as exc:
                log.error("rag_request_tracking_write_failed attempt_id=%s error_type=%s",
                          self.identity["attempt_id"], type(exc).__name__)
