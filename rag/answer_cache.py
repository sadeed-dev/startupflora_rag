"""Durable accepted-answer cache shared by the SQLite and MySQL stores.

The historical table name is retained; cache keys include verification settings.
"""

import json
import time


class AnswerCache:
    """Use bound values and keep cached responses separate from request history."""

    def answer_context_anchor(self, cid, meaningful):
        """Find the last non-repeat user turn in bounded, indexed history pages.

        Its sequence identifies the immutable user-history prefix, even when the
        recent model context rolls over or a derived summary is rewritten.
        """
        marker = "%s" if hasattr(self, "s") else "?"
        before = None
        with self.connect() as db:
            while True:
                boundary = f" AND seq<{marker}" if before is not None else ""
                params = (cid, before) if before is not None else (cid,)
                rows = db.execute(
                    f"SELECT seq,content FROM messages WHERE conversation_id={marker} "
                    f"AND role='user'{boundary} ORDER BY seq DESC LIMIT 100", params,
                ).fetchall()
                for row in rows:
                    if meaningful(row["content"]):
                        return row["seq"]
                if len(rows) < 100:
                    return 0
                before = rows[-1]["seq"]

    def answer_cache_table(self, db):
        mysql = hasattr(self, "s")
        db.execute("""CREATE TABLE IF NOT EXISTS verified_answer_cache (
            cache_key VARCHAR(64) PRIMARY KEY,
            response TEXT NOT NULL,
            created_at DOUBLE NOT NULL
        )""" if not mysql else """CREATE TABLE IF NOT EXISTS verified_answer_cache (
            cache_key VARCHAR(64) PRIMARY KEY,
            response LONGTEXT NOT NULL,
            created_at DATETIME(6) NOT NULL
        ) CHARACTER SET utf8mb4""")

    def cached_answer(self, key, ttl):
        if ttl == 0:
            return None
        marker = "%s" if hasattr(self, "s") else "?"
        with self.connect() as db:
            self.answer_cache_table(db)
            expiry = f" AND created_at>={marker}" if ttl > 0 else ""
            params = (key, time.time() - ttl) if ttl > 0 else (key,)
            row = db.execute(
                f"SELECT response FROM verified_answer_cache WHERE cache_key={marker}{expiry}", params,
            ).fetchone()
            return json.loads(row[0]) if row else None

    def cache_answer(self, key, result):
        mysql = hasattr(self, "s")
        with self.connect() as db:
            self.answer_cache_table(db)
            sql = ("REPLACE INTO verified_answer_cache (cache_key,response,created_at) VALUES (%s,%s,%s)"
                   if mysql else "INSERT OR REPLACE INTO verified_answer_cache (cache_key,response,created_at) VALUES (?,?,?)")
            db.execute(sql, (key, json.dumps(result, ensure_ascii=False), time.time()))
