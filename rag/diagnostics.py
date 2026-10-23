"""Persist reviewable retrieval failures separately from chat storage."""

import json
import logging

import pymysql

log = logging.getLogger(__name__)

SCHEMA = """CREATE TABLE IF NOT EXISTS rag_unanswered_questions (
    conversation_id VARCHAR(36) NOT NULL,
    request_id VARCHAR(36) NOT NULL,
    question TEXT NOT NULL,
    search_query TEXT NOT NULL,
    retrieved_pages JSON NOT NULL,
    failure_reason VARCHAR(64) NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT (DATE_ADD(UTC_TIMESTAMP(6), INTERVAL 330 MINUTE)),
    PRIMARY KEY (conversation_id, request_id),
    INDEX idx_unanswered_created (created_at)
) CHARACTER SET utf8mb4"""


def record_unanswered(settings, cid, request_id, question, diagnostic):
    """Write only to configured MySQL; audit outages must not fail a chat turn."""
    try:
        tls = {}
        if settings.mysql_ssl_ca:
            tls = dict(ssl_ca=settings.mysql_ssl_ca, ssl_verify_cert=True, ssl_verify_identity=True)
        with pymysql.connect(
            host=settings.mysql_host, port=settings.mysql_port,
            user=settings.mysql_user, password=settings.mysql_password,
            database=settings.mysql_database, charset="utf8mb4",
            connect_timeout=3, read_timeout=5, write_timeout=5, **tls,
        ) as db:
            with db.cursor() as cursor:
                cursor.execute(SCHEMA)
                cursor.execute(
                    """INSERT INTO rag_unanswered_questions
                    (conversation_id,request_id,question,search_query,retrieved_pages,failure_reason)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE request_id=VALUES(request_id)""",
                    (cid, request_id, question, diagnostic["search_query"],
                     json.dumps(diagnostic["retrieved_pages"], ensure_ascii=False),
                     diagnostic["failure_reason"]),
                )
            db.commit()
    except Exception as exc:
        log.error("unanswered_log_failed request_id=%s error_type=%s", request_id, type(exc).__name__)
