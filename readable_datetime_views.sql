-- Run this complete file in MySQL Workbench with the RAG database selected.
-- Open the *_report views for readable dates in India Standard Time (UTC+05:30).
-- Application tables retain Unix seconds for cache expiry, locking and pagination.
-- Default database; change this line if MYSQL_DATABASE uses a different name.
USE startupflora_rag;

CREATE OR REPLACE VIEW chat_messages_report AS
SELECT
    m.conversation_id,
    c.user_id AS visitor_id,
    m.seq,
    m.role,
    m.content,
    DATE_FORMAT(
        DATE_ADD('1970-01-01 00:00:00', INTERVAL FLOOR(m.created_at) SECOND)
        + INTERVAL 330 MINUTE,
        '%Y-%m-%d %H:%i:%s'
    ) AS createdAt
FROM messages AS m
JOIN conversations AS c ON c.id = m.conversation_id;

CREATE OR REPLACE VIEW chat_conversations_report AS
SELECT
    id,
    user_id,
    DATE_FORMAT(
        DATE_ADD('1970-01-01 00:00:00', INTERVAL FLOOR(created_at) SECOND)
        + INTERVAL 330 MINUTE,
        '%Y-%m-%d %H:%i:%s'
    ) AS createdAt
FROM conversations;



-- documents: indexed_at is displayed as an IST datetime in this view.
CREATE OR REPLACE VIEW `documents_report` AS
SELECT
    `source`,
    `title`,
    `hash`,
    `embedding_signature`,
    DATE_ADD(
        DATE_ADD('1970-01-01 00:00:00', INTERVAL FLOOR(`indexed_at`) SECOND),
        INTERVAL 330 MINUTE
    ) AS `indexed_at`
FROM `documents`;

-- chunks: indexed_at is displayed as an IST datetime in this view.
CREATE OR REPLACE VIEW `chunks_report` AS
SELECT
    `id`,
    `source`,
    `position`,
    `title`,
    `text`,
    DATE_ADD(
        DATE_ADD('1970-01-01 00:00:00', INTERVAL FLOOR(`indexed_at`) SECOND),
        INTERVAL 330 MINUTE
    ) AS `indexed_at`
FROM `chunks`;

-- embedding_cache: created_at is displayed as an IST datetime in this view.
CREATE OR REPLACE VIEW `embedding_cache_report` AS
SELECT
    `key`,
    `vector`,
    DATE_ADD(
        DATE_ADD('1970-01-01 00:00:00', INTERVAL FLOOR(`created_at`) SECOND),
        INTERVAL 330 MINUTE
    ) AS `created_at`
FROM `embedding_cache`;

-- conversations: created_at is displayed as an IST datetime in this view.
CREATE OR REPLACE VIEW `conversations_report` AS
SELECT
    `id`,
    `user_id`,
    `summary`,
    `summarized_seq`,
    DATE_ADD(
        DATE_ADD('1970-01-01 00:00:00', INTERVAL FLOOR(`created_at`) SECOND),
        INTERVAL 330 MINUTE
    ) AS `created_at`
FROM `conversations`;

-- messages: created_at is displayed as an IST datetime in this view.
CREATE OR REPLACE VIEW `messages_report` AS
SELECT
    `conversation_id`,
    `seq`,
    `role`,
    `content`,
    `metadata`,
    DATE_ADD(
        DATE_ADD('1970-01-01 00:00:00', INTERVAL FLOOR(`created_at`) SECOND),
        INTERVAL 330 MINUTE
    ) AS `created_at`
FROM `messages`;

-- http_cache: checked_at is displayed as an IST datetime in this view.
CREATE OR REPLACE VIEW `http_cache_report` AS
SELECT
    `source`,
    `canonical_source`,
    `etag`,
    `last_modified`,
    `links`,
    DATE_ADD(
        DATE_ADD('1970-01-01 00:00:00', INTERVAL FLOOR(`checked_at`) SECOND),
        INTERVAL 330 MINUTE
    ) AS `checked_at`
FROM `http_cache`;

-- search_cache: created_at is displayed as an IST datetime in this view.
CREATE OR REPLACE VIEW `search_cache_report` AS
SELECT
    `key`,
    `chunk_ids`,
    DATE_ADD(
        DATE_ADD('1970-01-01 00:00:00', INTERVAL FLOOR(`created_at`) SECOND),
        INTERVAL 330 MINUTE
    ) AS `created_at`
FROM `search_cache`;

-- leases: expires is displayed as an IST datetime in this view.
CREATE OR REPLACE VIEW `leases_report` AS
SELECT
    `conversation_id`,
    `token`,
    DATE_ADD(
        DATE_ADD('1970-01-01 00:00:00', INTERVAL FLOOR(`expires`) SECOND),
        INTERVAL 330 MINUTE
    ) AS `expires`
FROM `leases`;

-- requests: at is displayed as an IST datetime in this view.
CREATE OR REPLACE VIEW `requests_report` AS
SELECT
    `user_id`,
    DATE_ADD(
        DATE_ADD('1970-01-01 00:00:00', INTERVAL FLOOR(`at`) SECOND),
        INTERVAL 330 MINUTE
    ) AS `at`
FROM `requests`;

-- rate_locks: touched is displayed as an IST datetime in this view.
CREATE OR REPLACE VIEW `rate_locks_report` AS
SELECT
    `user_id`,
    DATE_ADD(
        DATE_ADD('1970-01-01 00:00:00', INTERVAL FLOOR(`touched`) SECOND),
        INTERVAL 330 MINUTE
    ) AS `touched`
FROM `rate_locks`;

-- This cache is created lazily, so install its view only if the table exists.
SET @cache_report_sql = IF(
    EXISTS(SELECT 1 FROM information_schema.TABLES
           WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'verified_answer_cache'),
    'CREATE OR REPLACE VIEW verified_answer_cache_report AS SELECT cache_key, response, DATE_ADD(DATE_ADD(''1970-01-01 00:00:00'', INTERVAL FLOOR(created_at) SECOND), INTERVAL 330 MINUTE) AS created_at FROM verified_answer_cache',
    'SELECT ''verified_answer_cache is not created yet; rerun after the first cached answer'' AS info'
);
PREPARE cache_report_statement FROM @cache_report_sql;
EXECUTE cache_report_statement;
DEALLOCATE PREPARE cache_report_statement;

-- Refresh the Views section in Workbench after running this file.
-- Timing rows store UTC DATETIME values (not Unix seconds).
-- Convert only for display, including existing rows; DATETIME has no session timezone.
CREATE OR REPLACE VIEW `rag_request_timings_report` AS
SELECT
    `attempt_id`,
    `request_id`,
    `conversation_id`,
    DATE_ADD(`created_at`, INTERVAL 330 MINUTE) AS `created_at`,
    `status_code`,
    `outcome`,
    `answer_cache`,
    `total_ms`,
    `provider_calls`,
    `input_tokens`,
    `output_tokens`,
    `total_tokens`,
    `unreported_usage_calls`,
    `usage_complete`,
    `stages_ms`,
    `calls`,
    `token_usage`
FROM `rag_request_timings`;

SELECT * FROM rag_request_timings_report ORDER BY created_at DESC LIMIT 100;
SELECT * FROM documents_report ORDER BY indexed_at DESC LIMIT 100;
SELECT * FROM messages_report ORDER BY created_at DESC LIMIT 100;
SELECT * FROM conversations_report ORDER BY created_at DESC LIMIT 100;
