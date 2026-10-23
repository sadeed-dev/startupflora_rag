-- Run python migrate_dates.py --apply with writers stopped before installing these views.
-- Dates in the base tables already display IST. Never add a second timezone offset.
USE startupflora_rag;

CREATE OR REPLACE VIEW `documents_report` AS SELECT `source`, `title`, `hash`, `indexed_at`, `embedding_signature` FROM `documents`;

CREATE OR REPLACE VIEW `chunks_report` AS SELECT `id`, `source`, `position`, `title`, `text`, `indexed_at` FROM `chunks`;

CREATE OR REPLACE VIEW `embedding_cache_report` AS SELECT `key`, `vector`, `created_at` FROM `embedding_cache`;

CREATE OR REPLACE VIEW `conversations_report` AS SELECT `id`, `user_id`, `summary`, `summarized_seq`, `created_at` FROM `conversations`;

CREATE OR REPLACE VIEW `messages_report` AS SELECT `conversation_id`, `seq`, `role`, `content`, `created_at`, `metadata` FROM `messages`;

CREATE OR REPLACE VIEW `leases_report` AS SELECT `conversation_id`, `token`, `expires` FROM `leases`;

CREATE OR REPLACE VIEW `requests_report` AS SELECT `user_id`, `at` FROM `requests`;

CREATE OR REPLACE VIEW `rate_locks_report` AS SELECT `user_id`, `touched` FROM `rate_locks`;

CREATE OR REPLACE VIEW `search_cache_report` AS SELECT `key`, `chunk_ids`, `created_at` FROM `search_cache`;

CREATE OR REPLACE VIEW `http_cache_report` AS SELECT `source`, `canonical_source`, `etag`, `last_modified`, `links`, `checked_at` FROM `http_cache`;

CREATE OR REPLACE VIEW `verified_answer_cache_report` AS SELECT `cache_key`, `response`, `created_at` FROM `verified_answer_cache`;

CREATE OR REPLACE VIEW `rag_request_timings_report` AS SELECT `attempt_id`, `request_id`, `conversation_id`, `created_at`, `status_code`, `outcome`, `answer_cache`, `total_ms`, `provider_calls`, `input_tokens`, `output_tokens`, `total_tokens`, `unreported_usage_calls`, `usage_complete`, `stages_ms`, `calls`, `token_usage` FROM `rag_request_timings`;

CREATE OR REPLACE VIEW `rag_unanswered_questions_report` AS SELECT `conversation_id`, `request_id`, `question`, `search_query`, `retrieved_pages`, `failure_reason`, `created_at` FROM `rag_unanswered_questions`;

CREATE OR REPLACE VIEW chat_messages_report AS SELECT m.conversation_id,c.user_id AS visitor_id,m.seq,m.role,m.content,m.created_at AS createdAt FROM messages m JOIN conversations c ON c.id=m.conversation_id;
CREATE OR REPLACE VIEW chat_conversations_report AS SELECT id,user_id,created_at AS createdAt FROM conversations;
SELECT * FROM messages_report ORDER BY created_at DESC LIMIT 100;
