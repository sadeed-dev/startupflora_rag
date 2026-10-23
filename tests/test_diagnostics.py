"""Offline diagnostic persistence checks; never contact a real database."""

import json
from unittest.mock import MagicMock, patch

from rag.config import Settings
from rag.diagnostics import record_unanswered


def test_unanswered_insert_binds_question_and_page_metadata():
    connection = MagicMock()
    connection.__enter__.return_value = connection
    cursor = connection.cursor.return_value.__enter__.return_value
    diagnostic = {"search_query": "funding", "retrieved_pages": [{"chunk_id": "123", "page_url": "https://example.com"}], "failure_reason": "unsupported_claims"}
    with patch("rag.diagnostics.pymysql.connect", return_value=connection):
        record_unanswered(Settings(), "cid", "rid", "what's funding?", diagnostic)
    sql, params = cursor.execute.call_args.args
    assert "what's funding?" not in sql
    assert params[2] == "what's funding?"
    assert json.loads(params[4]) == diagnostic["retrieved_pages"]
    assert params[5] == "unsupported_claims"
    connection.commit.assert_called_once()


def test_mysql_outage_does_not_raise_or_log_question(caplog):
    with patch("rag.diagnostics.pymysql.connect", side_effect=RuntimeError("secret")):
        record_unanswered(Settings(), "cid", "rid", "private question", {})
    assert "unanswered_log_failed" in caplog.text
    assert "secret" not in caplog.text and "private question" not in caplog.text
