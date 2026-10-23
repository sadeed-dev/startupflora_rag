import json
import logging

import pytest

from rag.timing import request_timings, stage


def test_nested_traces_are_isolated_and_log_on_failure(caplog):
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        with request_timings("outer") as outer:
            with stage("database_query"):
                pass
            with pytest.raises(RuntimeError):
                with request_timings("inner") as inner:
                    with stage("verification"):
                        raise RuntimeError("private failure text")
            with stage("database_query"):
                pass
    assert outer["calls"] == {"database_query": 2}
    assert inner["calls"] == {"verification": 1}
    assert "private failure text" not in caplog.text
    logs = [json.loads(record.message.split("rag_request_timing ", 1)[1]) for record in caplog.records]
    assert [item["request_id"] for item in logs] == ["inner", "outer"]
    assert all(item["total_ms"] >= 0 for item in logs)
