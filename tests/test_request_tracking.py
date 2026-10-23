"""Durable per-attempt telemetry, API retries and fail-open tracking writes."""

import json
import uuid

from fastapi.testclient import TestClient
from test_rag import rig as rig

from rag.request_tracking import RequestTimingWriter
from rag.timing import record_token_usage, request_timings, stage


def test_tracking_preserves_reported_totals_and_updates_same_attempt(rig):
    with request_timings("request-with-'quote") as trace:
        writer = RequestTimingWriter(rig.store, trace, "test-conversation")
        with stage("generation"):
            record_token_usage("gemini", "test-model", "chat", {
                "input_tokens": 10, "output_tokens": 2, "total_tokens": 15,
            })
        trace.update(status_code=504, outcome="timeout")
    writer.http_finished()
    writer.save()
    with rig.store.connect() as db:
        row = db.execute("SELECT * FROM rag_request_timings").fetchone()
    assert row["usage_complete"] == 0
    assert row["total_tokens"] == 15
    trace["token_usage"].append({
        "provider": "openai", "model": "test-embedding", "operation": "embedding",
        "usage_reported": False, "input_tokens": None, "output_tokens": None,
        "total_tokens": None, "status": "error",
    })
    writer.worker_finished()
    with rig.store.connect() as db:
        rows = db.execute("SELECT * FROM rag_request_timings").fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["status_code"] == 504 and row["usage_complete"] == 1
    assert row["provider_calls"] == 2 and row["unreported_usage_calls"] == 1
    assert row["input_tokens"] == 10 and row["output_tokens"] == 2
    assert len(json.loads(row["token_usage"])) == 2
    assert json.loads(row["calls"])["generation"] == 1


def test_http_attempts_cache_hits_replays_and_errors_are_separate_rows(rig, monkeypatch):
    from rag import api

    monkeypatch.setattr(api, "runtime", lambda: (rig.s, rig.store, rig.models, rig.r, rig.agent))
    monkeypatch.setattr(api, "close_runtime", lambda: None)
    headers = {"Authorization": "Bearer " + rig.s.backend_key, "X-User-ID": "alice"}
    cid = rig.store.conversation("alice")["id"]
    body = {"conversation_id": cid, "request_id": str(uuid.uuid4()), "message": "DPIIT?"}
    with TestClient(api.app) as client:
        assert client.post("/chat", headers=headers, json=body).status_code == 200
        assert client.post("/chat", headers=headers, json=body).status_code == 200
        assert client.post("/chat", headers=headers, json={**body, "request_id": str(uuid.uuid4())}).status_code == 200
        assert client.post("/chat", headers=headers, json={**body, "message": "Different?"}).status_code == 409
    with rig.store.connect() as db:
        rows = db.execute("SELECT * FROM rag_request_timings").fetchall()
    assert len(rows) == 4
    assert len({row["attempt_id"] for row in rows}) == 4
    assert {row["answer_cache"] for row in rows} >= {"miss", "hit", "request_replay"}
    assert sorted(row["status_code"] for row in rows) == [200, 200, 200, 409]
    assert all(row["usage_complete"] == 1 for row in rows)
    assert all(row["conversation_id"] == cid for row in rows)


def test_tracking_failure_does_not_fail_the_chat(rig, monkeypatch, caplog):
    from rag import api

    def unavailable(snapshot):
        raise RuntimeError("private database failure details")

    monkeypatch.setattr(rig.store, "save_request_timing", unavailable)
    monkeypatch.setattr(api, "runtime", lambda: (rig.s, rig.store, rig.models, rig.r, rig.agent))
    monkeypatch.setattr(api, "close_runtime", lambda: None)
    headers = {"Authorization": "Bearer " + rig.s.backend_key, "X-User-ID": "alice"}
    cid = rig.store.conversation("alice")["id"]
    with TestClient(api.app) as client:
        response = client.post("/chat", headers=headers, json={
            "conversation_id": cid, "request_id": str(uuid.uuid4()), "message": "hello",
        })
    assert response.status_code == 200
    assert "rag_request_tracking_write_failed" in caplog.text
    assert "private database failure details" not in caplog.text
