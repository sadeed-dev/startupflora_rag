"""Verify the fast path and HTTP deadline without live provider calls."""

import threading
import time
import uuid

from fastapi.testclient import TestClient

from test_rag import rig  # noqa: F401


def test_local_qdrant_connection_avoids_ipv6_without_changing_remote_hosts():
    from rag.retrieval import qdrant_connection_url

    assert qdrant_connection_url("http://localhost:6333") == "http://127.0.0.1:6333"
    assert qdrant_connection_url("http://localhost:6333/path") == "http://127.0.0.1:6333/path"
    assert qdrant_connection_url("https://localhost.example:6333") == "https://localhost.example:6333"


def test_fast_followup_uses_one_generation_without_utilities(rig):
    rig.s.fast_mode = True
    rig.s.verify_answers = False
    cid = rig.store.conversation("alice")["id"]
    rig.agent.respond("alice", cid, "DPIIT documents?", str(uuid.uuid4()))
    before = rig.models.chat.calls
    result = rig.agent.respond("alice", cid, "What certificate?", str(uuid.uuid4()))
    assert result["status"] == "answered"
    assert rig.models.chat.calls == before + 1
    assert rig.models.chat.seen_tools == ["Answer"]
    assert rig.models.utilities == []
    assert result["sources"]


def test_fast_empty_retrieval_skips_generation(rig, monkeypatch):
    rig.s.fast_mode = True
    monkeypatch.setattr(rig.r, "search", lambda query: [])
    cid = rig.store.conversation("alice")["id"]
    result = rig.agent.respond("alice", cid, "Unknown service?", str(uuid.uuid4()))
    assert result["status"] == "insufficient_evidence"
    assert rig.models.chat.calls == 0
    assert rig.models.utilities == []


def test_deadline_returns_before_provider_and_does_not_save_late_answer(rig, monkeypatch):
    from rag import api

    rig.s.response_timeout_seconds = 0.1
    rig.s.fast_mode = True
    rig.s.verify_answers = False
    release, finished = threading.Event(), threading.Event()
    invoke = rig.models.chat.invoke
    respond = rig.agent.respond

    def slow_model(messages):
        release.wait(2)
        return invoke(messages)

    def tracked_respond(*args, **kwargs):
        try:
            return respond(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(rig.models.chat, "invoke", slow_model)
    monkeypatch.setattr(rig.agent, "respond", tracked_respond)
    monkeypatch.setattr(api, "runtime", lambda: (rig.s, rig.store, rig.models, rig.r, rig.agent))
    monkeypatch.setattr(api, "close_runtime", lambda: None)
    cid = rig.store.conversation("alice")["id"]
    headers = {"Authorization": "Bearer " + rig.s.backend_key, "X-User-ID": "alice"}
    with TestClient(api.app) as client:
        try:
            start = time.monotonic()
            response = client.post("/chat", headers=headers, json={
                "conversation_id": cid, "request_id": str(uuid.uuid4()), "message": "DPIIT?",
            })
            assert response.status_code == 504
            assert time.monotonic() - start < 1
        finally:
            release.set()
            assert finished.wait(2)
    assert rig.store.history_page(cid) == []
    tracking_deadline = time.monotonic() + 2
    while True:
        with rig.store.connect() as db:
            row = db.execute("SELECT * FROM rag_request_timings").fetchone()
        if row["usage_complete"] or time.monotonic() >= tracking_deadline:
            break
        time.sleep(0.01)
    assert row["status_code"] == 504
    assert row["usage_complete"] == 1
