"""Regression checks for accepted-answer reuse without provider calls."""

import uuid

import pytest

from test_rag import rig as rig


def ask(rig, cid, question="DPIIT documents?", user="alice"):
    return rig.agent.respond(user, cid, question, str(uuid.uuid4()))


def test_exact_repeat_reuses_answer_but_saves_new_turn(rig):
    cid = rig.store.conversation("alice")["id"]
    first = ask(rig, cid)
    calls = (rig.models.chat.calls, len(rig.models.utilities), rig.models.provider.calls)
    second = ask(rig, cid)
    assert first["answer"] == second["answer"]
    assert first["request_id"] != second["request_id"]
    assert calls == (rig.models.chat.calls, len(rig.models.utilities), rig.models.provider.calls)
    assert len(rig.store.recent(cid, 10)) == 4


def test_other_users_are_isolated_but_standalone_repeat_ignores_unrelated_turns(rig):
    cid = rig.store.conversation("alice")["id"]
    ask(rig, cid)
    before = rig.models.chat.calls
    ask(rig, rig.store.conversation("bob")["id"], user="bob")
    assert rig.models.chat.calls > before
    ask(rig, cid, "What about eligibility?")
    before = rig.models.chat.calls
    ask(rig, cid)
    assert rig.models.chat.calls == before


def test_expiry_and_corpus_revision_force_fresh_answer(rig):
    rig.s.answer_cache_seconds = 86400
    cid = rig.store.conversation("alice")["id"]
    ask(rig, cid)
    with rig.store.connect() as db:
        db.execute("UPDATE verified_answer_cache SET created_at=0")
    before = rig.models.chat.calls
    ask(rig, cid)
    assert rig.models.chat.calls > before
    with rig.store.connect() as db:
        db.execute("UPDATE corpus_meta SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")
    before = rig.models.chat.calls
    ask(rig, cid)
    assert rig.models.chat.calls > before


def test_persistent_cache_survives_old_timestamp_new_chat_and_store_restart(rig):
    from rag.agent import Agent
    from rag.store import Store
    from rag.timing import request_timings

    rig.s.answer_cache_seconds = -1
    first = ask(rig, rig.store.conversation("alice")["id"])
    with rig.store.connect() as db:
        db.execute("UPDATE verified_answer_cache SET created_at=0")
    before = (rig.models.chat.calls, len(rig.models.utilities), rig.models.provider.calls)
    restarted_store = Store(rig.s.db_path)
    restarted = Agent(rig.s, restarted_store, rig.models, rig.r)
    cid = restarted_store.conversation("alice")["id"]
    with request_timings("persistent-hit") as trace:
        result = restarted.respond("alice", cid, "  dpiit   DOCUMENTS?  ", str(uuid.uuid4()))
    assert result["answer"] == first["answer"]
    assert trace["answer_cache"] == "hit"
    assert before == (rig.models.chat.calls, len(rig.models.utilities), rig.models.provider.calls)
    assert len(restarted_store.recent(cid, 10)) == 2


def test_standalone_cache_skips_memory_and_all_providers_after_long_history(rig, monkeypatch):
    cid = rig.store.conversation("alice")["id"]
    ask(rig, cid)
    # An old conversation with a summary previously bypassed the answer cache.
    rig.store.update_summary(cid, "Earlier conversation", 2)
    for i in range(55):
        rig.store.save_turn(cid, str(uuid.uuid4()), f"Unrelated message {i}", {
            "answer": "Context", "sources": [], "status": "clarification",
        })
    def forbidden(*args, **kwargs):
        raise AssertionError("Cache hit must not invoke memory, retrieval, or models")
    monkeypatch.setattr(rig.agent, "memory", forbidden)
    monkeypatch.setattr(rig.r, "search", forbidden)
    monkeypatch.setattr(rig.models, "structured", forbidden)
    monkeypatch.setattr(rig.models.chat, "invoke", forbidden)
    assert ask(rig, cid)["status"] == "answered"


def test_contextual_question_does_not_reuse_standalone_answer(rig):
    from rag.agent import standalone_cache_question

    for question in ("What are its fees?", "What about DPIIT?", "DPIIT for my company?", "DPIIT for our business?"):
        assert not standalone_cache_question(question)
    cid = rig.store.conversation("alice")["id"]
    ask(rig, cid)
    before = rig.models.chat.calls
    ask(rig, cid, "What are its fees?")
    assert rig.models.chat.calls > before


@pytest.mark.parametrize("minimum_calls", [True, False])
def test_success_story_count_repeats_hit_cache_after_other_questions(rig, minimum_calls):
    from rag.timing import request_timings

    rig.s.minimum_calls = minimum_calls
    cid = rig.store.conversation("alice")["id"]
    ask(rig, cid, "GST registration documents?")
    question = "give me count of total success stories"
    with request_timings("stories-first") as first_trace:
        first = ask(rig, cid, question)
    assert first_trace["answer_cache"] == "miss"
    assert first["status"] == "answered"
    ask(rig, cid, "DPIIT documents?")
    before = (rig.models.chat.calls, len(rig.models.utilities), rig.models.provider.calls)
    for i in range(2):
        with request_timings(f"stories-repeat-{i}") as trace:
            repeated = ask(rig, cid, question)
        assert trace["answer_cache"] == "hit"
        assert trace["token_usage"] == []
        assert repeated["answer"] == first["answer"]
        assert before == (rig.models.chat.calls, len(rig.models.utilities), rig.models.provider.calls)


@pytest.mark.parametrize("question", [
    "What are its fees?", "Tell me more", "How much?", "What about eligibility?",
    "How do I apply?", "Give me their success stories", "I have a GST registration",
])
def test_contextual_requests_keep_cache_bypass_reason(rig, question):
    from rag.agent import standalone_cache_question
    from rag.timing import request_timings

    assert not standalone_cache_question(question)
    cid = rig.store.conversation("alice")["id"]
    ask(rig, cid)
    with request_timings("context") as trace:
        ask(rig, cid, question)
    assert trace["answer_cache"] == "bypass"
    assert trace["answer_cache_reason"] == "context_dependent"


def test_verification_disabled_reuses_answer_without_models_or_embeddings(rig):
    from rag.timing import request_timings

    rig.s.verify_answers = False
    cid = rig.store.conversation("alice")["id"]
    with request_timings("first") as first_trace:
        first = ask(rig, cid)
    before = (rig.models.chat.calls, len(rig.models.utilities), rig.models.provider.calls)
    with request_timings("repeat") as second_trace:
        second = ask(rig, cid)
    assert first["answer"] == second["answer"]
    assert before == (rig.models.chat.calls, len(rig.models.utilities), rig.models.provider.calls)
    assert first_trace["answer_cache"] == "miss"
    assert second_trace["answer_cache"] == "hit"
    assert second_trace["token_usage"] == []


def test_verification_or_model_change_does_not_reuse_previous_answer(rig):
    rig.s.verify_answers = False
    cid = rig.store.conversation("alice")["id"]
    ask(rig, cid)
    before = rig.models.chat.calls
    rig.s.verify_answers = True
    ask(rig, cid)
    assert rig.models.chat.calls > before
    before = rig.models.chat.calls
    rig.s.chat_model = "different-model"
    ask(rig, cid)
    assert rig.models.chat.calls > before


def test_disabled_cache_and_rejected_answers_are_not_reused(rig):
    rig.s.verify_answers = False
    rig.s.answer_cache_seconds = 0
    cid = rig.store.conversation("alice")["id"]
    ask(rig, cid)
    before = rig.models.chat.calls
    ask(rig, cid)
    assert rig.models.chat.calls > before
    rig.s.answer_cache_seconds = 86400
    rig.s.verify_answers = True
    rig.models.grounding = False
    assert ask(rig, cid)["status"] == "insufficient_evidence"
    before = rig.models.chat.calls
    ask(rig, cid)
    assert rig.models.chat.calls > before
