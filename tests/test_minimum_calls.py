"""Minimum-call contracts: real retrieval/storage and mocked providers, never paid APIs."""

import json
import uuid

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from test_rag import rig as rig

from rag.config import Settings
from rag.gemini import GEMINI_BASE_URL, GeminiChat
from rag.llm import ModelFailure, Models
from rag.minimum import resolve_followup
from rag.schemas import Answer
from rag.store import BusyConversation, RateLimited, RequestConflict
from rag.text import fit_messages, token_count
from rag.timing import record_token_usage, request_timings, stage


@pytest.fixture
def minimum(rig, monkeypatch):
    # Failure tests must never connect to the configured audit database.
    monkeypatch.setattr("rag.agent.record_unanswered", lambda *args: None)
    rig.s.minimum_calls = True
    rig.s.model_input_tokens = 8000
    rig.s.context_tokens = 2500
    rig.s.history_tokens = 1000
    rig.s.recent_messages = 6
    rig.s.candidates = 8
    rig.s.final_chunks = 3
    # Deliberately leave legacy utility flags enabled: the mode must override them.
    return rig


def ask(rig, cid, question="DPIIT documents?", request_id=None, user="alice"):
    return rig.agent.respond(user, cid, question, request_id or str(uuid.uuid4()))


def test_fresh_question_and_completed_retry(minimum):
    cid = minimum.store.conversation("alice")["id"]
    request_id = str(uuid.uuid4())
    before = minimum.models.provider.calls
    with request_timings(request_id) as trace:
        result = ask(minimum, cid, request_id=request_id)
    assert result["status"] == "answered"
    assert minimum.models.provider.calls == before + 1
    assert minimum.models.chat.calls == 1
    assert minimum.models.chat.seen_tools == ["Answer"]
    assert minimum.models.utilities == []
    assert trace["calls"]["retrieval"] == trace["calls"]["generation"] == 1
    assert trace["automatic_retry_limit"] == 0
    assert ask(minimum, cid, request_id=request_id) == result
    assert minimum.models.chat.calls == 1
    assert minimum.models.provider.calls == before + 1
    with pytest.raises(RequestConflict):
        ask(minimum, cid, "Different question", request_id=request_id)


def test_cached_search_and_embedding_skip_provider(minimum):
    minimum.s.answer_cache_seconds = 0
    before = minimum.models.provider.calls
    ask(minimum, minimum.store.conversation("alice")["id"])
    cid = minimum.store.conversation("bob")["id"]
    ask(minimum, cid, user="bob")
    assert minimum.models.provider.calls == before + 1
    minimum.s.search_cache_seconds = 0
    ask(minimum, cid, user="bob")
    assert minimum.models.provider.calls == before + 1
    assert minimum.models.chat.calls == 3


def test_social_and_unresolved_followup_need_no_models(minimum):
    cid = minimum.store.conversation("alice")["id"]
    before = minimum.models.provider.calls
    assert ask(minimum, cid, "Hello")["status"] == "conversational"
    assert ask(minimum, cid, "What are its fees?")["status"] == "clarification"
    assert minimum.models.chat.calls == 0
    assert minimum.models.provider.calls == before


def test_followup_uses_recent_user_topic_and_new_topic_drops_history(minimum, monkeypatch):
    queries = []
    search = minimum.r.search

    def track(query):
        queries.append(query)
        return search(query)

    monkeypatch.setattr(minimum.r, "search", track)
    cid = minimum.store.conversation("alice")["id"]
    ask(minimum, cid)
    ask(minimum, cid, "What are its fees?")
    assert "DPIIT documents?" in queries[-1]
    assert "What are its fees?" in queries[-1]
    assert minimum.models.chat.calls == 2 and minimum.models.utilities == []
    ask(minimum, cid, "What is GST registration?")
    assert queries[-1] == "What is GST registration?"
    assert not any(m.type == "ai" for m in minimum.models.chat.last_messages)
    assert "DPIIT documents?" not in str(minimum.models.chat.last_messages)


def test_ambiguous_previous_topics_ask_clarification():
    history = [HumanMessage(content="Compare DPIIT and MSME"), AIMessage(content="Context only")]
    assert resolve_followup("What are its fees?", history)[2]


def test_loan_followup_retains_user_topic():
    history = [HumanMessage(content="I need a loan"), AIMessage(content="Which business?")]
    query, retained, unresolved = resolve_followup("What documents are required?", history)
    assert not unresolved
    assert "I need a loan" in query and retained == history
    query, retained, unresolved = resolve_followup("I need a loan", history)
    assert query == "I need a loan" and retained == [] and not unresolved


@pytest.mark.parametrize("answer,ids,status,error", [
    ("A supported statement.", ["S1"], "answered", "inline_source_ids_mismatch"),
    ("Is this a new business?", [], "answered", "answered_without_sources"),
    ("Statement. [S99]", ["S99"], "answered", "unknown_source_ids"),
    ("See https://example.com [S1]", ["S1"], "answered", "model_generated_url"),
    (" ", [], "clarification", "empty_answer"),
    ("x" * 8001, [], "clarification", "answer_too_long"),
])
def test_citation_diagnostic_identifies_specific_validation_error(
    minimum, answer, ids, status, error
):
    minimum.models.chat.next_tools = [("Answer", {
        "answer": answer, "source_ids": ids, "status": status,
    })]
    with request_timings("citation-test") as trace:
        result = ask(minimum, minimum.store.conversation("alice")["id"])
    validation = trace["answer_diagnostic"]["validation"]
    assert error in validation["errors"]
    assert validation["declared_source_ids"] == ids
    assert validation["available_source_ids"] == ["S1"]
    assert result["status"] == "insufficient_evidence"
    assert "_validation_details" not in result


def test_uncited_clarification_is_accepted(minimum):
    minimum.models.chat.next_tools = [("Answer", {
        "answer": "Is this for a new or existing business?",
        "source_ids": [], "status": "clarification",
    })]
    result = ask(minimum, minimum.store.conversation("alice")["id"])
    assert result["status"] == "clarification" and result["sources"] == []


@pytest.mark.parametrize("question", ["i need a loan", "I need a loan!", "I want funding", "I need a business loan"])
def test_bare_financing_request_needs_no_model_or_retrieval(minimum, monkeypatch, question):
    before = minimum.models.provider.calls
    def unexpected_search(query):
        raise AssertionError("A bare request needs clarification before factual retrieval")
    monkeypatch.setattr(minimum.r, "search", unexpected_search)
    result = ask(minimum, minimum.store.conversation("alice")["id"], question)
    assert result["status"] == "clarification" and result["sources"] == []
    assert "?" in result["answer"]
    assert minimum.models.chat.calls == 0 and minimum.models.provider.calls == before


def test_missing_citation_metadata_is_reconciled_without_another_model_call(minimum):
    minimum.models.chat.next_tools = [("Answer", {
        "answer": "The guide lists a registration certificate. [S1]",
        "source_ids": [], "status": "answered",
    })]
    with request_timings("metadata-test") as trace:
        result = ask(minimum, minimum.store.conversation("alice")["id"])
    assert result["status"] == "answered"
    assert [source["id"] for source in result["sources"]] == ["S1"]
    assert trace["answer_diagnostic"]["citation_metadata_reconciled"] is True
    assert minimum.models.chat.calls == 1


@pytest.mark.parametrize("question", ["Which loan schemes are available?", "What is the loan interest rate?", "I need a loan of 10 lakh"])
def test_specific_loan_questions_still_require_retrieval(minimum, monkeypatch, question):
    searches = []
    monkeypatch.setattr(minimum.r, "search", lambda query: searches.append(query) or [])
    result = ask(minimum, minimum.store.conversation("alice")["id"], question)
    assert searches == [question]
    assert result["status"] == "insufficient_evidence"


@pytest.mark.parametrize("kind,reason", [
    ("truncated", "generation_truncated"),
    ("citation", "invalid_citations_or_format"),
    ("schema", "invalid_answer_schema"),
    ("unsupported", "model_insufficient_evidence"),
    ("empty", "no_retrieved_chunks"),
    ("budget", "context_budget_exhausted"),
])
def test_failure_diagnostics_survive_persistence_without_leaking_text(
    minimum, monkeypatch, caplog, kind, reason
):
    recorded = []
    monkeypatch.setattr("rag.agent.record_unanswered", lambda *args: recorded.append(args[-1]))
    if kind == "truncated":
        monkeypatch.setattr(minimum.models.chat, "invoke", lambda _: AIMessage(
            content="", response_metadata={"finish_reason": "length"}))
    elif kind == "empty":
        monkeypatch.setattr(minimum.r, "search", lambda _: [])
    elif kind == "budget":
        minimum.s.context_tokens = 1
    else:
        args = {
            "answer": "Private generated text [S99]", "source_ids": ["S99"], "status": "answered"
        }
        if kind == "schema":
            args = {"answer": "Private generated text"}
        elif kind == "unsupported":
            args = {"answer": "Private generated text", "source_ids": [], "status": "insufficient_evidence"}
        minimum.models.chat.next_tools = [("Answer", args)]
    cid = minimum.store.conversation("alice")["id"]
    with caplog.at_level("INFO"), request_timings("diagnostic-request") as trace:
        result = ask(minimum, cid, "Private question")
    assert result["status"] == "insufficient_evidence"
    assert "_diagnostic" not in result
    assert recorded[0]["failure_reason"] == reason
    assert recorded[0]["search_query"] == "Private question"
    assert trace["answer_diagnostic"]["reason"] == reason
    assert "diagnostic-request" in caplog.text
    assert "Private question" not in caplog.text
    assert "Private generated text" not in caplog.text
    assert minimum.models.chat.calls <= 1 and minimum.models.utilities == []


def test_long_history_stays_stored_without_summary(minimum):
    cid = minimum.store.conversation("alice")["id"]
    for _ in range(8):
        ask(minimum, cid)
    rows = minimum.store.recent(cid, 100)
    assert len(rows) == 16
    conversation = minimum.store.conversation("alice", cid)
    summary, history = minimum.agent.memory(conversation, rows)
    assert summary == "" and len(history) <= 6 and len(history) % 2 == 0
    assert sum(token_count(m.content) + 8 for m in history) <= 1000
    assert minimum.models.utilities == []
    minimum.s.history_tokens = 500
    assert fit_messages([{"role": "user", "content": "partial"}], 500) == []


@pytest.mark.parametrize(
    "response",
    [
        AIMessage(content="", response_metadata={"finish_reason": "length"}),
        AIMessage(content="bad JSON"),
        AIMessage(
            content="",
            tool_calls=[{"name": "Answer", "args": {"answer": "missing fields"}, "id": "x"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_knowledge", "args": {"query": "more"}, "id": "x"}],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "Answer",
                    "args": {"answer": "Fake [S99]", "source_ids": ["S99"], "status": "answered"},
                    "id": "x",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "Answer",
                    "args": {
                        "answer": "See https://evil.test [S1]",
                        "source_ids": ["S1"],
                        "status": "answered",
                    },
                    "id": "x",
                }
            ],
        ),
    ],
)
def test_invalid_or_truncated_output_never_triggers_repair(minimum, monkeypatch, response):
    invocations = []
    monkeypatch.setattr(
        minimum.models.chat, "invoke", lambda messages: invocations.append(messages) or response
    )
    result = ask(minimum, minimum.store.conversation("alice")["id"])
    assert result["status"] == "insufficient_evidence" and result["sources"] == []
    assert len(invocations) == 1 and minimum.models.utilities == []
    assert "_failure_reason" not in result


def test_empty_retrieval_skips_generation(minimum, monkeypatch):
    monkeypatch.setattr(minimum.r, "search", lambda query: [])
    assert (
        ask(minimum, minimum.store.conversation("alice")["id"])["status"] == "insufficient_evidence"
    )
    assert minimum.models.chat.calls == 0 and minimum.models.utilities == []


def test_ownership_and_concurrency_still_apply(minimum):
    cid = minimum.store.conversation("alice")["id"]
    with pytest.raises(LookupError):
        ask(minimum, cid, user="bob")
    with minimum.store.lease(cid), pytest.raises(BusyConversation):
        ask(minimum, cid)
    assert minimum.models.chat.calls == 0


def test_rate_limit_still_applies(minimum):
    minimum.s.rpm = 1
    cid = minimum.store.conversation("alice")["id"]
    minimum.agent.respond("alice", cid, "Hello", str(uuid.uuid4()), enforce_rate_limit=True)
    with pytest.raises(RateLimited):
        minimum.agent.respond("alice", cid, "DPIIT?", str(uuid.uuid4()), enforce_rate_limit=True)
    assert minimum.models.chat.calls == 0


def test_duplicate_evidence_is_not_sent_twice(minimum, monkeypatch):
    chunks = minimum.r.search("DPIIT documents?")
    monkeypatch.setattr(
        minimum.r, "search", lambda query: [chunks[0], dict(chunks[0], id="duplicate")]
    )
    ask(minimum, minimum.store.conversation("alice")["id"])
    evidence = json.loads(minimum.models.chat.last_messages[-1].content.split("\n", 1)[1])
    assert len(evidence) == 1


def test_whole_serialized_input_includes_schema_and_question(minimum):
    ask(minimum, minimum.store.conversation("alice")["id"])
    messages = minimum.models.chat.last_messages
    estimate = Models.guard(minimum.models, messages, [Answer])
    assert estimate <= minimum.s.model_input_tokens
    assert estimate > sum(token_count(m.content) for m in messages)
    # A huge essential question must fail before any generation invocation.
    minimum.s.model_input_tokens = 8000
    with pytest.raises(ModelFailure, match="Essential"):
        minimum.agent.generate("question " * 9000, "", [], "DPIIT documents?")
    assert minimum.models.chat.calls == 1


def test_guard_trims_whole_history_pairs_and_evidence(minimum, monkeypatch):
    from rag.minimum import generate_minimum

    chunks = minimum.r.search("DPIIT documents?")
    chunks = [dict(chunks[0], id=str(i), text=chunks[0]["text"] + str(i)) for i in range(3)]
    monkeypatch.setattr(minimum.r, "search", lambda query: chunks)
    checked = []
    original = minimum.models.guard

    def tight_guard(messages, tools):
        checked.append(messages)
        if any(m.type == "ai" for m in messages):
            raise ModelFailure("history too large")
        evidence = json.loads(messages[-1].content.split("\n", 1)[1])
        if len(evidence) > 1:
            raise ModelFailure("evidence too large")
        return original(messages, tools)

    monkeypatch.setattr(minimum.models, "guard", tight_guard)
    result = generate_minimum(
        minimum.agent,
        "DPIIT documents?",
        [HumanMessage(content="old"), AIMessage(content="old")],
        "DPIIT",
    )
    assert result["status"] == "answered"
    assert minimum.models.chat.calls == 1
    assert len(checked) >= 3
    assert len(json.loads(checked[-1][-1].content.split("\n", 1)[1])) == 1


def test_essential_input_cannot_silently_expand_budget(minimum):
    minimum.s.model_input_tokens = 1
    with pytest.raises(ModelFailure, match="Essential"):
        ask(minimum, minimum.store.conversation("alice")["id"])
    assert minimum.models.chat.calls == 0


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_real_adapter_output_limit_and_zero_http_retries(provider, tmp_path):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(503, json={"error": {"message": "offline failure"}})

    settings = Settings(
        openai_key="fake",
        gemini_key="fake",
        chat_provider=provider,
        minimum_calls=True,
        db_path=str(tmp_path / "db"),
    )
    models = Models(settings, None)
    cls = GeminiChat if provider == "gemini" else type(models.chat)
    chat = cls(
        api_key="fake",
        model=settings.effective_chat_model,
        base_url=GEMINI_BASE_URL if provider == "gemini" else "https://api.openai.com/v1",
        use_responses_api=False,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        assert models.chat.root_client.max_retries == 0
        assert models.embeddings.provider.client._client.max_retries == 0
        assert models.chat.max_tokens == 400
        with pytest.raises(Exception):
            chat.bind_tools(
                [Answer],
                tool_choice="Answer",
                strict=True,
                parallel_tool_calls=False,
                max_tokens=400,
            ).invoke([HumanMessage(content="Question")])
        assert len(requests) == 1
        key = "max_tokens" if provider == "gemini" else "max_completion_tokens"
        assert requests[0][key] == 400
    finally:
        chat.root_client.close()
        models.close()


def test_model_purpose_is_logged_without_content():
    with request_timings("request") as trace:
        with stage("generation"):
            record_token_usage("gemini", "fake", "chat", {"input_tokens": 10, "output_tokens": 5})
    assert trace["token_usage"][0]["purpose"] == "generation"
    assert trace["token_usage"][0]["total_tokens"] == 15
