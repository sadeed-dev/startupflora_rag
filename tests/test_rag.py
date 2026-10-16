"""Offline integration tests: real LangChain tools, Qdrant and SQLite; no paid provider calls."""

import gzip
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from rag.agent import Agent
from rag.config import Settings, load_settings
from rag.crawler import Website
from rag.ingest import Ingestor
from rag.llm import CachedEmbeddings, ModelFailure, Models
from rag.retrieval import Evidence, Retriever
from rag.schemas import Answer, Grounding, Memory, Query, Ranking
from rag.store import BusyConversation, RateLimited, RequestConflict, Store
from rag.text import canonical_url, fit_messages, rrf, split_text
from rag.tools import build_tools


class ToyEmbeddings(Embeddings):
    """Return deterministic three-dimensional vectors for semantic retrieval tests."""

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [[1.0, 0.1, 0.0] if "dpiit" in t.lower() else [0.0, 0.1, 1.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class FakeChat:
    """Emulate LangChain tool messages while keeping the real tool execution path."""

    def __init__(self):
        self.calls = 0
        self.next_tools = []
        self.fail = False
        self.last_messages = []
        self.seen_tools = []

    def bind_tools(self, tools, **kwargs):
        self.seen_tools = [getattr(t, "name", getattr(t, "__name__", "")) for t in tools]
        return self

    def invoke(self, messages):
        self.calls += 1
        self.last_messages = list(messages)
        if self.fail:
            raise RuntimeError("synthetic model outage")
        if self.next_tools:
            name, args = self.next_tools.pop(0)
        else:
            name, args = (
                "Answer",
                {
                    "answer": "The indexed guide lists a registration certificate. [S1]",
                    "source_ids": ["S1"],
                    "status": "answered",
                },
            )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"call-{self.calls}",
                    "type": "tool_call",
                }
            ],
        )


class FakeModels:
    """Supply typed utility results and cached embeddings without network access."""

    signature = "test-model:3"

    def __init__(self, settings, store):
        self.s = settings
        self.provider = ToyEmbeddings()
        self.embeddings = CachedEmbeddings(store, self.provider, self.signature)
        self.chat = FakeChat()
        self.utilities = []
        self.grounding = True
        self.empty_rank = False
        self.fail_schema = None

    def structured(self, schema, instructions, payload, max_tokens=1000):
        self.utilities.append(schema.__name__)
        if schema is self.fail_schema:
            raise RuntimeError("synthetic utility outage")
        if schema is Ranking:
            return Ranking(
                chunk_ids=[]
                if self.empty_rank
                else [p["chunk_id"] for p in payload["passages"]][:5]
            )
        if schema is Query:
            return Query(query="DPIIT registration documents")
        if schema is Memory:
            return Memory(summary="User asked about DPIIT registration.")
        if schema is Grounding:
            return Grounding(supported=self.grounding)
        raise AssertionError(schema)

    def guard(self, messages, tools=None):
        Models.guard(self, messages, tools)


@pytest.fixture
def rig(tmp_path):
    s = Settings(
        openai_key="fake",
        backend_key="x" * 40,
        db_path=str(tmp_path / "store.db"),
        qdrant_path=str(tmp_path / "qdrant"),
        dimensions=3,
        collection="tests",
        recent_messages=2,
        crawl_hosts=("example.com",),
        crawl_delay=0,
    )
    store = Store(s.db_path)
    models = FakeModels(s, store)
    q = QdrantClient(":memory:")
    retriever = Retriever(s, store, models, q)
    ingestor = Ingestor(s, store, models, retriever)
    ingestor.document(
        "https://example.com/dpiit",
        "DPIIT guide",
        "DPIIT registration documents: the example guide lists a registration certificate. "
        + "Prepare your registration documents. " * 12,
    )
    agent = Agent(s, store, models, retriever)
    yield SimpleNamespace(s=s, store=store, models=models, r=retriever, ing=ingestor, agent=agent)
    q.close()


def test_settings_default_to_local_and_hide_secrets(tmp_path):
    s = Settings(openai_key="private-secret", db_path=str(tmp_path / "db"))
    assert s.qdrant_mode == "local" and "private-secret" not in repr(s)
    with pytest.raises(ValueError) as error:
        Settings(openai_key="private-secret", chunk_size=200, chunk_overlap=150)
    assert "private-secret" not in str(error.value)


def test_settings_read_current_environment(monkeypatch):
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_PATH", "testing-current-path")
    s = load_settings()
    assert Path(s.qdrant_path).name == "testing-current-path"


def test_recursive_split_preserves_content_and_overlap():
    text = "abcdefghij" * 400
    pieces = split_text(text, 400, 40)
    assert len(pieces) > 1 and max(map(len, pieces)) <= 400
    assert pieces[0][-40:] == pieces[1][:40]


def test_urls_are_allowlisted():
    assert canonical_url("https://example.com/a?x=1#f", ("example.com",)) == "https://example.com/a"
    for url in (
        "http://example.com",
        "https://evil.test",
        "https://a:b@example.com",
        "https://example.com:9999",
    ):
        with pytest.raises(ValueError):
            canonical_url(url, ("example.com",))


def test_fusion_rewards_agreement():
    assert rrf([["a", "b"], ["b", "c"]])[0] == "b"


def test_history_budget_keeps_complete_pairs():
    messages = [
        {"role": "user", "content": "a" * 100},
        {"role": "assistant", "content": "b" * 100},
    ]
    assert fit_messages(messages, 1) == []


def test_embedding_cache_deduplicates_and_survives_restart(tmp_path):
    store = Store(str(tmp_path / "db"))
    provider = ToyEmbeddings()
    embeddings = CachedEmbeddings(store, provider, "test:3")
    result = embeddings.embed_documents(["DPIIT", "DPIIT", "funding"])
    assert result[0] == result[1] and provider.calls == 2
    restarted = CachedEmbeddings(Store(store.path), provider, "test:3")
    assert restarted.embed_query("DPIIT") == result[0] and provider.calls == 2
    CachedEmbeddings(store, provider, "different:3").embed_query("DPIIT")
    assert provider.calls == 3


def test_unchanged_document_avoids_embedding(rig):
    count = rig.models.provider.calls
    assert (
        rig.ing.document(
            "https://example.com/dpiit",
            "DPIIT guide",
            "DPIIT registration documents: the example guide lists a registration certificate. "
            + "Prepare your registration documents. " * 12,
        )["status"]
        == "unchanged"
    )
    assert rig.models.provider.calls == count


def test_updates_replace_active_chunks_and_invalidate_search_cache(rig):
    first = rig.r.search("DPIIT documents")
    old_ids = [c["id"] for c in first]
    before = rig.models.utilities.count("Ranking")
    assert rig.r.search("DPIIT documents") == first
    assert rig.models.utilities.count("Ranking") == before
    rig.ing.document(
        "https://example.com/dpiit",
        "DPIIT guide",
        "DPIIT updated guide lists a company certificate. " * 12,
    )
    assert rig.store.get_chunks(old_ids) == []
    assert all("company certificate" in c["text"] for c in rig.r.search("DPIIT documents"))
    assert rig.models.utilities.count("Ranking") == before + 1


def test_empty_ranking_is_cached_as_empty(rig):
    rig.models.empty_rank = True
    assert rig.r.search("unrelated question") == []
    calls = rig.models.utilities.count("Ranking")
    assert rig.r.search("unrelated question") == []
    assert rig.models.utilities.count("Ranking") == calls


def test_reranking_failure_has_a_fused_fallback(rig):
    rig.models.fail_schema = Ranking
    assert rig.r.search("DPIIT")


def test_incompatible_namespace_and_embedding_rejected(rig):
    with pytest.raises(ValueError):
        rig.store.bind_namespace("another-target")
    rig.models.signature = "different:3"
    with pytest.raises(ValueError):
        rig.r.search("DPIIT")


def test_keyword_expression_cannot_inject_operators(rig):
    assert rig.store.keyword_search('DPIIT OR " NOT * -', 10)


def test_tool_arguments_are_scoped_to_current_evidence(rig):
    e = Evidence(6000)
    tools = build_tools(rig.r, rig.store, e)
    assert set(tools) == {"search_knowledge", "read_chunk_context"}
    assert "error" in tools["read_chunk_context"].invoke({"chunk_id": "unknown"})
    found = tools["search_knowledge"].invoke({"query": "DPIIT"})
    chunk = found["evidence"][0]["chunk_id"]
    context = tools["read_chunk_context"].invoke({"chunk_id": chunk})
    assert context["already_supplied_source_ids"]
    assert tools["search_knowledge"].invoke({"query": "DPIIT"})["evidence"] == []


def test_evidence_budget_is_enforced(rig):
    assert Evidence(1).add(rig.r.search("DPIIT")) == []


def test_chat_is_user_isolated_and_durable(rig):
    cid = rig.store.conversation("alice")["id"]
    with pytest.raises(LookupError):
        rig.agent.respond("bob", cid, "DPIIT?", str(uuid.uuid4()))
    result = rig.agent.respond("alice", cid, "DPIIT?", str(uuid.uuid4()))
    assert result["sources"][0]["id"] == "S1"
    history = Store(rig.store.path).history_page(cid)
    assert len(history) == 2 and history[1]["sources"] == result["sources"]


def test_retry_is_idempotent_and_rejects_changed_message(rig):
    cid = rig.store.conversation("alice")["id"]
    rid = str(uuid.uuid4())
    first = rig.agent.respond("alice", cid, "DPIIT?", rid)
    calls = rig.models.chat.calls
    assert rig.agent.respond("alice", cid, "DPIIT?", rid) == first
    assert rig.models.chat.calls == calls and len(rig.store.history_page(cid)) == 2
    with pytest.raises(RequestConflict):
        rig.agent.respond("alice", cid, "Different?", rid)


def test_followups_use_history_and_rolling_summary(rig):
    cid = rig.store.conversation("alice")["id"]
    for message in ("DPIIT registration?", "Which documents?", "And next steps?"):
        rig.agent.respond("alice", cid, message, str(uuid.uuid4()))
    assert len(rig.store.history_page(cid)) == 6
    assert rig.store.conversation("alice", cid)["summary"]
    assert "Query" in rig.models.utilities and "Memory" in rig.models.utilities
    assert any(isinstance(m, AIMessage) for m in rig.models.chat.last_messages)


def test_custom_tool_roundtrip_uses_langchain_messages(rig):
    rig.models.chat.next_tools = [("search_knowledge", {"query": "DPIIT registration"})]
    cid = rig.store.conversation("alice")["id"]
    result = rig.agent.respond("alice", cid, "DPIIT?", str(uuid.uuid4()))
    assert result["status"] == "answered" and rig.models.chat.calls == 2
    assert any(m.type == "tool" for m in rig.models.chat.last_messages)


def test_tool_budget_forces_final_answer(rig):
    s = rig.s.model_copy(update={"max_tools": 0})
    agent = Agent(s, rig.store, rig.models, rig.r)
    cid = rig.store.conversation("alice")["id"]
    agent.respond("alice", cid, "DPIIT?", str(uuid.uuid4()))
    assert rig.models.chat.seen_tools == ["Answer"]


def test_unknown_tools_are_never_executed(rig):
    rig.models.chat.next_tools = [("execute_shell", {"command": "unsafe"})]
    cid = rig.store.conversation("alice")["id"]
    with pytest.raises(ModelFailure):
        rig.agent.respond("alice", cid, "DPIIT?", str(uuid.uuid4()))
    assert rig.store.history_page(cid) == []


def test_invalid_citations_and_links_fail_closed(rig):
    e = Evidence(6000)
    e.add(rig.r.search("DPIIT"))
    for answer in (
        Answer(answer="Unknown [S999]", source_ids=["S999"], status="answered"),
        Answer(answer="Made up fact", source_ids=[], status="answered"),
        Answer(
            answer="Visit https://invented.test [S1]",
            source_ids=["S1"],
            status="answered",
        ),
    ):
        assert rig.agent.validate_answer(answer, e)["status"] == "insufficient_evidence"


def test_failed_grounding_and_verifier_outage_fail_closed(rig):
    e = Evidence(6000)
    e.add(rig.r.search("DPIIT"))
    answer = Answer(answer="Certificate required. [S1]", source_ids=["S1"], status="answered")
    rig.models.grounding = False
    assert rig.agent.validate_answer(answer, e)["status"] == "insufficient_evidence"
    rig.models.fail_schema = Grounding
    assert rig.agent.validate_answer(answer, e)["status"] == "insufficient_evidence"


def test_generation_failure_has_no_partial_turn_or_stuck_lease(rig):
    rig.models.chat.fail = True
    cid = rig.store.conversation("alice")["id"]
    with pytest.raises(RuntimeError):
        rig.agent.respond("alice", cid, "DPIIT?", str(uuid.uuid4()))
    assert rig.store.history_page(cid) == []
    with rig.store.lease(cid):
        pass


def test_simple_greeting_avoids_models_and_embeddings(rig):
    cid = rig.store.conversation("alice")["id"]
    count = rig.models.provider.calls
    assert rig.agent.respond("alice", cid, "hello", str(uuid.uuid4()))["status"] == "conversational"
    assert rig.models.chat.calls == 0 and rig.models.provider.calls == count


def test_leases_rate_limits_and_deletion(rig):
    cid = rig.store.conversation("alice")["id"]
    with rig.store.lease(cid), pytest.raises(BusyConversation):
        with rig.store.lease(cid):
            pass
    rig.store.check_rate("alice", 1)
    with pytest.raises(RateLimited):
        rig.store.check_rate("alice", 1)
    rig.store.delete_conversation(cid)
    with pytest.raises(LookupError):
        rig.store.conversation("alice", cid)


def test_history_migration_keeps_ownership_but_not_corpus_or_retry_cache(rig, tmp_path):
    cid = rig.store.conversation("alice")["id"]
    rig.agent.respond("alice", cid, "DPIIT?", str(uuid.uuid4()))
    target = Store(str(tmp_path / "new-history.sqlite3"))
    assert target.import_history(rig.store.path) == 1
    assert target.import_history(rig.store.path) == 0
    assert len(target.history_page(cid)) == 2 and target.document_count() == 0
    with pytest.raises(LookupError):
        target.conversation("bob", cid)


def test_remove_source_invalidates_active_results(rig):
    rig.r.search("DPIIT")
    assert rig.ing.remove_source("https://example.com/dpiit")["chunks"] > 0
    assert rig.r.search("DPIIT") == []


def website_for(rig, monkeypatch, routes):
    """Use a real HTTP parser/robots implementation over an in-process transport."""
    calls = []

    def handler(request):
        calls.append(request)
        result = routes.get(request.url.path, (404, "", {}))
        if callable(result):
            result = result(request)
        status, body, headers = result
        if isinstance(body, str):
            body = body.encode()
        return httpx.Response(status, content=body, headers=headers, request=request)

    site = Website(rig.s, httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(site, "checked_url", lambda u: canonical_url(u, rig.s.crawl_hosts))
    return site, calls


def test_nested_gzip_sitemaps_discover_unlinked_pages(rig, monkeypatch):
    xml = '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://example.com/page6</loc></url><url><loc>https://evil.test/private</loc></url></urlset>'
    routes = {
        "/robots.txt": (
            200,
            "User-agent: *\nAllow: /\nSitemap: https://example.com/maps.xml",
            {},
        ),
        "/maps.xml": (
            200,
            "<sitemapindex><sitemap><loc>https://example.com/pages.xml.gz</loc></sitemap></sitemapindex>",
            {},
        ),
        "/pages.xml.gz": (200, gzip.compress(xml.encode()), {}),
    }
    site, _ = website_for(rig, monkeypatch, routes)
    assert site.sitemap_urls("https://example.com/") == ["https://example.com/page6"]
    site.close()


def test_malicious_xml_entities_are_rejected(rig, monkeypatch):
    xml = '<!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><urlset><url><loc>&xxe;</loc></url></urlset>'
    site, _ = website_for(
        rig, monkeypatch, {"/robots.txt": (404, "", {}), "/bad.xml": (200, xml, {})}
    )
    assert site.sitemap_urls("https://example.com/", "https://example.com/bad.xml") == []
    assert site.discovery_warnings
    site.close()


def test_robots_and_redirects_are_enforced(rig, monkeypatch):
    routes = {
        "/robots.txt": (200, "User-agent: *\nDisallow: /blocked", {}),
        "/go": (302, "", {"location": "/blocked"}),
    }
    site, calls = website_for(rig, monkeypatch, routes)
    with pytest.raises(PermissionError):
        site.page("https://example.com/go")
    assert all(r.url.path != "/blocked" for r in calls)
    site.close()


def test_private_dns_is_rejected(rig, monkeypatch):
    site = Website(rig.s)
    monkeypatch.setattr(
        "rag.crawler.socket.getaddrinfo",
        lambda *a, **k: [(None, None, None, None, ("127.0.0.1", 443))],
    )
    with pytest.raises(ValueError):
        site.checked_url("https://example.com/")
    site.close()


def test_conditional_refresh_reuses_links_and_removes_confirmed_404(rig, monkeypatch):
    html = (
        "<html><title>Funding</title><main>"
        + "Funding information for startups. " * 20
        + '<a href="/page6">More</a></main></html>'
    )
    state = {"deleted": False}

    def conditional(request):
        if state["deleted"]:
            return 404, "", {}
        if request.headers.get("if-none-match") == "v1":
            return 304, "", {"etag": "v1"}
        return 200, html, {"content-type": "text/html", "etag": "v1"}

    routes = {"/robots.txt": (404, "", {}), "/": conditional, "/page6": conditional}
    site, _ = website_for(rig, monkeypatch, routes)
    report = rig.ing.crawl("https://example.com/", 0, website=site)
    assert report["indexed"] == 2 and report["attempted"] == 2
    count = rig.models.provider.calls
    site, _ = website_for(rig, monkeypatch, routes)
    report = rig.ing.crawl("https://example.com/", 0, website=site)
    assert report["unchanged"] == 2 and rig.models.provider.calls == count
    state["deleted"] = True
    site, _ = website_for(rig, monkeypatch, routes)
    report = rig.ing.crawl("https://example.com/", 0, website=site)
    assert report["removed"] == 1 and rig.store.document("https://example.com/") is None


def test_crawl_cap_is_reported(rig, monkeypatch):
    html = (
        "<html><main>"
        + "Public business information. " * 20
        + '<a href="/second">Second</a></main></html>'
    )
    site, _ = website_for(
        rig,
        monkeypatch,
        {"/robots.txt": (404, "", {}), "/": (200, html, {"content-type": "text/html"})},
    )
    report = rig.ing.crawl("https://example.com/", 1, website=site)
    assert report["limited"] and report["remaining"] == 1


def test_api_auth_isolation_validation_and_history(rig, monkeypatch):
    from rag import api

    monkeypatch.setattr(api, "runtime", lambda: (rig.s, rig.store, rig.models, rig.r, rig.agent))
    monkeypatch.setattr(api, "close_runtime", lambda: None)
    with TestClient(api.app) as client:
        assert client.post("/conversations").status_code == 401
        headers = {"Authorization": "Bearer " + "x" * 40, "X-User-ID": "alice"}
        cid = client.post("/conversations", headers=headers).json()["conversation_id"]
        assert (
            client.get(
                f"/conversations/{cid}/messages",
                headers={**headers, "X-User-ID": "bob"},
            ).status_code
            == 404
        )
        body = {
            "conversation_id": cid,
            "request_id": str(uuid.uuid4()),
            "message": "DPIIT?",
        }
        assert (
            client.post("/chat", headers=headers, json={**body, "message": "  "}).status_code == 422
        )
        assert client.post("/chat", headers=headers, json=body).status_code == 200
        assert (
            client.post("/chat", headers=headers, json={**body, "message": "Another?"}).status_code
            == 409
        )
        assert client.get(f"/conversations/{cid}/messages", headers=headers).json()["messages"][1][
            "sources"
        ]
        assert client.delete(f"/conversations/{cid}", headers=headers).status_code == 200


def test_real_langchain_openai_structured_wire_format(tmp_path):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4.1-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"query":"DPIIT registration"}',
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )

    s = Settings(openai_key="fake", dimensions=3, db_path=str(tmp_path / "sdk.db"))
    chat = ChatOpenAI(
        api_key="fake",
        model="gpt-4.1-mini",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        use_responses_api=False,
        store=False,
    )
    models = Models(s, Store(s.db_path), chat=chat, utility=chat, embeddings=ToyEmbeddings())
    assert (
        models.structured(Query, "Rewrite", {"question": "registration?"}, max_tokens=123).query
        == "DPIIT registration"
    )
    assert seen[0]["store"] is False
    assert seen[0].get("max_tokens", seen[0].get("max_completion_tokens")) == 123
    assert seen[0]["response_format"]["type"] == "json_schema"
    models.close()


def test_real_langchain_function_call_roundtrip(rig):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            function = {
                "name": "search_knowledge",
                "arguments": '{"query":"DPIIT registration"}',
            }
        else:
            function = {
                "name": "Answer",
                "arguments": json.dumps(
                    {
                        "answer": "The guide lists a registration certificate. [S1]",
                        "source_ids": ["S1"],
                        "status": "answered",
                    }
                ),
            }
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{len(bodies)}",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4.1-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{len(bodies)}",
                                    "type": "function",
                                    "function": function,
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )

    chat = ChatOpenAI(
        api_key="fake",
        model="gpt-4.1-mini",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        use_responses_api=False,
        store=False,
    )
    rig.models.chat = chat
    cid = rig.store.conversation("alice")["id"]
    result = rig.agent.respond("alice", cid, "DPIIT?", str(uuid.uuid4()))
    assert result["status"] == "answered" and len(bodies) == 2
    assert bodies[0]["parallel_tool_calls"] is False
    assert all(t["function"]["strict"] is True for t in bodies[0]["tools"])
    assert any(m["role"] == "tool" for m in bodies[1]["messages"])
    chat.root_client.close()


def test_real_langchain_embedding_transport_and_cache(tmp_path):
    from langchain_openai import OpenAIEmbeddings

    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": [1.0, 0.0, 0.0]}
                    for i, _ in enumerate(body["input"])
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
        )

    provider = OpenAIEmbeddings(
        api_key="fake",
        model="text-embedding-3-small",
        dimensions=3,
        check_embedding_ctx_length=False,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    cached = CachedEmbeddings(
        Store(str(tmp_path / "embeddings.db")), provider, "text-embedding-3-small:3"
    )
    assert cached.embed_documents(["DPIIT", "DPIIT"]) == [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert cached.embed_query("DPIIT") == [1.0, 0.0, 0.0]
    assert len(requests) == 1 and requests[0]["input"] == ["DPIIT"]
    provider.client._client.close()


def test_orphan_vector_cleanup_keeps_active_chunks(rig):
    from qdrant_client.models import PointStruct

    orphan = str(uuid.uuid4())
    rig.r.q.upsert(rig.s.collection, points=[PointStruct(id=orphan, vector=[1.0, 0.0, 0.0])])
    count = rig.r.q.count(rig.s.collection).count
    assert rig.ing.cleanup_vectors()["orphan_vectors_removed"] == 1
    assert rig.r.q.count(rig.s.collection).count == count - 1
    assert rig.r.search("DPIIT")


def test_conversation_listing_is_owned_and_cursor_paginated(rig):
    ids = {rig.store.conversation("alice")["id"] for _ in range(3)}
    rig.store.conversation("bob")
    first = rig.store.list_conversations("alice", 2)
    second = rig.store.list_conversations("alice", 2, first["next_cursor"])
    assert {r["id"] for r in first["conversations"] + second["conversations"]} == ids
    assert not second["next_cursor"]
    assert len(rig.store.list_conversations("bob")["conversations"]) == 1
    with pytest.raises(ValueError):
        rig.store.list_conversations("alice", cursor="bad")


def test_doctor_hides_keys_and_does_not_create_storage(monkeypatch, tmp_path, capsys):
    import main

    target = tmp_path / "not-created"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-sensitive-test-key")
    monkeypatch.setenv("BACKEND_API_KEY", "private-backend-test-key-" * 3)
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_PATH", str(target))
    main.doctor()
    captured = capsys.readouterr().out
    assert "sk-sensitive-test-key" not in captured
    assert "private-backend-test-key" not in captured
    assert not target.exists()


def test_retrieval_evaluator_uses_reviewed_sources(rig, tmp_path, monkeypatch):
    import evaluate

    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "query": "DPIIT registration",
                "expected_sources": ["https://example.com/dpiit"],
                "should_abstain": False,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        evaluate, "runtime", lambda: (rig.s, rig.store, rig.models, rig.r, rig.agent)
    )
    result = evaluate.evaluate(cases)
    assert result["mean_source_recall"] == 1.0
    cases.write_text(json.dumps({"query": "Example", "expected_sources": ["REPLACE_ME"]}) + "\n")
    with pytest.raises(ValueError):
        evaluate.evaluate(cases)
