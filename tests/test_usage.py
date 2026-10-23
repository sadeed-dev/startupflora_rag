"""Exercise real SDK/LangChain usage handling over fake HTTP; no paid calls."""

import json
import logging

import httpx
from openai import OpenAI

from rag.config import Settings
from rag.llm import Models
from rag.schemas import Grounding
from rag.timing import record_token_usage, request_timings
from rag.usage import EmbeddingUsageClient


def test_chat_structured_and_embedding_usage_are_logged_once(caplog):
    def handler(request):
        body = json.loads(request.content)
        if request.url.path.endswith("embeddings"):
            return httpx.Response(200, json={
                "object": "list", "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1] * 1536}],
                "usage": {"prompt_tokens": 7, "total_tokens": 7},
            })
        content = '{"supported":true}' if "response_format" in body else "OK"
        return httpx.Response(200, json={
            "id": "fake", "object": "chat.completion", "created": 0,
            "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content, "tool_calls": None}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        })

    settings = Settings(openai_key="hidden-openai-key", gemini_key="hidden-gemini-key",
                        chat_provider="gemini")
    models = Models(settings, None)
    try:
        with OpenAI(api_key="hidden-sdk-key", http_client=httpx.Client(
            transport=httpx.MockTransport(handler),
        )) as sdk:
            models.chat.root_client.close()
            models.utility.root_client.close()
            models.chat.root_client = sdk
            models.utility.root_client = sdk
            models.chat.client = sdk.chat.completions
            models.utility.client = sdk.chat.completions
            models.embeddings.provider.client = EmbeddingUsageClient(sdk.embeddings, settings.embedding_model)
            with caplog.at_level(logging.INFO, logger="uvicorn.error"):
                with request_timings("usage-test") as trace:
                    models.chat.invoke("private prompt text")
                    assert models.structured(Grounding, "private instructions", {"answer": "private"}).supported
                    models.embeddings.provider.embed_documents(["private embedding text"])
        entries = trace["token_usage"]
        assert len(entries) == 3
        assert [(e["provider"], e["operation"], e["total_tokens"]) for e in entries] == [
            ("gemini", "chat", 15), ("gemini", "chat", 15), ("openai", "embedding", 7),
        ]
        assert all(e["request_id"] == "usage-test" and e["usage_reported"] for e in entries)
        assert entries[2]["input_tokens"] == 7 and entries[2]["output_tokens"] == 0
        assert len([r for r in caplog.records if r.message.startswith("rag_token_usage ")]) == 3
        assert "hidden-" not in caplog.text and "private" not in caplog.text
    finally:
        models.close()


def test_missing_usage_and_failed_calls_are_not_reported_as_zero():
    with request_timings("missing") as trace:
        record_token_usage("gemini", "test-model", "chat", None, status="error")
        with request_timings("nested") as nested:
            record_token_usage("openai", "test-model", "chat", {"input_tokens": 2, "output_tokens": 1})
    entry = trace["token_usage"][0]
    assert entry["usage_reported"] is False
    assert entry["total_tokens"] is None and entry["status"] == "error"
    assert nested["token_usage"][0]["total_tokens"] == 3
    assert len(trace["token_usage"]) == 1
