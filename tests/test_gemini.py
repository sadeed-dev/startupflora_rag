"""Provider routing and Gemini tool signatures using real clients and fake HTTP."""

import json

import httpx
import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from rag.config import Settings
from rag.gemini import GEMINI_BASE_URL, GeminiChat
from rag.llm import Models
from rag.schemas import Grounding, Query


def test_hybrid_clients_keep_embedding_provider_and_namespace():
    settings = Settings(openai_key="openai-test", gemini_key="gemini-test", chat_provider="gemini",
                        model_timeout_seconds=75)
    models = Models(settings, None)
    try:
        assert isinstance(models.chat, GeminiChat)
        assert isinstance(models.utility, GeminiChat)
        assert models.chat.root_client.timeout == 75
        assert models.utility.root_client.timeout == 75
        assert str(models.chat.root_client.base_url) == GEMINI_BASE_URL
        assert models.chat.openai_api_key.get_secret_value() == "gemini-test"
        assert str(models.embeddings.provider.client._client.base_url) == "https://api.openai.com/v1/"
        assert models.embeddings.provider.openai_api_key.get_secret_value() == "openai-test"
        assert models.signature == "text-embedding-3-small:1536"
        assert models.embeddings.provider.model == settings.embedding_model
    finally:
        models.close()


def test_gemini_configuration_requires_both_keys_and_hides_secret_repr():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        Settings(openai_key="fake", chat_provider="gemini").validate()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings(gemini_key="fake", chat_provider="gemini").validate()
    settings = Settings(openai_key="openai-secret", gemini_key="gemini-secret", chat_provider="gemini")
    settings.validate()
    assert "gemini-secret" not in repr(settings)
    assert settings.effective_chat_model.startswith("gemini-")


def test_fast_clients_disable_retries_and_bound_embedding_wait():
    settings = Settings(openai_key="fake", gemini_key="fake", chat_provider="gemini",
                        fast_mode=True, model_timeout_seconds=10)
    models = Models(settings, None)
    try:
        assert models.chat.root_client.max_retries == 0
        assert models.utility.root_client.max_retries == 0
        assert models.chat.root_client.timeout == 10
        embedding_client = models.embeddings.provider.client._client
        assert embedding_client.max_retries == 0
        assert embedding_client.timeout == 5
    finally:
        models.close()


def test_gemini_preserves_signed_tool_roundtrip_and_wire_parameters():
    requests = []
    extension = {"google": {"thought_signature": "opaque-test-signature"}}

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.headers["authorization"] == "Bearer gemini-test"
        if len(requests) == 1:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-query", "type": "function", "extra_content": extension,
                "function": {"name": "Query", "arguments": '{"query":"DPIIT"}'},
            }]}
        else:
            assert body["messages"][1]["tool_calls"][0]["extra_content"] == extension
            message = {"role": "assistant", "content": "Complete", "tool_calls": None}
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": "gemini-3.8-flash", "choices": [{
                "index": 0, "finish_reason": "tool_calls" if len(requests) == 1 else "stop", "message": message,
            }],
        })

    client = GeminiChat(api_key="gemini-test", base_url=GEMINI_BASE_URL,
                        model="gemini-3.8-flash", max_tokens=150, reasoning_effort="low",
                        use_responses_api=False, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        question = HumanMessage(content="Search DPIIT")
        response = client.bind_tools([Query], tool_choice="Query", strict=True, parallel_tool_calls=False).invoke([question])
        assert response.additional_kwargs["gemini_tool_extensions"]["call-query"] == extension
        client.invoke([question, response, ToolMessage(content="Found", tool_call_id="call-query")])
        assert requests[0]["max_tokens"] == 150
        assert "max_completion_tokens" not in requests[0]
        assert "store" not in requests[0]
        assert requests[0]["reasoning_effort"] == "low"
    finally:
        client.root_client.close()


def test_gemini_structured_verification_with_null_tool_calls():
    def handler(request):
        payload = json.loads(request.content)
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["max_tokens"] == 150
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": "gemini-3.8-flash", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"supported":true}', "tool_calls": None}}],
        })

    client = GeminiChat(api_key="gemini-test", base_url=GEMINI_BASE_URL,
                        model="gemini-3.8-flash", use_responses_api=False,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    settings = Settings(openai_key="fake", gemini_key="gemini-test", chat_provider="gemini")
    models = Models(settings, None, chat=client, utility=client)
    try:
        assert models.structured(Grounding, "Verify", {"answer": "test"}, max_tokens=150).supported
    finally:
        models.close()
