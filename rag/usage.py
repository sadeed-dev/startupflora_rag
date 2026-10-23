"""Capture synchronous provider usage without logging message text or credentials."""

from langchain_core.callbacks import BaseCallbackHandler

from .timing import record_token_usage


class TokenUsageCallback(BaseCallbackHandler):
    """Capture all generation and structured-output calls using the shared clients."""

    def __init__(self, provider, model):
        self.provider, self.model = provider, model

    def on_llm_end(self, response, **kwargs):
        usage = (response.llm_output or {}).get("token_usage")
        if not usage and response.generations and response.generations[0]:
            message = getattr(response.generations[0][0], "message", None)
            usage = getattr(message, "usage_metadata", None)
        record_token_usage(self.provider, self.model, "chat", usage)

    def on_llm_error(self, error, **kwargs):
        record_token_usage(self.provider, self.model, "chat", None, status="error")


class EmbeddingUsageClient:
    """Intercept SDK responses before LangChain discards embedding token usage."""

    def __init__(self, client, model):
        self.client, self.model = client, model

    def create(self, *args, **kwargs):
        try:
            response = self.client.create(*args, **kwargs)
        except Exception:
            record_token_usage("openai", self.model, "embedding", None, status="error")
            raise
        payload = response if isinstance(response, dict) else response.model_dump()
        record_token_usage("openai", self.model, "embedding", payload.get("usage"))
        return response

    def __getattr__(self, name):
        return getattr(self.client, name)
