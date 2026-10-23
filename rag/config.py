"""Typed, validated settings. Environment values override the project's .env."""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"


def project_path(value: str) -> str:
    """Resolve relative data paths against the project, not the shell directory."""
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else ROOT / path)


class Settings(BaseModel):
    """Define validated limits and service settings without requiring an API key to inspect them."""

    model_config = ConfigDict(hide_input_in_errors=True)
    openai_key: str = Field(default="", repr=False)
    gemini_key: str = Field(default="", repr=False)
    backend_key: str = Field(default="", repr=False)
    chat_provider: Literal["openai", "gemini"] = "openai"
    gemini_chat_model: str = "gemini-3.8-flash"
    gemini_utility_model: str = "gemini-3.8-flash"
    gemini_reasoning_effort: Literal["low", "medium", "high"] = "low"
    chat_model: str = "gpt-4.1-mini"
    utility_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    dimensions: int = Field(default=1536, ge=1)
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_key: str = Field(default="", repr=False)
    qdrant_path: str = "data/v2/qdrant"
    collection: str = "startupflora_lc_v2"
    db_path: str = "data/v2/rag.sqlite3"
    # DB_BACKEND=mysql: records/caches/history MySQL mein; vectors Qdrant mein.
    db_backend: Literal["sqlite", "mysql"] = "mysql"
    mysql_host: str = "127.0.0.1"
    mysql_port: int = Field(default=3306, ge=1, le=65535)
    mysql_database: str = "startupflora_rag"
    mysql_user: str = "rag_app"
    mysql_password: str = Field(default="", repr=False)
    mysql_ssl_ca: str = ""
    mysql_pool_size: int = Field(default=5, ge=1, le=32)
    mysql_pool_timeout: float = Field(default=10, gt=0, le=60)
    answer_max_tokens: int = Field(default=1000, ge=400, le=1800)
    max_output_tokens: int = Field(default=400, ge=100, le=1800)
    minimum_calls: bool = False
    model_timeout_seconds: float = Field(default=90, gt=0, le=300)
    fast_mode: bool = False
    response_timeout_seconds: float = Field(default=18, gt=0, le=300)
    chunk_size: int = Field(default=1600, ge=200, le=6000)
    chunk_overlap: int = Field(default=200, ge=0)
    verify_answers: bool = True
    rerank: bool = True
    candidates: int = Field(default=16, ge=5, le=40)
    final_chunks: int = Field(default=3, ge=1, le=10)
    context_tokens: int = Field(default=6000, ge=1000, le=10000)
    history_tokens: int = Field(default=2500, ge=500, le=6000)
    model_input_tokens: int = Field(default=18000, ge=2000, le=40000)
    recent_messages: int = Field(default=12, ge=2, le=40)
    max_tools: int = Field(default=2, ge=0, le=4)
    rpm: int = Field(default=15, ge=1, le=1000)
    search_cache_seconds: int = Field(default=900, ge=0, le=86400)
    # -1 persists until corpus/model/code changes; 0 disables; positive values expire.
    answer_cache_seconds: int = Field(default=-1, ge=-1, le=604800)
    max_crawl_urls: int = Field(default=10000, ge=1, le=100000)
    max_sitemaps: int = Field(default=100, ge=1, le=500)
    crawl_delay: float = Field(default=0.5, ge=0, le=30)
    crawl_hosts: tuple[str, ...] = ("startupflora.com", "www.startupflora.com")

    @model_validator(mode="after")
    def validate_limits(self):
        """Reject inconsistent budgets before any paid calls or data writes."""
        if self.chunk_overlap >= self.chunk_size // 2:
            raise ValueError("CHUNK_OVERLAP must be less than half CHUNK_SIZE")
        if self.recent_messages % 2:
            raise ValueError("RECENT_MESSAGES must be even to preserve complete turns")
        if self.final_chunks > self.candidates:
            raise ValueError("FINAL_CHUNKS must not exceed RETRIEVAL_CANDIDATES")
        if self.context_tokens + self.history_tokens + 4500 > self.model_input_tokens:
            raise ValueError(
                "MODEL_INPUT_TOKENS is too small for the context/history budgets"
            )
        self.db_path = project_path(self.db_path)
        self.qdrant_path = project_path(self.qdrant_path)
        if self.mysql_ssl_ca:
            self.mysql_ssl_ca = project_path(self.mysql_ssl_ca)
        return self

    def validate(self):
        """Require a real provider key only when indexing or chatting, not for doctor/status."""
        if not self.openai_key or self.openai_key == "replace-me":
            raise ValueError(
                f"Set OPENAI_API_KEY in {ENV_FILE}. Run: python main.py doctor"
            )
        if self.chat_provider == "gemini" and (not self.gemini_key or self.gemini_key == "replace-me"):
            raise ValueError(f"Set GEMINI_API_KEY in {ENV_FILE}. OPENAI_API_KEY is still required for embeddings.")

    @property
    def effective_chat_model(self):
        return self.gemini_chat_model if self.chat_provider == "gemini" else self.chat_model

    @property
    def effective_utility_model(self):
        return self.gemini_utility_model if self.chat_provider == "gemini" else self.utility_model


def load_settings() -> Settings:
    """Read current environment values at runtime so test overrides and .env edits are predictable."""
    load_dotenv(ENV_FILE, override=False)
    fields = {
        "openai_key": "OPENAI_API_KEY",
        "gemini_key": "GEMINI_API_KEY",
        "chat_provider": "CHAT_PROVIDER",
        "gemini_chat_model": "GEMINI_CHAT_MODEL",
        "gemini_utility_model": "GEMINI_UTILITY_MODEL",
        "gemini_reasoning_effort": "GEMINI_REASONING_EFFORT",
        "backend_key": "BACKEND_API_KEY",
        "chat_model": "CHAT_MODEL",
        "utility_model": "UTILITY_MODEL",
        "embedding_model": "EMBEDDING_MODEL",
        "dimensions": "EMBEDDING_DIMENSIONS",
        "qdrant_mode": "QDRANT_MODE",
        "qdrant_url": "QDRANT_URL",
        "qdrant_key": "QDRANT_API_KEY",
        "qdrant_path": "QDRANT_PATH",
        "collection": "QDRANT_COLLECTION",
        "db_path": "DB_PATH",
        "db_backend": "DB_BACKEND",
        "mysql_host": "MYSQL_HOST",
        "mysql_port": "MYSQL_PORT",
        "mysql_database": "MYSQL_DATABASE",
        "mysql_user": "MYSQL_USER",
        "mysql_password": "MYSQL_PASSWORD",
        "mysql_ssl_ca": "MYSQL_SSL_CA",
        "mysql_pool_size": "MYSQL_POOL_SIZE",
        "mysql_pool_timeout": "MYSQL_POOL_TIMEOUT",
        "answer_max_tokens": "ANSWER_MAX_TOKENS",
        "max_output_tokens": "MAX_OUTPUT_TOKENS",
        "minimum_calls": "MINIMUM_CALLS",
        "model_timeout_seconds": "MODEL_TIMEOUT_SECONDS",
        "fast_mode": "FAST_MODE",
        "response_timeout_seconds": "RESPONSE_TIMEOUT_SECONDS",
        "chunk_size": "CHUNK_SIZE",
        "chunk_overlap": "CHUNK_OVERLAP",
        "verify_answers": "VERIFY_ANSWERS",
        "rerank": "ENABLE_RERANK",
        "candidates": "RETRIEVAL_CANDIDATES",
        "final_chunks": "FINAL_CHUNKS",
        "context_tokens": "CONTEXT_TOKENS",
        "history_tokens": "HISTORY_TOKENS",
        "model_input_tokens": "MODEL_INPUT_TOKENS",
        "recent_messages": "RECENT_MESSAGES",
        "max_tools": "MAX_TOOL_CALLS",
        "rpm": "REQUESTS_PER_MINUTE",
        "search_cache_seconds": "SEARCH_CACHE_SECONDS",
        "answer_cache_seconds": "ANSWER_CACHE_SECONDS",
        "max_crawl_urls": "MAX_CRAWL_URLS",
        "max_sitemaps": "MAX_SITEMAPS",
        "crawl_delay": "CRAWL_DELAY",
    }
    values = {
        field: os.environ[key]
        for field, key in fields.items()
        if os.environ.get(key, "").strip()
    }
    if os.environ.get("ALLOWED_CRAWL_HOSTS"):
        values["crawl_hosts"] = tuple(
            h.strip().lower()
            for h in os.environ["ALLOWED_CRAWL_HOSTS"].split(",")
            if h.strip()
        )
    return Settings.model_validate(values)
