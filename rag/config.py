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
    backend_key: str = Field(default="", repr=False)
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
    db_backend: Literal["sqlite", "mysql"] = "sqlite"
    mysql_host: str = "127.0.0.1"
    mysql_port: int = Field(default=3306, ge=1, le=65535)
    mysql_database: str = "startupflora_rag"
    mysql_user: str = "rag_app"
    mysql_password: str = Field(default="", repr=False)
    mysql_ssl_ca: str = ""
    chunk_size: int = Field(default=1600, ge=200, le=6000)
    chunk_overlap: int = Field(default=200, ge=0)
    verify_answers: bool = True
    rerank: bool = True
    candidates: int = Field(default=16, ge=5, le=40)
    final_chunks: int = Field(default=5, ge=1, le=10)
    context_tokens: int = Field(default=6000, ge=1000, le=10000)
    history_tokens: int = Field(default=2500, ge=500, le=6000)
    model_input_tokens: int = Field(default=18000, ge=12000, le=40000)
    recent_messages: int = Field(default=12, ge=2, le=40)
    max_tools: int = Field(default=2, ge=0, le=4)
    rpm: int = Field(default=15, ge=1, le=1000)
    search_cache_seconds: int = Field(default=900, ge=0, le=86400)
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


def load_settings() -> Settings:
    """Read current environment values at runtime so test overrides and .env edits are predictable."""
    load_dotenv(ENV_FILE, override=False)
    fields = {
        "openai_key": "OPENAI_API_KEY",
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
