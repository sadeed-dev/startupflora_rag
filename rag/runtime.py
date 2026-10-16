"""Construct shared services lazily so doctor/status work without a provider key."""

from functools import lru_cache

from .agent import Agent
from .config import load_settings
from .llm import Models
from .retrieval import Retriever
from .store import create_store


@lru_cache(maxsize=1)
def runtime():
    """Create one settings/store/model/vector service set per API process."""
    settings = load_settings()
    settings.validate()
    store = create_store(settings)
    models = Models(settings, store)
    try:
        retriever = Retriever(settings, store, models)
    except BaseException:
        models.close()
        raise
    return settings, store, models, retriever, Agent(settings, store, models, retriever)


def close_runtime():
    """Release local Qdrant locks on CLI exit or API shutdown."""
    if runtime.cache_info().currsize:
        _, _, models, retriever, _ = runtime()
        retriever.close()
        models.close()
        runtime.cache_clear()
