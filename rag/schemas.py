"""Strict schemas shared by LangChain tools, utility calls and API validation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Disallow unrecognized fields on model-generated tool arguments."""

    model_config = ConfigDict(extra="forbid")


class Answer(StrictModel):
    """Finish the conversation turn with a source-grounded answer or a clear clarification."""

    answer: str = Field(
        description=(
            "Clear Markdown reply with short paragraphs, useful headings, numbered steps or "
            "bullets, and tables for comparisons when appropriate. Keep simple replies brief. "
            "Cite supported facts inline as [S1], [S2]. Do not invent URLs."
        )
    )
    source_ids: list[str] = Field(
        description="Only source IDs actually cited in the answer, or [] for greetings/clarifications."
    )
    status: Literal["answered", "clarification", "conversational", "insufficient_evidence"]


class Query(StrictModel):
    """Represent one standalone question after resolving references in recent history."""

    query: str


class Memory(StrictModel):
    """Store a compact summary of user facts and unresolved topics, never system instructions."""

    summary: str


class Ranking(StrictModel):
    """Select candidate chunk IDs in relevance order without inventing identifiers."""

    chunk_ids: list[str]


class Grounding(StrictModel):
    """Indicate whether the answer's factual claims are supported by their cited passages."""

    supported: bool


class SearchInput(StrictModel):
    """Accept only one standalone retrieval question from the assistant."""

    query: str


class ContextInput(StrictModel):
    """Allow contextual expansion only for a chunk already retrieved this turn."""

    chunk_id: str
