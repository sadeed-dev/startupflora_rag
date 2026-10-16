"""Private service: call from an authenticated website backend, not browsers."""

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from .runtime import close_runtime, runtime
from .store import BusyConversation, RateLimited, RequestConflict

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    """Validate server authentication at startup and release storage locks on shutdown."""
    s, *_ = runtime()
    if len(s.backend_key) < 32 or s.backend_key.startswith("replace-"):
        raise RuntimeError("Set BACKEND_API_KEY to a random secret of at least 32 characters")
    try:
        yield
    finally:
        close_runtime()


app = FastAPI(
    title="StartupFlora RAG",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


class ChatRequest(BaseModel):
    conversation_id: UUID
    request_id: UUID  # Reuse this ID only when retrying the SAME message.
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def nonempty(cls, value):
        """Reject blank or whitespace-only messages before retrieval or model calls."""
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


def trusted_user(
    authorization: Annotated[str | None, Header()] = None,
    x_user_id: Annotated[str | None, Header()] = None,
):
    """Accept identity only from the trusted authenticated website backend."""
    s, *_ = runtime()
    token = (authorization or "").removeprefix("Bearer ")
    if (
        not authorization
        or not authorization.startswith("Bearer ")
        or not secrets.compare_digest(token.encode("utf-8"), s.backend_key.encode("utf-8"))
    ):
        raise HTTPException(401, "Unauthorized")
    if not x_user_id or not 1 <= len(x_user_id) <= 128:
        raise HTTPException(400, "Trusted backend user identity is required")
    return x_user_id


User = Annotated[str, Depends(trusted_user)]


@app.get("/health")
def health():
    """Provide liveness without exposing configuration or claiming upstream readiness."""
    return {"status": "ok"}  # Liveness only; does not claim upstream availability.


@app.post("/conversations")
def create_conversation(user: User):
    """Create a rate-limited user-owned conversation with a server-generated UUID."""
    s, store, *_ = runtime()
    try:
        store.check_rate(user, s.rpm)
    except RateLimited:
        raise HTTPException(429, "Too many requests", headers={"Retry-After": "60"})
    return {"conversation_id": store.conversation(user)["id"]}


@app.post("/chat")
def chat(body: ChatRequest, user: User):
    """Enforce ownership and rate limits, then map known failures to stable HTTP statuses."""
    s, store, _, _, agent = runtime()
    try:
        # Check ownership before spending rate allowance or calling AI.
        store.conversation(user, str(body.conversation_id))
        store.check_rate(user, s.rpm)
        return agent.respond(user, str(body.conversation_id), body.message, str(body.request_id))
    except LookupError:
        raise HTTPException(404, "Conversation not found")
    except BusyConversation:
        raise HTTPException(
            409,
            "Another message is still being processed",
            headers={"Retry-After": "3"},
        )
    except RateLimited:
        raise HTTPException(429, "Too many requests", headers={"Retry-After": "60"})
    except RequestConflict:
        raise HTTPException(409, "Request ID was reused for a different message")
    except Exception as exc:
        log.error(
            "chat_failed request_id=%s error_type=%s",
            body.request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            503,
            "Chat service is temporarily unavailable; retry with the same request_id",
        )


@app.get("/conversations/{cid}/messages")
def history(
    cid: UUID,
    user: User,
    after: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    """Return only this user's paginated persisted conversation messages."""
    _, store, *_ = runtime()
    try:
        store.conversation(user, str(cid))
    except LookupError:
        raise HTTPException(404, "Conversation not found")
    rows = store.history_page(str(cid), after, limit)
    return {"messages": rows, "next_after": rows[-1]["seq"] if rows else after}


@app.delete("/conversations/{cid}")
def delete(cid: UUID, user: User):
    """Delete owned conversation data only after its active turn has finished."""
    _, store, *_ = runtime()
    try:
        store.conversation(user, str(cid))
        with store.lease(str(cid)):
            store.delete_conversation(str(cid))
    except LookupError:
        raise HTTPException(404, "Conversation not found")
    except BusyConversation:
        raise HTTPException(409, "Wait for the current message to finish")
    return {"deleted": True}


@app.get("/conversations")
def conversations(
    user: User,
    limit: int = Query(20, ge=1, le=100),
    before: str | None = Query(None, max_length=256),
):
    """Let the website resume prior chats without trusting browser-supplied user IDs."""
    _, store, *_ = runtime()
    try:
        return store.list_conversations(user, limit, before)
    except ValueError:
        raise HTTPException(400, "Invalid conversation cursor")
