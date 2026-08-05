"""Request schemas for the generic streaming transport.

Domain-neutral: these describe the *inbound* HTTP contract only (a question,
a session id, and prior chat turns). They carry no domain specifics, so a
second service can reuse them unchanged.

Kept separate from transport.py so an upstream client (Spring Boot, a Next.js
frontend, etc.) can generate matching DTOs from this file alone without pulling
in the FastAPI runtime.

Outbound SSE events are NOT modeled here — they go over the wire as
`event: <name>\ndata: <json>\n\n`. The transport emits four generic events
(`meta`, `stage`, `thinking`, `token`, plus `error`); every *terminal* event
(the answer projection) is produced by your injected `finalize` seam, so their
shapes live with your pipeline, not here.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One turn of conversational history sent by the upstream backend.

    Role mirrors the OpenAI / Spring Boot chat schema. Content is the raw
    text — the transport's `augment_query` seam decides how to fold these
    into the prompt your pipeline sees.
    """

    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    """Inbound query payload. The transport is stateless — the caller owns
    session storage and sends whatever prior turns it wants the model to see.
    """

    query: str = Field(..., min_length=1, description="The user's current question.")
    session_id: str = Field(
        ...,
        description="Opaque session identifier — used only for log correlation.",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior turns in chronological order (user→assistant pairs).",
    )
