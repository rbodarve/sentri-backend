"""Request/response schemas for the Sentri FastAPI server.

Kept separate from server.py so the Spring Boot client can generate matching
DTOs from this file alone without pulling in the FastAPI runtime.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One turn of conversational history sent by the upstream backend.

    Role mirrors the OpenAI / Spring Boot chat schema. Content is the raw
    text — the server wraps these into the
    "Turn N — Q/A" format the Director expects.
    """

    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    """Inbound query payload from the Spring Boot backend.

    The backend owns session storage; this server is stateless. It receives
    the current question plus whatever prior turns the backend has saved
    and wants the model to see.
    """

    query: str = Field(..., min_length=1, description="The user's current question.")
    session_id: str = Field(
        ...,
        description="Opaque session identifier — used only for log correlation.",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description=(
            "Prior turns in chronological order. Pairs of user→assistant "
            "messages become Turn N — Q/A blocks in the augmented prompt."
        ),
    )


# Outbound SSE events are not modeled as Pydantic objects because they go
# over the wire as `event: <name>\ndata: <json>\n\n`. The shapes are:
#
#   event: meta         {"session_id": str, "received_at": iso8601}
#   event: stage        {"stage": str, ...stage-specific summary}
#   event: thinking     {"text": str}
#   event: token        {"text": str}
#   event: contract_ids {"contract_ids": [str, ...]}
#   event: sources      {"sources": [{"name": str, "page": int, "score": float|null,
#                                     "chunk_id": str|null}, ...]}
#   event: graph        {"nodes": [{"id": str, "label": str, "pages": [int, ...]}, ...],
#                        "edges": [{"source": str, "target": str, "weight": float,
#                                   "shared": [str, ...]}, ...]}
#                        Document-relation graph over the answer's cited documents.
#                        An edge means two documents share named entities; `weight`
#                        is IDF-summed (rare shared entities weigh more) and `shared`
#                        lists the linking entities. Nodes with no edges = documents
#                        cited but not interconnected to others in this result.
#   event: done         {"confidence": float, "uncertain": bool,
#                        "tokens": int, "cost_usd": float, "files": [str],
#                        "compose_backend": str|null, "compose_model": str|null,
#                        "response_text": str}
#   event: error        {"message": str}
