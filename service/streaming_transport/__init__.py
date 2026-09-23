"""Generic FastAPI + SSE streaming transport for a RAG/agent pipeline.

Reusable across services: implement the five seams for your pipeline (see
`create_app` and SEAMS.txt) and hand them in — you never edit this package.
"""
from .schemas import ChatMessage, QueryRequest
from .tracer import QueueTracer, StreamTracer
from .transport import (
    StreamingPipeline,
    create_app,
    default_augment_query,
)

__all__ = [
    "ChatMessage",
    "QueryRequest",
    "StreamTracer",
    "QueueTracer",
    "StreamingPipeline",
    "create_app",
    "default_augment_query",
]
