"""FastAPI wrapper around the Sentri agentic pipeline.

Mirrors the model/pipeline initialization done by `cmd_sentri_interactive`
in main.py, but strips the concerns the Spring Boot backend now owns:
session loading from investigations.json, temp_memory audit files, Earth
Engine bootstrap, map data fetching, and stdout marker emission.

Single endpoint: POST /query streams the answer back as Server-Sent
Events. The Spring Boot client decides what to persist; this server is
stateless.

Run:
    cd <repo root>
    python -m uvicorn sentri_api.server:app --host 0.0.0.0 --port 8000

The pipeline is initialized ONCE at startup (FastAPI lifespan) and shared
across all requests, so model loads happen exactly once.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from dotenv import load_dotenv
load_dotenv()

# Make the repo root importable when this server is launched from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from rag_module.agents import SentriPipeline
from rag_module.agents.trail import (
    STAGE_COMPOSE,
    STAGE_DIRECT_ANSWER,
    STAGE_DIRECTOR,
    STAGE_EXPERT_DISPATCH,
    STAGE_EXPERT_RESULT,
    STAGE_EXPERT_RETRY,
    STAGE_RERANK,
    STAGE_SEARCH,
    SentriTracer,
)
from rag_module.core.config import load_config
from rag_module.pipeline import build_pipeline

from .schemas import ChatMessage, QueryRequest

logger = logging.getLogger("sentri_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_CONTRACT_ID_RE = re.compile(r"\b\d{2}[A-Za-z]{1,3}\d{4,6}\b")

# ---------------------------------------------------------------------------
# Pipeline holder — initialized once at startup, shared across requests
# ---------------------------------------------------------------------------

class _PipelineHolder:
    """Lazily holds the singleton SentriPipeline after lifespan startup."""

    def __init__(self) -> None:
        self.sentri: SentriPipeline | None = None
        self.config: dict[str, Any] | None = None
        # GraphStore captured at startup so /query can project the answer's
        # evidence into a document-relation graph (the `graph` SSE event).
        self.graph_store: Any | None = None

    def require(self) -> SentriPipeline:
        if self.sentri is None:
            raise RuntimeError("Pipeline not initialized — lifespan startup did not run.")
        return self.sentri


_holder = _PipelineHolder()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build the RAG retriever + Sentri pipeline once at process startup.

    The model loads (BGE embedder, ms-marco reranker, ChromaDB clients,
    knowledge graph JSONs, LLM clients) all happen here so the first
    request doesn't pay a ~30 s cold-start penalty.
    """
    logger.info("Initializing Sentri pipeline (this takes ~30s on first boot)…")
    config = load_config()
    pipeline = await run_in_threadpool(build_pipeline, config)
    sentri = await run_in_threadpool(
        SentriPipeline.from_config,
        config,
        pipeline.retriever,
        pipeline.reranker,
    )
    _holder.sentri = sentri
    _holder.config = config
    # retriever.store is the GraphStore holding the per-category KnowledgeGraphs;
    # None if a non-graph retriever is configured (document graph then stays empty).
    _holder.graph_store = getattr(pipeline.retriever, "store", None)
    logger.info("Sentri pipeline ready. Listening for queries.")
    try:
        yield
    finally:
        logger.info("Shutting down Sentri pipeline.")
        # No explicit teardown needed — models live in-process and exit
        # cleanly when uvicorn does. Add explicit cleanup here if any
        # backend ever holds OS-level resources (sockets, file locks).


app = FastAPI(
    title="Sentri Agentic API",
    version="1.0.0",
    description=(
        "Stateless Q&A endpoint over the Sentri agentic pipeline. "
        "The caller (Spring Boot backend) owns sessions, history, and storage."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _augment_query(query: str, history: list[ChatMessage]) -> str:
    """Build the CURRENT QUESTION / PREVIOUS CONTEXT prompt the Director expects.

    Pairs each user message with the next assistant message in chronological
    order. Unpaired trailing user messages are dropped (they'd be the same
    as `query`, or noise). Mirrors the format
    `cmd_sentri_interactive._load_session_history` emits, so the Director's
    back-reference resolver finds the same turn boundaries.
    """
    if not history:
        return query

    lines: list[str] = ["[Session history from this interactive session]"]
    turn_idx = 1
    pending_q: str | None = None
    for msg in history:
        content = (msg.content or "").strip()
        if not content:
            continue
        if msg.role == "user":
            pending_q = content
        elif msg.role == "assistant" and pending_q is not None:
            lines.append(
                f"Turn {turn_idx} — Q: {pending_q}\n"
                f"Turn {turn_idx} — A: {content}"
            )
            turn_idx += 1
            pending_q = None

    if turn_idx == 1:
        return query

    lines.append("[End of session history]\n")
    history_block = "\n".join(lines)
    return (
        f"CURRENT QUESTION: {query}\n\n"
        f"PREVIOUS CONTEXT (for reference only — classify and answer based on "
        f"CURRENT QUESTION):\n{history_block}"
    )


def _extract_contract_ids(result: Any) -> list[str]:
    """Prefer FinalOutput.contract_ids; fall back to scanning evidence file_ids.

    Same precedence rule as cmd_sentri_interactive — response-named IDs
    reflect the answer's subject; evidence-derived IDs over-broaden when
    retrieval surfaced sibling contracts.
    """
    response_ids = list(getattr(result, "contract_ids", None) or [])
    if response_ids:
        return response_ids

    seen: list[str] = []
    for ev in getattr(result, "evidence", None) or []:
        file_id = getattr(ev, "file_id", "") or ""
        for m in _CONTRACT_ID_RE.findall(file_id):
            cid = m.upper()
            if cid not in seen:
                seen.append(cid)
    return seen


def _build_sources(result: Any) -> list[dict[str, Any]]:
    """Shape evidence for the UI's source-panel.

    Emits the ``{name, page, score, chunk_id}`` source-panel shape the
    Spring Boot side consumes, so it doesn't need a second adapter.
    """
    sources: list[dict[str, Any]] = []
    for ev in getattr(result, "evidence", None) or []:
        sources.append({
            "name": getattr(ev, "file_id", None),
            "page": getattr(ev, "page", None),
            "score": getattr(ev, "score", None),
            "chunk_id": getattr(ev, "chunk_id", None),
        })
    return sources


# Entities appearing in more than this many distinct chunks corpus-wide are
# boilerplate hubs (DPWH, Philippines, region names) — they co-occur in nearly
# every document, so they'd link everything to everything. Excluded from edges.
_HUB_CHUNK_CAP = 15
# Cap the shared-entity label list per edge so the payload stays small.
_MAX_SHARED_LABELS = 8


def _build_document_graph(result: Any, graph_store: Any) -> dict[str, Any]:
    """Project the answer's evidence into a document-relation graph.

    Nodes are the individual documents (``file_id``) cited in the evidence.
    An edge joins two documents that share named entities in the knowledge
    graph; edge weight is the sum of each shared entity's inverse corpus
    frequency (IDF-style), so a rare shared entity (a contract ID) links
    strongly while common ones contribute little. Entities above
    ``_HUB_CHUNK_CAP`` are dropped so boilerplate never manufactures edges.

    Uses the KGs already loaded in-process — no disk reads, no re-ingest.
    Degrades to ``{"nodes": [...], "edges": []}`` (or fully empty) when the
    graph store is absent or evidence chunk_ids don't resolve to entities.
    """
    kgs = getattr(graph_store, "kgs", None) or {}

    nodes: dict[str, dict[str, Any]] = {}
    # file_id -> {entity_name: corpus_df}
    doc_entities: dict[str, dict[str, int]] = {}

    for ev in getattr(result, "evidence", None) or []:
        file_id = getattr(ev, "file_id", None)
        chunk_id = getattr(ev, "chunk_id", None)
        if not file_id:
            continue
        node = nodes.setdefault(file_id, {"id": file_id, "label": file_id, "pages": set()})
        page = getattr(ev, "page", None)
        if page:
            node["pages"].add(int(page))
        if not chunk_id:
            continue

        # A chunk_id lives in exactly one category KG; get_chunk_entities
        # returns [] for the others.
        bucket = doc_entities.setdefault(file_id, {})
        for kg in kgs.values():
            for name in kg.get_chunk_entities(chunk_id):
                df = len(kg._entity_chunks.get(name, ()))
                if df <= 0 or df > _HUB_CHUNK_CAP:
                    continue
                if name not in bucket or df < bucket[name]:
                    bucket[name] = df

    edges: list[dict[str, Any]] = []
    linked = [f for f in nodes if doc_entities.get(f)]
    for i in range(len(linked)):
        for j in range(i + 1, len(linked)):
            a, b = linked[i], linked[j]
            shared = set(doc_entities[a]) & set(doc_entities[b])
            if not shared:
                continue
            weight = sum(1.0 / doc_entities[a][name] for name in shared)
            shared_sorted = sorted(shared, key=lambda n: doc_entities[a][n])
            edges.append({
                "source": a,
                "target": b,
                "weight": round(weight, 4),
                "shared": shared_sorted[:_MAX_SHARED_LABELS],
            })

    node_list = [
        {"id": n["id"], "label": n["label"], "pages": sorted(n["pages"])}
        for n in nodes.values()
    ]
    return {"nodes": node_list, "edges": edges}


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _summarize_stage(stage: str, data: dict[str, Any]) -> dict[str, Any]:
    """Compact per-stage payload for SSE — heavy fields (chunk lists, raw
    graph facts, retrieved text) stay server-side; only the shape the UI
    needs to render a progress line goes on the wire.
    """
    if stage == STAGE_DIRECTOR:
        return {"intent": data.get("intent"), "confidence": data.get("confidence")}
    if stage == STAGE_SEARCH:
        return {"chunks": data.get("total_retrieved", 0)}
    if stage == STAGE_RERANK:
        return {"chunks": len(data.get("chunks") or [])}
    if stage == STAGE_EXPERT_DISPATCH:
        return {"phase": data.get("phase"), "experts": data.get("experts") or []}
    if stage in (STAGE_EXPERT_RESULT, STAGE_EXPERT_RETRY):
        return {
            "phase": data.get("phase"),
            "expert": data.get("expert"),
            "confidence": data.get("confidence"),
            "findings": len(data.get("findings") or []),
            "error": data.get("error"),
        }
    if stage in (STAGE_COMPOSE, STAGE_DIRECT_ANSWER):
        return {"model": data.get("compose_model"), "backend": data.get("compose_backend")}
    return {}


class _StreamingTracer(SentriTracer):
    """SentriTracer subclass that pushes a compact summary of each stage
    event onto an asyncio.Queue while still recording the full event on
    the parent (kept in case future code wants to persist the trace).

    record() is called from the pipeline worker thread, so the put is
    scheduled on the event-loop thread via call_soon_threadsafe.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        super().__init__()
        self._loop = loop
        self._queue = queue

    def record(self, stage: str, data: dict[str, Any]) -> None:
        super().record(stage, data)
        summary = _summarize_stage(stage, data)
        # Tagged 3-tuple so the same queue can carry stage events alongside
        # the ("token", ...) / ("thinking", ...) / ("final_output", ...)
        # events the pipeline generator emits.
        self._loop.call_soon_threadsafe(
            self._queue.put_nowait, ("stage", stage, summary),
        )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    """Lightweight health probe. Returns 'ready' once the pipeline is built."""
    return {
        "status": "ready" if _holder.sentri is not None else "starting",
        "service": "sentri-api",
    }


@app.post("/query")
async def query(req: QueryRequest, request: Request) -> StreamingResponse:
    """Run one Sentri query and stream the answer back as SSE.

    Events emitted, in order:
      meta          — once, immediately after the request is accepted
      stage         — many, one per pipeline stage as it completes
                       (director, search, rerank, expert_dispatch,
                       expert_result, expert_retry, compose, direct_answer)
      thinking      — many, content inside <think>...</think> blocks emitted
                       by reasoning models (DeepSeek-R1 family etc.). Live.
      token         — many, model output tokens as the LLM generates them.
                       Live; this is the actual answer streamed character-by-
                       character (or chunk-by-chunk, depending on the backend).
      contract_ids  — once, after the model finishes; subjects of the answer
      sources       — once, after the model finishes; evidence for UI panel
      graph         — once, after the model finishes; document-relation graph
                       ({nodes, edges}) over the cited documents. Edges join
                       documents that share named entities (KG projection).
      done          — once, terminal. Includes the FINAL response_text after
                       all backstops ran — may differ from the concatenation
                       of streamed `token` events when a backstop modified it
                       (groundedness rejection, BoQ rescue, ambiguous-scope
                       refusal). Client decides whether to reconcile.
      error         — only on failure, terminal
    """
    sentri = _holder.require()
    augmented = _augment_query(req.query, req.history)
    logger.debug("Augmented query: %s", augmented)

    received_at = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Query received: session=%s history_turns=%d query=%r",
        req.session_id,
        len(req.history),
        req.query[:120],
    )

    async def event_stream() -> AsyncIterator[str]:
        yield _sse("meta", {
            "session_id": req.session_id,
            "received_at": received_at,
        })

        # Bridge the blocking generator `sentri.run_stream` (worker thread)
        # to this async generator via a shared queue. The same queue carries:
        #   ("stage", stage_name, summary)   — pushed by _StreamingTracer
        #   ("token", text)                  — yielded by run_stream
        #   ("thinking", text)               — yielded by run_stream
        #   ("final_output", FinalOutput)    — yielded by run_stream (terminal)
        #   ("error", Exception)             — pushed by worker on exception
        #   ("done", None)                   — pushed by worker after exit
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue = asyncio.Queue()
        tracer = _StreamingTracer(loop, event_queue)
        # Set when the client disconnects. run_stream polls this at stage
        # boundaries and between tokens to stop early (the worker thread can't
        # be force-killed, so cancellation is cooperative).
        cancel_event = threading.Event()

        def _drive_pipeline() -> None:
            try:
                for ev_type, payload in sentri.run_stream(
                    augmented, tracer, cancel_check=cancel_event.is_set,
                ):
                    loop.call_soon_threadsafe(
                        event_queue.put_nowait, (ev_type, payload),
                    )
            except Exception as exc:  # noqa: BLE001 - surfaced as SSE error
                logger.exception(
                    "Sentri pipeline failed for session=%s", req.session_id,
                )
                loop.call_soon_threadsafe(
                    event_queue.put_nowait, ("error", exc),
                )
            finally:
                loop.call_soon_threadsafe(
                    event_queue.put_nowait, ("done", None),
                )

        worker_task = asyncio.create_task(run_in_threadpool(_drive_pipeline))

        try:
            while True:
                try:
                    ev = await asyncio.wait_for(event_queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        cancel_event.set()
                        worker_task.cancel()
                        return
                    continue

                head = ev[0]

                if head == "done":
                    # Sentinel — worker finished (cleanly or via error already emitted).
                    return

                if head == "error":
                    yield _sse("error", {"message": str(ev[1])})
                    return

                if head == "stage":
                    _, stage_name, summary = ev
                    yield _sse("stage", {"stage": stage_name, **summary})
                    continue

                if head == "token":
                    yield _sse("token", {"text": ev[1]})
                    continue

                if head == "thinking":
                    yield _sse("thinking", {"text": ev[1]})
                    continue

                if head == "final_output":
                    final = ev[1]
                    contract_ids = _extract_contract_ids(final)
                    sources = _build_sources(final)
                    yield _sse("contract_ids", {"contract_ids": contract_ids})
                    yield _sse("sources", {"sources": sources})
                    graph = _build_document_graph(final, _holder.graph_store)
                    yield _sse("graph", graph)
                    audit = getattr(final, "audit", None)
                    yield _sse("done", {
                        "confidence": getattr(final, "confidence", None),
                        "uncertain": getattr(final, "uncertain", False),
                        "tokens": getattr(audit, "total_tokens", 0) if audit else 0,
                        "cost_usd": getattr(audit, "total_cost", 0.0) if audit else 0.0,
                        "files": list(getattr(final, "files", None) or []),
                        "compose_backend": getattr(final, "compose_backend", None),
                        "compose_model": getattr(final, "compose_model", None),
                        # Post-backstop final text. Compare against the streamed
                        # token concatenation client-side if you need to detect
                        # backstop corrections.
                        "response_text": getattr(final, "response_text", ""),
                    })
                    # Don't return — wait for the "done" sentinel from the worker
                    # so a late stage event (rare) still gets a chance to flush.
                    continue
        finally:
            cancel_event.set()
            if not worker_task.done():
                worker_task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable response buffering on intermediate proxies (nginx).
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# CLI entry point so `python -m sentri_api.server` works alongside uvicorn.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "sentri_api.server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )
