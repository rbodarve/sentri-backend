"""Generic FastAPI + Server-Sent-Events transport for a streaming RAG/agent
pipeline.

This is the ~70% of an external agent service's SSE server that has NO dependency
on that service, RAG, DPWH, or any knowledge-graph shape. It owns the hard parts:

  * build-once-at-startup lifespan + a singleton holder
  * the blocking-generator → async-queue bridge (pipeline runs in a threadpool
    worker, the async side drains a queue and frames each item as SSE)
  * cooperative cancellation when the client disconnects (the worker thread
    can't be force-killed, so it polls a threading.Event at stage/token
    boundaries)
  * SSE framing, the streaming-response headers, and a /health probe

Everything domain-specific is injected as one of five seams (see `create_app`).
To stand up a second service you implement those five callables against your
own pipeline and hand them here — you never edit this file.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Iterator, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from .schemas import ChatMessage, QueryRequest
from .tracer import QueueTracer, StreamTracer

logger = logging.getLogger("streaming_transport")


# ---------------------------------------------------------------------------
# The contract your pipeline must satisfy (Seam 2 — streaming protocol)
# ---------------------------------------------------------------------------

class StreamingPipeline(Protocol):
    """The only method the transport calls on your pipeline object.

    `run_stream` is a *blocking* generator (it runs in a worker thread). It
    yields tagged 2-tuples in this vocabulary, in order:

        ("thinking", str)        # reasoning-model <think> content (optional)
        ("token", str)           # answer tokens as they generate
        ("final_output", object) # terminal; the object handed to `finalize`

    It must call `tracer.record(stage, data)` as each internal stage finishes,
    and poll `cancel_check()` at stage/token boundaries — when it returns True
    the client has gone; stop as soon as it's safe.
    """

    def run_stream(
        self,
        query: str,
        tracer: StreamTracer,
        cancel_check: Callable[[], bool],
    ) -> Iterator[tuple[str, Any]]:
        ...


# ---------------------------------------------------------------------------
# Optional default for the query-augmentation seam
# ---------------------------------------------------------------------------

def default_augment_query(query: str, history: list[ChatMessage]) -> str:
    """Fold chat history into a CURRENT QUESTION / PREVIOUS CONTEXT prompt.

    Pairs each user message with the next assistant message and numbers the
    turns. Reusable as-is ONLY if your pipeline's query understanding expects
    this exact framing; otherwise pass your own `augment_query` to `create_app`.
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
            lines.append(f"Turn {turn_idx} — Q: {pending_q}\nTurn {turn_idx} — A: {content}")
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


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class _PipelineHolder:
    """Holds the singleton pipeline + opaque startup context after lifespan."""

    def __init__(self) -> None:
        self.pipeline: StreamingPipeline | None = None
        # Whatever your `build_pipeline` returned as its second value (e.g. a
        # graph store) — passed straight to `finalize`. None if unused.
        self.context: Any = None

    def require(self) -> StreamingPipeline:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized — lifespan startup did not run.")
        return self.pipeline


# Types for the injected seams, for reference:
#   build_pipeline  () -> tuple[StreamingPipeline, Any]      (Seam 1)
#   summarize_stage (stage: str, data: dict) -> dict         (Seam 4)
#   finalize        (final_output, context) -> list[(event_name, data)]  (Seam 5)
#   augment_query   (query: str, history: list[ChatMessage]) -> str

def create_app(
    *,
    service_name: str,
    build_pipeline: Callable[[], "tuple[StreamingPipeline, Any]"],
    summarize_stage: Callable[[str, dict[str, Any]], dict[str, Any]],
    finalize: Callable[[Any, Any], "list[tuple[str, dict[str, Any]]]"],
    augment_query: Callable[[str, list[ChatMessage]], str] = default_augment_query,
    startup_message: str = "Building pipeline (this can take ~30s on first boot)…",
) -> FastAPI:
    """Wire the generic transport around your pipeline.

    Seams (the ~30% that is yours to implement):
      build_pipeline   — Seam 1. Build heavy state once at startup. Returns
                         `(pipeline, context)`; `pipeline.run_stream` must
                         exist; `context` is opaque and forwarded to finalize.
                         Runs in a threadpool so model loads don't block boot.
      run_stream       — Seam 2. On the returned pipeline (see StreamingPipeline).
      tracer           — Seam 3. The transport hands your run_stream a
                         `QueueTracer`; your pipeline just calls .record(...).
      summarize_stage  — Seam 4. `(stage, data) -> compact dict` put on the
                         wire as `event: stage`. Heavy fields stay server-side.
      finalize         — Seam 5. `(final_output, context) -> [(event, data)]`,
                         the terminal projection. The transport emits each in
                         order (e.g. sources, graph, done). Owns the `done`
                         event and its payload — the transport hardcodes no
                         domain event names.
    """
    holder = _PipelineHolder()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info("%s %s", service_name, startup_message)
        pipeline, context = await run_in_threadpool(build_pipeline)
        holder.pipeline = pipeline
        holder.context = context
        logger.info("%s ready. Listening for queries.", service_name)
        try:
            yield
        finally:
            logger.info("%s shutting down.", service_name)

    app = FastAPI(title=service_name, version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ready" if holder.pipeline is not None else "starting",
            "service": service_name,
        }

    @app.post("/query")
    async def query(req: QueryRequest, request: Request) -> StreamingResponse:
        pipeline = holder.require()
        augmented = augment_query(req.query, req.history)
        received_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Query received: session=%s history_turns=%d query=%r",
            req.session_id, len(req.history), req.query[:120],
        )

        async def event_stream() -> AsyncIterator[str]:
            yield _sse("meta", {"session_id": req.session_id, "received_at": received_at})

            # Bridge the blocking generator (worker thread) to this async
            # generator via a shared queue. The queue carries, tagged by head:
            #   ("stage", name, summary)   — pushed by QueueTracer
            #   ("token", text)            — yielded by run_stream
            #   ("thinking", text)         — yielded by run_stream
            #   ("final_output", obj)      — yielded by run_stream (terminal)
            #   ("error", Exception)       — pushed by the worker on exception
            #   ("done", None)             — pushed by the worker after exit
            loop = asyncio.get_running_loop()
            event_queue: "asyncio.Queue[Any]" = asyncio.Queue()
            tracer = QueueTracer(loop, event_queue, summarize_stage)
            # Set when the client disconnects; run_stream polls it to stop early
            # (a worker thread can't be force-killed — cancellation is cooperative).
            cancel_event = threading.Event()

            def _drive_pipeline() -> None:
                try:
                    for ev_type, payload in pipeline.run_stream(
                        augmented, tracer, cancel_check=cancel_event.is_set,
                    ):
                        loop.call_soon_threadsafe(
                            event_queue.put_nowait, (ev_type, payload),
                        )
                except Exception as exc:  # noqa: BLE001 - surfaced as SSE error
                    logger.exception("Pipeline failed for session=%s", req.session_id)
                    loop.call_soon_threadsafe(event_queue.put_nowait, ("error", exc))
                finally:
                    loop.call_soon_threadsafe(event_queue.put_nowait, ("done", None))

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
                        # Seam 5: your projection owns every terminal event,
                        # including `done`. Don't `return` here — keep draining
                        # so a late stage event still flushes before the sentinel.
                        for ev_name, ev_data in finalize(ev[1], holder.context):
                            yield _sse(ev_name, ev_data)
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
                # Disable response buffering on intermediate proxies (nginx/ngrok).
                "X-Accel-Buffering": "no",
            },
        )

    return app
