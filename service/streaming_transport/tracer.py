"""Stage-tracer plumbing for the streaming transport.

`StreamTracer` is the base your pipeline calls: `record(stage, data)` once per
stage as it completes. `QueueTracer` is the transport-owned subclass that turns
each stage event into a compact SSE `stage` frame — it is fully generic and you
should not need to touch it. What each stage *summarizes to* is your seam: pass
a `summarize_stage(stage, data) -> dict` function into the transport.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable


class StreamTracer:
    """Base stage tracer. The pipeline calls `record(stage, data)` as each
    stage completes; the base just accumulates the full events (kept in case
    you want to persist or inspect the trace). Subclasses forward them.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def record(self, stage: str, data: dict[str, Any]) -> None:
        self.events.append((stage, data))


class QueueTracer(StreamTracer):
    """Pushes a compact per-stage summary onto an asyncio.Queue while still
    recording the full event on the base.

    `record()` is invoked from the pipeline worker thread, so the queue put is
    scheduled onto the event-loop thread via `call_soon_threadsafe`. The tagged
    3-tuple `("stage", name, summary)` shares one queue with the
    `("token", ...)` / `("thinking", ...)` / `("final_output", ...)` events the
    pipeline generator yields.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: "asyncio.Queue[Any]",
        summarize: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> None:
        super().__init__()
        self._loop = loop
        self._queue = queue
        self._summarize = summarize

    def record(self, stage: str, data: dict[str, Any]) -> None:
        super().record(stage, data)
        summary = self._summarize(stage, data)
        self._loop.call_soon_threadsafe(
            self._queue.put_nowait, ("stage", stage, summary),
        )
