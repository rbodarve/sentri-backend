"""Disk-backed, time-bounded trace of every /query and its full SSE stream.

Taps the ONE choke point that sees everything — the ASGI boundary — so the
vendored `streaming_transport` stays untouched and the seams (which never see
token events) don't have to. A pure ASGI middleware tees the request body (the
query) and every outgoing SSE frame (meta / stage / thinking / token / sources /
done / error) into a bounded in-memory ring, and appends each finished record to
`traces.jsonl` so it survives a restart. Retention is time-based: records older
than `_MAX_AGE_DAYS` (7) are dropped from disk on startup and never served, so a
trace lives at most a week regardless of how many times the process restarts.

Read it back at GET /trace (all records) or GET /trace/{n} (one record).
"""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("streaming_transport")

# How many recent /query requests to keep in memory. Bounds RAM over a long
# session while keeping the whole conversation's worth of traces in practice.
_MAX_RECORDS = 500

# How long a persisted trace survives. Records older than this (by started_at)
# are pruned from disk on startup and filtered out of every read.
_MAX_AGE_DAYS = 7

# Where finished records are appended, one JSON object per line. Lives next to
# this module; gitignored.
_TRACE_PATH = Path(__file__).with_name("traces.jsonl")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceStore:
    """A bounded ring of per-request trace records, newest last, disk-backed.

    Finished records are appended to `path`; on startup the file is reloaded and
    rewritten with anything older than `max_age_days` dropped, and reads filter
    by age too — so no trace older than the window is ever kept or served.
    """

    def __init__(
        self,
        path: Path = _TRACE_PATH,
        maxlen: int = _MAX_RECORDS,
        max_age_days: int = _MAX_AGE_DAYS,
    ) -> None:
        self._path = path
        self._max_age = timedelta(days=max_age_days)
        self._records: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._seq = 0
        # finish() runs in a threadpool worker (see TracingMiddleware), so concurrent
        # requests could interleave appends and corrupt a JSONL line — serialize them.
        self._write_lock = threading.Lock()
        self._load()

    def _fresh(self, rec: dict[str, Any]) -> bool:
        """True if the record is within the retention window (fail-open)."""
        cutoff = datetime.now(timezone.utc) - self._max_age
        try:
            return datetime.fromisoformat(rec["started_at"]) >= cutoff
        except (KeyError, ValueError, TypeError):
            return True

    def _load(self) -> None:
        """Reload surviving records and rewrite the file without expired ones."""
        if not self._path.exists():
            return
        kept: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if self._fresh(rec):
                kept.append(rec)
        self._path.write_text(
            "".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8"
        )
        for rec in kept[-(self._records.maxlen or len(kept)):]:
            self._records.append(rec)
        self._seq = max((r.get("n", 0) for r in kept), default=0)

    def start(self) -> dict[str, Any]:
        self._seq += 1
        rec: dict[str, Any] = {
            "n": self._seq,
            "started_at": _now(),
            "finished_at": None,
            "session_id": None,
            "query": None,
            "events": [],   # [{"t", "event", "data"}]
        }
        self._records.append(rec)
        return rec

    def finish(self, rec: dict[str, Any]) -> None:
        """Persist a completed record — appended once, when the stream ends.

        Runs off the event loop (threadpool), so guard the append with a lock to
        keep concurrent writes from interleaving into a corrupt line."""
        with self._write_lock, self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def as_list(self) -> list[dict[str, Any]]:
        return [r for r in self._records if self._fresh(r)]

    def get(self, n: int) -> dict[str, Any] | None:
        return next(
            (r for r in self._records if r["n"] == n and self._fresh(r)), None
        )


def _parse_sse_frame(frame: bytes) -> dict[str, Any] | None:
    """Turn one raw SSE frame (`event: X\\ndata: {...}`) into {event, data}."""
    event = "message"
    data_lines: list[str] = []
    for line in frame.decode("utf-8", "replace").splitlines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if not data_lines:
        return None
    raw = "\n".join(data_lines)
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        data = raw
    return {"t": _now(), "event": event, "data": data}


class TracingMiddleware:
    """Pure ASGI middleware — tees /query without buffering the stream.

    Only POST /query is intercepted; everything else (including GET /trace) is
    passed straight through. The response is forwarded chunk-by-chunk as it
    arrives, so streaming is preserved — capture is a side effect, never a gate.
    """

    def __init__(self, app: Any, store: TraceStore) -> None:
        self.app = app
        self.store = store

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST" \
                or scope.get("path") != "/query":
            await self.app(scope, receive, send)
            return

        rec = self.store.start()
        req_body = bytearray()

        async def recv() -> dict[str, Any]:
            msg = await receive()
            if msg["type"] == "http.request":
                req_body.extend(msg.get("body", b""))
                if not msg.get("more_body", False):
                    _fill_query(rec, bytes(req_body))
            return msg

        sse_buf = bytearray()

        async def snd(msg: dict[str, Any]) -> None:
            if msg["type"] == "http.response.body":
                sse_buf.extend(msg.get("body", b""))
                while b"\n\n" in sse_buf:
                    frame, _, remainder = sse_buf.partition(b"\n\n")
                    sse_buf[:] = remainder
                    parsed = _parse_sse_frame(frame)
                    if parsed is not None:
                        rec["events"].append(parsed)
                if not msg.get("more_body", False):
                    rec["finished_at"] = _now()
                    # Offload the synchronous append so the one disk write per
                    # request never blocks the event loop mid-stream.
                    await run_in_threadpool(self.store.finish, rec)
                    logger.info(
                        "Trace #%d captured: session=%s events=%d",
                        rec["n"], rec["session_id"], len(rec["events"]),
                    )
            await send(msg)

        await self.app(scope, recv, snd)


def _fill_query(rec: dict[str, Any], body: bytes) -> None:
    try:
        payload = json.loads(body)
        rec["query"] = payload.get("query")
        rec["session_id"] = payload.get("session_id")
    except (json.JSONDecodeError, AttributeError):
        rec["query"] = body.decode("utf-8", "replace")


def add_trace_route(app: FastAPI, store: TraceStore) -> None:
    """Register the read-back endpoints on the FastAPI app."""

    @app.get("/trace")
    async def trace() -> dict[str, Any]:
        records = store.as_list()
        return {
            "count": len(records),
            "records": [
                {
                    "n": r["n"],
                    "session_id": r["session_id"],
                    "query": r["query"],
                    "started_at": r["started_at"],
                    "finished_at": r["finished_at"],
                    "events": len(r["events"]),
                    # Reconstructed answer text — the token frames rejoined, so a
                    # trace is readable without stitching hundreds of frames.
                    "answer": "".join(
                        e["data"].get("text", "")
                        for e in r["events"]
                        if e["event"] == "token" and isinstance(e["data"], dict)
                    ),
                }
                for r in records
            ],
        }

    @app.get("/trace/{n}")
    async def trace_one(n: int) -> dict[str, Any]:
        rec = store.get(n)
        if rec is None:
            return {"error": f"no trace #{n} (retained: last {_MAX_RECORDS})"}
        return rec
