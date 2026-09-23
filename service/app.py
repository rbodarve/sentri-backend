"""Streaming API for the ragtest agentic RAG pipeline.

Wires this workspace into the generic `streaming_transport` package by handing
it the five seams from `seams.py` — the transport itself (the `streaming_transport/`
package in this folder) is domain-neutral and vendored unchanged (you never edit it).

Run from INSIDE this folder so the flat sibling imports (seams, pipeline,
trace_recorder, streaming_transport) resolve (see serve.sh):

    cd service && python -m uvicorn app:app --host 0.0.0.0 --port 8000

Then POST to /query:

    curl -N -X POST localhost:8000/query -H 'content-type: application/json' \
      -d '{"query": "Which contractor was awarded contract 24AJ0052?", "session_id": "s1"}'
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# The transport logs each query via logging.getLogger("streaming_transport") at
# INFO, but nothing configures a handler, so those records fall through to the
# root logger (default WARNING) and are dropped — only uvicorn's access log
# shows. Configure the root logger at INFO so "Query received: query=..." emits.
logging.basicConfig(level=logging.INFO)

# Expose the repo root so `import rag.*` resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

# rag/ resolves its artifacts (index_store/, manifest, trace) off PERSIST_DIR,
# which defaults to the *relative* "index_store" — correct only when cwd is the
# repo root. serve.sh runs uvicorn from inside this folder, so pin PERSIST_DIR to an absolute path BEFORE rag is imported,
# making artifact loading independent of the process's working directory.
os.environ.setdefault("RAG_PERSIST_DIR", str(_REPO_ROOT / "index_store"))

from streaming_transport import create_app
from seams import build_pipeline, finalize, passthrough_query, summarize_stage
from trace_recorder import SessionStore, TraceStore, TracingMiddleware, add_trace_route

app = create_app(
    service_name="ragtest-api",
    build_pipeline=build_pipeline,      # Seam 1
    summarize_stage=summarize_stage,    # Seam 4
    finalize=finalize,                  # Seam 5
    augment_query=passthrough_query,    # override: deterministic router wants a clean question
    startup_message="Loading index + reranker + LLM (first boot can take ~30s)…",
)

# Disk-backed, time-bounded trace of every /query and its full SSE stream.
# Read at GET /trace and /trace/{n}; finished records are appended to
# service/traces.jsonl (gitignored) and survive a restart, with anything older
# than the retention window (7 days) pruned on startup. Wrap AFTER the FastAPI
# app is built so /trace is reachable through the middleware (which only taps
# POST /query and passes everything else straight through).
_trace_store = TraceStore()
_session_store = SessionStore()
add_trace_route(app, _trace_store)
app = TracingMiddleware(app, _trace_store, _session_store)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
