"""Streaming API for the ragtest agentic RAG pipeline.

Wires this workspace into the generic `streaming_transport` package by handing
it the five seams from `seams.py` — the transport itself is vendored unchanged
from `only_for_read/streaming_transport/` (you never edit it).

Run from INSIDE this folder so the installed `fastapi` library is not shadowed
by this folder's name (see serve.sh):

    cd fastapi && python -m uvicorn app:app --host 0.0.0.0 --port 8000

Then POST to /query:

    curl -N -X POST localhost:8000/query -H 'content-type: application/json' \
      -d '{"query": "Which contractor was awarded contract 24AJ0052?", "session_id": "s1"}'
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Expose the repo root so `import rag.*` resolves. Appended (not prepended) so
# the installed `fastapi` library still wins over this sibling folder named
# `fastapi` when the transport does `from fastapi import FastAPI`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

# rag/ resolves its artifacts (index_store/, manifest, trace) off PERSIST_DIR,
# which defaults to the *relative* "index_store" — correct only when cwd is the
# repo root. serve.sh runs uvicorn from inside this folder (to dodge the name
# shadow above), so pin PERSIST_DIR to an absolute path BEFORE rag is imported,
# making artifact loading independent of the process's working directory.
os.environ.setdefault("RAG_PERSIST_DIR", str(_REPO_ROOT / "index_store"))

# Import the transport (and thus the installed `fastapi` library) while cwd is
# still this folder, so the folder's name cannot shadow the library.
from streaming_transport import create_app
from seams import build_pipeline, finalize, passthrough_query, summarize_stage

app = create_app(
    service_name="ragtest-api",
    build_pipeline=build_pipeline,      # Seam 1
    summarize_stage=summarize_stage,    # Seam 4
    finalize=finalize,                  # Seam 5
    augment_query=passthrough_query,    # override: deterministic router wants a clean question
    startup_message="Loading index + reranker + LLM (first boot can take ~30s)…",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
