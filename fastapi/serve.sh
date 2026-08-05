#!/usr/bin/env bash
# Serve the ragtest agentic RAG pipeline as a streaming (SSE) HTTP API.
# Usage: bash fastapi/serve.sh            (defaults: HOST=0.0.0.0 PORT=8000)
#        HOST=127.0.0.1 PORT=9000 bash fastapi/serve.sh
# Requires: the `ragtest` conda env with fastapi/uvicorn installed
#   (pip install -r fastapi/requirements.txt), plus a running Ollama server.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Run from inside fastapi/ so the installed `fastapi` library is not shadowed by
# this folder's name; app.py adds the repo root back for `import rag.*`.
cd "$HERE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-ragtest}"
export PYTHONWARNINGS="ignore"

exec python -m uvicorn app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
