#!/usr/bin/env bash
# Serve the sentri-backend agentic RAG pipeline as a streaming (SSE) HTTP API.
# Usage: bash service/serve.sh            (defaults: HOST=0.0.0.0 PORT=8000)
#        HOST=127.0.0.1 PORT=9000 bash service/serve.sh
# Requires: the `sentri-backend` conda env with fastapi/uvicorn installed
#   (pip install -r service/requirements.txt), plus a running Ollama server.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Run from inside service/ so app.py's flat imports (seams, pipeline, ...) resolve;
# app.py adds the repo root back for `import rag.*`.
cd "$HERE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-sentri-backend}"
export PYTHONWARNINGS="ignore"

exec python -m uvicorn app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
