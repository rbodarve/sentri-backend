#!/usr/bin/env bash
# End-to-end simulation of a third-party device, no second device needed:
# starts serve_ngrok.sh (API + public tunnel), then runs remote_client.py against
# the PUBLIC ngrok URL (ngrok -> FastAPI -> agentic RAG -> FastAPI -> ngrok),
# prints each streamed event, checks the stream, and tears both down.
#
# Usage: bash service/simulate_remote.sh                  (default route-covering queries)
#        bash service/simulate_remote.sh "question" ...   (your own queries)
#
# Requires what serve_ngrok.sh requires (ragtest env + service deps, Ollama, ngrok authtoken).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$HERE/serve_ngrok.sh" &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true; wait "$SERVE_PID" 2>/dev/null || true' EXIT INT TERM

# serve_ngrok.sh only opens the tunnel once /health is ready, so a public URL on
# ngrok's inspection API (:4040) means the whole path is up.
until curl -sf http://127.0.0.1:4040/api/tunnels | grep -q '"public_url":"https://'; do
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "serve_ngrok.sh exited before the tunnel came up."; exit 1; }
  sleep 1
done

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-ragtest}"
QUERY_ARGS=()
for q in "$@"; do QUERY_ARGS+=(-q "$q"); done
python "$HERE/remote_client.py" "${QUERY_ARGS[@]}"
