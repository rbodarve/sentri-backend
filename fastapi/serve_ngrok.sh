#!/usr/bin/env bash
# Serve the ragtest agentic RAG SSE API (fastapi/app.py -> rag.agent, i.e. the
# same pipeline as `make chat-agentic`) and expose it publicly via an ngrok
# tunnel. Prints the public https URL once both the server and the tunnel are up.
#
# Usage: bash fastapi/serve_ngrok.sh          (defaults: PORT=8000)
#        PORT=9000 bash fastapi/serve_ngrok.sh
#
# Requires: the `ragtest` conda env with fastapi/uvicorn installed
#   (pip install -r fastapi/requirements.txt), a running Ollama server, and
#   ngrok with an authtoken configured (ngrok config add-authtoken <token>).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8000}"
NGROK_LOG="/tmp/ngrok_ragtest.log"

# Bind uvicorn to localhost only — ngrok is the public front door, so there's no
# reason to also listen on 0.0.0.0. serve.sh execs uvicorn, so $! is the server.
HOST=127.0.0.1 PORT="$PORT" bash "$HERE/serve.sh" &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  [[ -n "${NGROK_PID:-}" ]] && kill "$NGROK_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for the API to report ready — cold boot loads the embedder, reranker, and
# LLM once (~30s), and the tunnel is useless until /health flips to "ready".
echo "Waiting for the API on :$PORT to become ready (first boot ~30s)…"
until curl -sf "http://127.0.0.1:$PORT/health" | grep -qE '"status": ?"ready"'; do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "Server exited before becoming ready."; exit 1; }
  sleep 1
done
echo "API ready."

# Open the tunnel to the local port. ngrok's local inspection API on :4040 is the
# reliable way to read back the assigned public URL.
ngrok http "$PORT" --log=stdout >"$NGROK_LOG" 2>&1 &
NGROK_PID=$!

echo "Opening ngrok tunnel…"
PUBLIC_URL=""
for _ in $(seq 1 30); do
  PUBLIC_URL="$(curl -sf http://127.0.0.1:4040/api/tunnels 2>/dev/null \
    | grep -oE 'https://[a-zA-Z0-9.-]+' | head -1 || true)"
  [[ -n "$PUBLIC_URL" ]] && break
  kill -0 "$NGROK_PID" 2>/dev/null || { echo "ngrok exited early; see $NGROK_LOG"; exit 1; }
  sleep 1
done

if [[ -z "$PUBLIC_URL" ]]; then
  echo "Could not read the ngrok public URL; see $NGROK_LOG"
  exit 1
fi

cat <<EOF

  Public URL:  $PUBLIC_URL
  Health:      $PUBLIC_URL/health
  Query it:
    curl -N -X POST $PUBLIC_URL/query \\
      -H 'content-type: application/json' \\
      -d '{"query": "Which contractor was awarded contract 24AJ0052?", "session_id": "s1"}'

  ngrok inspector: http://127.0.0.1:4040
  Ctrl-C to stop both the server and the tunnel.

EOF

# Block on the server; the EXIT trap tears down ngrok when this returns.
wait "$SERVER_PID"
