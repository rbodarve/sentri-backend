#!/usr/bin/env bash
# Interactive RAG query loop: ask many questions in one session (index + LLM loaded once).
# Usage: scripts/chat.sh   (then type questions; empty line, 'exit', or Ctrl-D to quit)
# Requires a running Ollama server with the generation model pulled.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-ragtest}"
export PYTHONWARNINGS="ignore"

python -m rag.chat
