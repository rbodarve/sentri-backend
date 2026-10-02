#!/usr/bin/env bash
# Interactive agentic RAG query loop: each question is routed through the agentic controller
# (router + fan-out + semantic decomposition + verifier-driven self-correction).
# Usage: scripts/chat_agentic.sh   (then type questions; empty line, 'exit', or Ctrl-D to quit)
# Requires a running Ollama server with the generation model pulled.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-sentri-backend}"
export PYTHONWARNINGS="ignore"
export PYTHONUTF8=1  # UTF-8 stdio + default file encoding on Windows (cp1252 otherwise)

python -m rag.chat_agentic
