#!/usr/bin/env bash
# Ask the agentic controller a question (router -> fan-out / semantic decomposition ->
# verifier-driven self-correction over the single-pass RAG).
# Usage: scripts/agent.sh "Who is the District Engineer for contracts 24BJ0005 and 24CC0265?"
# With no argument, runs a few demo questions.
# Requires a running Ollama server with the generation model pulled.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-sentri-backend}"
export PYTHONWARNINGS="ignore"
export PYTHONUTF8=1  # UTF-8 stdio + default file encoding on Windows (cp1252 otherwise)

python -m rag.agent "$@"
