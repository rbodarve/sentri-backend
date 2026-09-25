#!/usr/bin/env bash
# Ask the full RAG pipeline a question (retrieve + rerank + generate a grounded answer).
# Usage: scripts/ask.sh "Which contractor was awarded contract 24CC0265?"
# With no argument, runs a few demo questions.
# Requires a running Ollama server with the generation model pulled.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-sentri-backend}"
export PYTHONWARNINGS="ignore"

python -m rag.generate "$@"
