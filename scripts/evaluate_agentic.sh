#!/usr/bin/env bash
# Answer-level eval for the agentic controller: routing, fan-out, semantic decomposition, and
# verifier-driven withholding, scored against eval/eval_agentic.json.
# Usage: scripts/evaluate_agentic.sh
# Unlike scripts/evaluate.sh (retrieval recall, zero-VRAM), this generates answers and so
# requires a running Ollama server with the generation model pulled.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-ragtest}"
export PYTHONWARNINGS="ignore"

python -m rag.evaluate_agentic
