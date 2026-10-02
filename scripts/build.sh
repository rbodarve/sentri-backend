#!/usr/bin/env bash
# Build the artifacts: the retrieval eval set (Step 5), the vector index (Step 4), and the
# corpus manifest (Step 9). The index build embeds all chunks once (CPU) and persists to
# index_store/; the manifest step then makes one LLM call per contract, so it needs a running
# Ollama server -- and runs last, after the (slow) embedding.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-sentri-backend}"
export PYTHONWARNINGS="ignore"
export PYTHONUTF8=1  # UTF-8 stdio + default file encoding on Windows (cp1252 otherwise)

python -m rag.build_retrieval_eval
python -m rag.index
python -m rag.manifest
