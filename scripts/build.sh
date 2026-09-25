#!/usr/bin/env bash
# Build the artifacts: the retrieval eval set (Step 5) and the vector index (Step 4).
# The index build embeds all chunks once (CPU) and persists to index_store/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-sentri-backend}"
export PYTHONWARNINGS="ignore"

python -m rag.build_retrieval_eval
python -m rag.index
python -m rag.manifest
