#!/usr/bin/env bash
# Print the model-free field-coverage report over the corpus manifest (rag.coverage).
# Pure Python: no model, no VRAM. Needs a built manifest (make build).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-sentri-backend}"
export PYTHONWARNINGS="ignore"

python -m rag.coverage
