#!/usr/bin/env bash
# Print the model-free field-coverage report over the corpus manifest (rag.coverage).
# Pure Python: no model, no VRAM. Needs a built manifest (make build).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-sentri-backend}"
export PYTHONWARNINGS="ignore"
export PYTHONUTF8=1  # UTF-8 stdio + default file encoding on Windows (cp1252 otherwise)

python -m rag.coverage
