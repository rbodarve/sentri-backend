#!/usr/bin/env bash
# Verify pipeline stages 1-3 (loader, enrich, relationships) via their self-checks, plus the
# signatory-parser agreement pin (rag.verify). Pure Python: no model, no VRAM.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-sentri-backend}"
export PYTHONWARNINGS="ignore"

python -m rag.loader
python -m rag.enrich
python -m rag.relationships
python -m rag.verify
