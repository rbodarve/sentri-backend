#!/usr/bin/env bash
# Verify pipeline stages 1-3 (loader, enrich, relationships) via their self-checks, plus the
# signatory-parser agreement pin (rag.verify), and the router gate. Pure Python: no model, no VRAM.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-sentri-backend}"
export PYTHONWARNINGS="ignore"
export PYTHONUTF8=1  # UTF-8 stdio + default file encoding on Windows (cp1252 otherwise)

python -m rag.loader
python -m rag.enrich
python -m rag.relationships
python -m rag.verify
python -m rag.manifest --check
# Router gate (model-free): every eval/eval_agentic.json question must route to its declared kind.
# It needs the built manifest, which a fresh clone lacks until `make build` -- skip, don't fail.
if [ -f "${RAG_PERSIST_DIR:-index_store}/manifest.json" ]; then
  python -m rag.evaluate_agentic --routes
else
  echo "routes: skipped (no manifest; run make build)"
fi
