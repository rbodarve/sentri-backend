#!/usr/bin/env bash
# Measure retrieval recall (hit_rate / mrr) against eval/eval_retrieval.json.
# Usage: scripts/evaluate.sh [baseline|filtered|rerank]   (default: rerank)
#   baseline  pure vector search
#   filtered  + contract-id metadata filter
#   rerank    + CPU cross-encoder rerank (the 100%-recall config)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAG_CONDA_ENV:-ragtest}"
export PYTHONWARNINGS="ignore"

MODE="${1:-rerank}"
case "$MODE" in
  baseline) ;;
  filtered) export RAG_FILTER_BY_CONTRACT=1 ;;
  rerank)   export RAG_FILTER_BY_CONTRACT=1 RAG_RERANK=1 ;;
  *) echo "unknown mode: $MODE (use baseline|filtered|rerank)" >&2; exit 1 ;;
esac

python -m rag.evaluate
