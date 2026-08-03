#!/usr/bin/env bash
# Create the conda env and install pinned dependencies.
# torch is installed CPU-only (no CUDA wheels) so the pipeline uses zero VRAM.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${RAG_CONDA_ENV:-ragtest}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python=3.13
conda activate "$ENV_NAME"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
echo "setup complete: conda env '$ENV_NAME' ready"
