# Workspace commands

Portable shell wrappers for the RAG pipeline. Run from anywhere; each script locates the
repo root and activates the `ragtest` conda env (override with `RAG_CONDA_ENV`). Also
available as `make <target>`.

| Command | Make | What it does |
|---|---|---|
| `bash scripts/setup.sh` | `make setup` | Create conda env `ragtest`, install pinned deps (CPU-only torch, zero VRAM) |
| `bash scripts/check.sh` | `make check` | Run stage 1-3 self-checks (loader, enrich, relationships) |
| `bash scripts/build.sh` | `make build` | Build the retrieval eval set + embed and persist the vector index |
| `bash scripts/evaluate.sh [mode]` | `make eval [MODE=...]` | Measure recall; `mode` = `baseline` \| `filtered` \| `rerank` (default `rerank`) |
| `bash scripts/ask.sh "<question>"` | `make ask Q="..."` | One-shot: retrieve + rerank + generate a grounded, cited answer (no arg = demo questions) |
| `bash scripts/chat.sh` | `make chat` | Interactive query loop: ask many questions in one session (index + LLM loaded once) |
| `bash scripts/agent.sh "<question>"` | `make agent Q="..."` | Agentic controller: router + fan-out + semantic decomposition + verifier-driven self-correction (no arg = demo questions) |
| `bash scripts/chat_agentic.sh` | `make chat-agentic` | Interactive query loop routed through the agentic controller (index + LLM loaded once) |
| `bash scripts/evaluate_agentic.sh` | `make eval-agentic` | Answer-level eval of the agent (routing / fan-out / decomposition / withholding); needs Ollama |
| — | `make all` | check + build + eval |

## Typical first run
```bash
make setup      # one-time
make build      # embed index + build retrieval eval set
make eval       # rerank mode -> hit_rate 1.000
```

## Notes
- `evaluate.sh rerank` reproduces the 100% retrieval-recall result (contract-filtered +
  CPU cross-encoder rerank). `baseline` and `filtered` show the ablations.
- Model choice is config-driven in `rag/config.py` (or `RAG_*` env vars) — swap the
  embedding/reranker models for better hardware without touching pipeline code.
