# CLAUDE.md

RAG pipeline over a corpus of DPWH (Philippine gov) construction-bid documents. It
retrieves and generates grounded, cited answers about contracts (bidders, awards, dates,
amounts) from OCR'd source PDFs. Built on LlamaIndex; runs entirely on CPU + local Ollama.

## Environment (read first)

- **Everything runs in the `ragtest` conda env.** `python -m rag.*` fails from base — the
  `scripts/*.sh` wrappers auto-activate it (override with `RAG_CONDA_ENV`). Run commands via
  `make`/scripts, or `conda activate ragtest` first.
- **CPU-only, zero-VRAM by design.** torch is installed CPU-only (`scripts/setup.sh`).
- **Generation needs a running Ollama server** with the model pulled
  (`ollama pull granite4.1:3b`). Retrieval/eval do not need Ollama.

## Commands

Use `make` (details in [scripts/README.md](scripts/README.md)):

- `make setup` — create the conda env + install pinned deps (one-time).
- `make check` — stage 1–3 self-checks (loader, enrich, relationships); pure Python, no model.
- `make build` — build the eval set + embed & persist the vector index to `index_store/`.
- `make eval [MODE=baseline|filtered|rerank]` — measure retrieval recall; default `rerank`.
- `make ask Q="..."` — one-shot retrieve + rerank + generate (needs Ollama).
- `make chat` — interactive query loop.
- `make clean` — remove the regenerable `index_store/` + caches.

## Architecture

Pipeline stages, each a module in [rag/](rag/) runnable as `python -m rag.<name>`:

`loader` → `enrich` → `relationships` (stages 1–3, validated by `make check`) →
`build_eval` + `index` + `manifest` (`make build`) → `evaluate` (recall) / `generate`,
`chat` (answers). `rerank.py` is the CPU cross-encoder stage; `config.py` is the single
config surface; `verify.py`/`trace.py` are support libs.

Data & artifacts:

- **`database/*.json` is the authoritative OCR transcription.** Chunks are UUID-keyed under
  `text`/`table`/`image`/`signature`, each with a `content` field. **Never re-OCR the
  `source/` PDFs to "verify" the database** — the DB is the trusted source of truth.
- `source/` — original bid PDFs (corpus input). `eval/eval_set.json` — ground-truth Q/chunk
  pairs. `index_store/` — the built vector index; **regenerable, gitignored** (`make build`).

Models are **config-driven** in [rag/config.py](rag/config.py) via `RAG_*` env vars — swap
embedding/reranker/LLM for better hardware without touching pipeline code. Defaults:
`BAAI/bge-small-en-v1.5` embed, `BAAI/bge-reranker-base` rerank, `granite4.1:3b` gen
(temp 0.0, `num_ctx` 4096 — capped deliberately to keep the KV cache on a 4GB GPU).

## Conventions

- **"Pass" = retrieval recall**, not answer prose. The headline result is
  `make eval` in `rerank` mode → `hit_rate 1.000` (contract-filtered + CPU rerank);
  `baseline`/`filtered` are ablations. When changing retrieval, re-run `make eval` and keep
  recall at 1.000.
- **Verify gate:** `.claude/verify.sh` (a Stop hook) validates `database/` OCR integrity on
  every turn. It must exit 0. It only checks JSON structure — it never re-OCRs.
- Match existing style; keep changes surgical. `PYTHONWARNINGS=ignore` is set by the scripts.
