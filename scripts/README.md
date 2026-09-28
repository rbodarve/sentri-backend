# Workspace commands

Portable shell wrappers for the RAG pipeline. Run from anywhere; each script locates the
repo root and activates the `sentri-backend` conda env (override with `RAG_CONDA_ENV`). Also
available as `make <target>`.

| Command | Make | What it does |
|---|---|---|
| `bash scripts/setup.sh` | `make setup` | Create conda env `sentri-backend`, install pinned deps (CPU-only torch, zero VRAM) |
| `bash scripts/check.sh` | `make check` | Run stage 1-3 self-checks (loader, enrich, relationships) |
| `bash scripts/coverage.sh` | `make coverage` | Model-free field-coverage report over the built manifest (the analytical route's data ceiling) |
| `bash scripts/build.sh` | `make build` | Build the retrieval eval set + embed and persist the vector index, then extract the corpus manifest (manifest step needs Ollama) |
| `bash scripts/evaluate.sh [mode]` | `make eval [MODE=...]` | Measure recall; `mode` = `baseline` \| `filtered` \| `rerank` (default `rerank`) |
| `bash scripts/ask.sh "<question>"` | `make ask Q="..."` | One-shot: retrieve + rerank + generate a grounded, cited answer (no arg = demo questions) |
| `bash scripts/chat.sh` | `make chat` | Interactive query loop: ask many questions in one session (index + LLM loaded once) |
| `bash scripts/agent.sh "<question>"` | `make agent Q="..."` | Agentic controller: router + fan-out + semantic decomposition + verifier-driven self-correction (no arg = demo questions) |
| `bash scripts/chat_agentic.sh` | `make chat-agentic` | Interactive query loop routed through the agentic controller; carries the conversation's contract into follow-ups (`rag/subject.py`) |
| `bash scripts/evaluate_agentic.sh` | `make eval-agentic` | Answer-level eval of the agent (routing / fan-out / decomposition / withholding); needs Ollama |
| `bash service/serve.sh` | `make serve [HOST=.. PORT=..]` | Serve the agent as a streaming SSE API (see `service/README.md`); needs Ollama + `pip install -r service/requirements.txt` |
| `bash service/serve_ngrok.sh` | `make serve-ngrok` | Same API behind an ngrok public tunnel |
| `bash service/simulate_remote.sh` | `make simulate-remote [Q="..."]` | `serve-ngrok` + a simulated remote client that checks the stream contract |
| — | `make all` | check + build + eval |
| — | `make clean` | Remove the regenerable `index_store/` + caches |
| — | `make help` | List the targets |

## Typical first run
```bash
make setup                 # one-time
ollama pull granite4.1:3b  # generation model; make build's manifest step needs a running Ollama
make build                 # retrieval eval set + embed index + extract manifest
make eval                  # rerank mode -> hit_rate 1.000
```

## Notes
- `evaluate.sh rerank` reproduces the 100% retrieval-recall result (contract-filtered +
  CPU cross-encoder rerank). `baseline` and `filtered` show the ablations.
- Model choice is config-driven in `rag/config.py` (or `RAG_*` env vars) — swap the
  embedding/reranker models for better hardware without touching pipeline code.
