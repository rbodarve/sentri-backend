# service — streaming API for the ragtest agentic RAG pipeline

Wires this workspace into the reusable SSE transport in
[`streaming_transport/`](streaming_transport/). That package is domain-neutral and
vendored **unchanged** (you never edit it); the workspace-specific ~30% is the
five seams in [seams.py](seams.py) + [pipeline.py](pipeline.py).

It serves the same agentic controller as `make agent` / `make chat-agentic`
(router → fan-out / semantic decomposition → verifier-driven self-correction over
the single-pass RAG), streamed over HTTP as Server-Sent Events.

## Layout

| File | Role |
|---|---|
| `streaming_transport/` | Vendored generic transport (FastAPI app + SSE bridge + cancel-on-disconnect). Domain-neutral — you never edit it. |
| `pipeline.py` | **Seam 2/3** — `StreamingAgenticRag` wraps `rag.agent.AgenticRag` in the `run_stream(query, tracer, cancel_check)` contract and records per-stage progress. |
| `seams.py` | **Seams 1, 4, 5 + augment** — `build_pipeline`, `summarize_stage`, `finalize`, `passthrough_query`. |
| `app.py` | `create_app(...)` wiring + `uvicorn` entrypoint. |
| `trace_recorder.py` | Session-history replay (`SessionStore`) + query tracing: `TracingMiddleware` taps `POST /query` and appends each finished record to `traces.jsonl` (gitignored, 7-day retention, survives a restart). Read back at `GET /trace` (all) or `GET /trace/{n}` (one). |
| `serve.sh` | Activates the `ragtest` env and launches uvicorn (mirrors `scripts/*.sh`). |
| `serve_ngrok.sh` | Same as `serve.sh`, but also exposes the API over an ngrok tunnel and prints the public https URL. Optional — needs `ngrok` with an authtoken configured. |
| `requirements.txt` | `fastapi`, `uvicorn`, `pydantic`, `starlette` — install into the `ragtest` env. |

## Run

```bash
pip install -r service/requirements.txt   # once, into the ragtest conda env
ollama serve                              # generation needs a running Ollama
bash service/serve.sh                      # -> http://0.0.0.0:8000
```

(`make serve [HOST=.. PORT=..]` and `make serve-ngrok` are aliases for `serve.sh` /
`serve_ngrok.sh`.)

`GET /health` reports `starting` → `ready`. Query with a streaming client:

```bash
curl -N -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"query": "Which contractor was awarded contract 24AJ0052?", "session_id": "s1"}'
```

> Run via `serve.sh` (or `cd service && python -m uvicorn app:app`) — the modules here
> import each other flat; `app.py` adds the repo root back onto `sys.path` for `import rag.*`.
> (The folder was renamed from `fastapi/`, which shadowed the installed `fastapi` library.)

## How the seams map to this workspace

| Seam | External API (reference) | ragtest (here) |
|---|---|---|
| 1 Construction | external service's build-from-config | `StreamingAgenticRag()` → `AgenticRag` (index + reranker + LLM, load-once) |
| 2 Streaming | live `token`/`thinking`/`final_output` | verified answer **replayed** as `token` (see below); no `thinking` |
| 4 Stages | director/search/rerank/experts | `route`, `decompose`, `subanswer`, `combine` |
| 5 Final | `contract_ids`, `sources`, `graph`, `done` | `contract_ids`, `sources`, `graph`, `done` (**matches the external API's shape** — `graph` has empty edges, no KG store) |

## Wire events

The terminal payloads are **shaped to match an external agent API** so a client
written against that API consumes this stream unchanged. See the parity notes below.

- Generic (from the transport): `meta`, `stage`, `token`, `error`.
- Terminal (from `finalize`): `contract_ids`, `sources`, `graph`, `parts`, `done`.

`stage` payloads keep this workspace's own vocabulary (the pipelines differ):
`route` `{kind, contract_ids}`, `decompose` `{strategy, parts}`,
`subanswer` `{contract_id, ok, citations, issues}`, `combine` `{ok}`.

`sources`: `[{name, page, score, chunk_id, bbox}]` — same shape as the external API plus
`bbox: [x, y, width, height]` in page-image pixels from the OCR database — image space, origin
top-left, y growing downward (NOT PDF user space, which is bottom-left in points; `null` for manifest
nodes). `name` is the source-file stem (e.g. `"24aj0052_contract_agreement"`).
`graph`: `{nodes: [{id, label, pages}], edges: []}` — nodes are the cited documents; edges are
always empty (no knowledge-graph store here).
`parts`: `[{question, contract_id, answer, ok, blocks, flags, citations, evidence}]` — one entry
per sub-question; `answer` is `""` when that part's `ok` is false (a withheld sub-answer never goes on
the wire — `blocks`/`flags` say why); `evidence` carries the same `{name, page, score, chunk_id, bbox}` shape as
`sources` but scoped to that sub-question only. Empty array for `analytical` queries (no sub-questions).
`done`: `{confidence, uncertain, tokens, cost_usd, files, compose_backend, compose_model, response_text, notice}`.

### Parity with the external API (honest stand-ins)

Fields the external API derives from state this pipeline lacks carry truthful defaults, so
the shape matches without faking numbers:

- `confidence` is **binary** — `1.0` passed the grounding check, `0.0` withheld
  (this pipeline withholds instead of scoring). `uncertain` mirrors the withhold.
- `tokens` and `cost_usd` are `0` — local Ollama, no token/cost accounting.
- `compose_backend` is `"ollama"`, `compose_model` is `RAG_GEN_MODEL`.
- `graph.edges` is always `[]` — nodes match the external API's shape, but there are no
  inter-document links to compute without a KG store.
- `notice` is an **additive** key (not in the external API's `done`): the grounding-check
  reason when `uncertain` is true, so a withhold is never silent. Clients
  that don't know the key ignore it.

## Why tokens are replayed, not streamed live

This pipeline **verifies before it displays**: `AgenticRag` withholds any answer
that fails the manifest grounding check (invented contract id, or a mis-bound
location/contractor). Streaming raw generation live would put text on the wire
that the verifier might then withhold. So `run_stream` runs the full
route → decompose → verify loop, emits progress as `stage` events, and only
replays the **verified** answer as `token` events. A withheld answer streams no
tokens — `done` carries `uncertain: true` and the grounding-check `notice`
instead. This is the honest adaptation of the streaming seam to a
withhold-before-display pipeline, not a shortcut.

Recall and answer content are unchanged from `make agent`: the adapter calls
`AgenticRag.answer()` itself — one orchestration, not a copy — observing it through
its `on_stage` hook. It adds only the stage records, the token replay, and cancel
polling (a cancel raises out of the hook).
