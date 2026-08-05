# Sentri Agentic API

Thin FastAPI wrapper around the Sentri agentic pipeline. The Spring Boot
backend owns sessions, history persistence, and map data; this server's
only job is to run a query through the pipeline and stream the answer back.

## Run

From the **repo root** (so the `rag_module` package is importable):

```bash
pip install -r sentri_api/requirements.txt
python -m uvicorn sentri_api.server:app --host 0.0.0.0 --port 8000
```

First boot takes ~30 s — that's the BGE embedder, ms-marco reranker, and
ChromaDB / knowledge graph being loaded once. Subsequent requests share
those models in memory and skip the cold-start.

Health probe: `GET /health` returns `{"status": "ready"}` once the
pipeline is initialized.

## Endpoint

### `POST /query`

Request body:

```json
{
  "query": "Who is the awarded contractor of contract ID 24CC0265?",
  "session_id": "9f01ed7c-b26d-4d14-a847-457176514d89",
  "history": [
    {"role": "user", "content": "List one project in Bulacan"},
    {"role": "assistant", "content": "Construction of … along Tabang River, Tibig."}
  ],
  "metadata": {"trace_id": "abc-123"}
}
```

- `query` — current question (required).
- `session_id` — opaque correlation ID for logs (required).
- `history` — prior turns in chronological order; consecutive
  user→assistant pairs become `Turn N — Q/A` blocks in the augmented
  prompt the Director sees.
- `metadata` — optional pass-through for tracing.

### Response — Server-Sent Events

`Content-Type: text/event-stream`. Frames are emitted in this order:

| Event | When | Payload |
|---|---|---|
| `meta` | immediately on accept | `{session_id, received_at}` |
| `stage` | many, one per pipeline stage as it completes | `{stage, ...summary}` |
| `thinking` | many, live `<think>…</think>` content from reasoning models | `{text: "..."}` |
| `token` | many, live as the LLM generates the answer | `{text: "..."}` |
| `contract_ids` | once, after the model finishes | `{contract_ids: [str, ...]}` |
| `sources` | once, after the model finishes | `{sources: [{name, page, score, chunk_id}]}` |
| `done` | once, terminal | `{confidence, uncertain, tokens, cost_usd, files, compose_backend, compose_model, response_text}` |
| `error` | only on failure, terminal | `{message: "..."}` |

`stage` frames (`director`, `search`, `rerank`, `expert_dispatch`,
`expert_result`, `expert_retry`, `compose`, `direct_answer`) let the UI
show pipeline progress before any answer text exists. `thinking` and
`token` then stream live. `contract_ids` and `sources` arrive **after**
the model finishes (the pipeline only knows the answer's subjects and
evidence once Compose completes), so the UI binds its sidebars (Location
viewer, Source panel) at that point — after the answer has streamed.

#### Streaming model

This is **true token-level streaming**, not answer-slicing. The pipeline
exposes `SentriPipeline.run_stream()`, a blocking generator that yields
`("token", text)` / `("thinking", text)` tuples as the composer's LLM
emits them, then a terminal `("final_output", FinalOutput)`. The server
drives it on a worker thread and bridges those tuples onto the SSE stream
via an async queue.

One consequence: the `done` event carries `response_text`, the **final**
answer after all deterministic backstops have run (groundedness
rejection, BoQ rescue, ambiguous-scope refusal). That may differ from the
concatenation of the streamed `token` events when a backstop rewrote the
answer — the client decides whether to reconcile.

## What this server does NOT do

Deliberately removed from the parity with `cmd_sentri_interactive`:

- No session loading from `db/investigations.json` — Spring Boot owns it.
- No `temp_memory/query_*.json` audit files — Spring Boot persists if needed.
- No Earth Engine bootstrap — map data is fetched by Spring Boot.
- No stdout `##SOURCES##` / `##LOCATIONS##` markers — replaced by SSE events.
- No `print()` interaction loop — each HTTP request is one query.

## Example: consuming the stream

Bash one-liner for smoke testing:

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who signed contract 24CC0265?", "session_id": "test", "history": []}'
```

The `-N` flag disables curl's output buffering so you see SSE frames
arrive incrementally.
