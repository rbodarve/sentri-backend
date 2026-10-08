# CLAUDE.md

RAG pipeline over a corpus of DPWH (Philippine gov) construction-bid documents. It
retrieves and generates grounded, cited answers about contracts (bidders, awards, dates,
amounts) from OCR'd source PDFs. Built on LlamaIndex; runs entirely on CPU + local Ollama.

## Working rules (this workspace)

The global `~/.claude/CLAUDE.md` is the base — its principles always apply. The rules below
are workspace-specific additions on top of it; keep such additions here, not in the global.

- **Never push to GitHub automatically.** Make changes locally, and commit only when asked.
  Leave `git push` to the user — do not push without an explicit, in-the-moment request.

## Environment (read first)

- **Everything runs in the `sentri-backend` conda env.** `python -m rag.*` fails from base — the
  `scripts/*.sh` wrappers auto-activate it (override with `RAG_CONDA_ENV`). Run commands via
  `make`/scripts, or `conda activate sentri-backend` first.
- **CPU-only, zero-VRAM by design.** torch is installed CPU-only (`scripts/setup.sh`).
- **Generation needs a running Ollama server** with the model pulled
  (`ollama pull granite4.1:3b`). Retrieval/recall eval do not need Ollama — but
  `make build` does: its final `rag.manifest` step extracts one record per contract
  via the LLM, and it runs *after* the CPU embedding, so a missing Ollama fails the
  build late (after the expensive embed).

## Commands
 
Use `make` (details in [scripts/README.md](scripts/README.md)):

- `make setup` — create the conda env + install pinned deps (one-time).
- `make check` — stage 1–3 self-checks (loader, enrich, relationships) + the model-free router
  gate (`python -m rag.evaluate_agentic --routes`; skipped until `make build` creates the manifest).
- `make coverage` — model-free field-coverage report over the built manifest (the analytical
  route's data ceiling).
- `make build` — build the retrieval eval set + embed & persist the vector index to `index_store/`,
  then extract the corpus manifest (this last step needs Ollama).
- `make eval [MODE=baseline|filtered|rerank]` — measure retrieval recall; default `rerank`.
- `make ask Q="..."` — one-shot retrieve + rerank + generate (needs Ollama).
- `make chat` — interactive query loop.
- `make agent Q="..."` — agentic controller: router + fan-out + semantic decomposition +
  verifier-driven self-correction over the single-pass RAG (needs Ollama).
- `make chat-agentic` — interactive query loop routed through the agentic controller (needs Ollama).
- `make eval-agentic` — answer-level eval of the agent (routing/fan-out/decomposition/
  withholding) against `eval/eval_agentic.json`; needs Ollama. `make eval` stays the recall gate.
- `make serve [HOST=.. PORT=..]` — serve the agent as a streaming (SSE) HTTP API via
  [service/](service/) (needs Ollama, plus `pip install -r service/requirements.txt`).
- `make serve-ngrok` — same SSE API, exposed through an ngrok public tunnel (needs Ollama +
  `ngrok` with an authtoken configured).
- `make simulate-remote [Q="..."]` — `serve-ngrok` plus a simulated third-party client
  ([service/remote_client.py](service/remote_client.py)) that queries the public URL, prints the
  streamed events and checks the stream contract; exits 1 on a bad stream.
- `make clean` — remove the regenerable `index_store/` + caches.

## Architecture

Pipeline stages, each a module in [rag/](rag/) runnable as `python -m rag.<name>`:

`loader` → `enrich` → `relationships` (stages 1–3, validated by `make check`) →
`build_retrieval_eval` + `index` + `manifest` (`make build`) → `evaluate` (recall) / `generate`,
`chat` (answers). `rerank.py` is the CPU cross-encoder stage; `config.py` is the single
config surface; `verify.py`/`trace.py` are support libs; `subject.py` carries a conversation's
contract into follow-ups (used by `chat_agentic` and the service). `loader.py` labels
context-free chunks from their own page (a notice's bare date stamp, the agreement's "made this
... day of" line, the notarial register entry, a BAC resolution's dates) — if you label a new
kind, check its unlabelled siblings: a label makes a chunk outrank them.

`agent.py` is the **agentic layer** over `generate.py` (it does not replace it): a
deterministic router classifies each question `simple | fanout | semantic | enumerate | rank |
aggregate | filter | analytical`. Fan-out splits a contiguous multi-contract question, an "each/all projects"
question, or a corpus-wide sweep ("... across the documents in the database") into one
single-contract sub-question each (by rule), and withholds only a part that fails the check.
`semantic` multi-hop questions are decomposed by the LLM (the only place the LLM drives
control); `enumerate`/`rank`/`aggregate`/`filter`/`analytical` answer from the manifest (`rank` — a corpus-wide
"highest / rank by amount", optionally among the contracts above / below X — sorts the manifest amounts in Decimal, and `aggregate` — a corpus-wide
total/average amount, optionally of the contracts above / below X — sums them in Decimal, and `filter` — a corpus-wide "amount above / below
X", or "how many contracts are above X" — compares each with the threshold in Decimal, all with no LLM; duration is not a
manifest field yet, see PLAN.md Phase B). Every sub-answer runs the
deterministic route + `Verifier` inside a self-correction ladder (widen k → withhold; a known
contract id stays filtered on every rung, an id not in the corpus is withheld before retrieval).
A contract-filtered pass sees only its own manifest row. It reuses `RagAnswerer.answer_once`,
so it never adds a retrieval path and can't move recall. The rule is that no LLM computes: prompts forbid
sums. A calc-cued one-contract question ("total / combined / difference ...") routes `calc`: the
answer cites guarded operands; when the operand set is pinned, `rag/calc.py` states the total in
Decimal (168191b), else it states no total. Sealed G10: 1 of 18 correct, 1 wrong (G04, a calc
phrasing routed `simple`); calc route 8 rows: 1 correct, 7 withheld, 0 wrong. Routing of new calc
phrasings is unmeasured. Known limit (G-total residual): a named item whose row is in no retrieved
chunk, with no stated count, can give a total that misses that item. An un-cued phrasing (e.g.
"differ") still routes `simple`, and the model can compute there (sealed E13). The numeric grounding
guard (`rag.verify.ungrounded_figures`, called by `agent._guard` on every model-written part, no retry)
withholds a figure >= 1,000 that no context node or the question holds: sealed 0 of 11 leaked
(bound < ~27%), 0 lookup controls withheld. Figures < 1,000 and amounts in words are not checked.
Its token rule is frozen (PLAN.md Phase F).
`evaluate_agentic.py` scores it at the answer level.

Data & artifacts:

- **`database/*.json` is the authoritative OCR transcription.** Chunks are UUID-keyed under
  `text`/`table`/`image`/`signature`, each with a `content` field. **Never re-OCR the
  `source/` PDFs to "verify" the database** — the DB is the trusted source of truth.
- `source/` — original bid PDFs (corpus input). `eval/eval_retrieval.json` — ground-truth Q/chunk
  pairs. `index_store/` — the built vector index; **regenerable, gitignored** (`make build`).
- [README.md](README.md) — the public overview; [verdict.txt](verdict.txt) — scalability analysis
  (what breaks at 100k documents). Keep both in step with architecture changes.
- Local-only (not in git): `docs/` (gitignored: `queries.txt`, `ANALYSIS.txt`, the data-flow deck)
  and `eval/runs/` (excluded via `.git/info/exclude`: the queries.txt stream harness
  `stream_sim.py` + judge `judge_stream.py`, per-question root causes, run files).

Models are **config-driven** in [rag/config.py](rag/config.py) via `RAG_*` env vars — swap
embedding/reranker/LLM for better hardware without touching pipeline code. Defaults:
`BAAI/bge-small-en-v1.5` embed, `BAAI/bge-reranker-base` rerank, `granite4.1:3b` gen
(temp 0.0, `num_ctx` 4096 — capped deliberately to keep the KV cache on a 4GB GPU).

## Conventions

- **"Pass" = retrieval recall**, not answer prose. The headline result is
  `make eval` in `rerank` mode → `hit_rate 1.000` (contract-filtered + CPU rerank);
  `baseline`/`filtered` are ablations. When changing retrieval, re-run `make eval` and keep
  recall at 1.000.
- **Agent eval is answer-level, not recall.** `make eval-agentic` scores the agent on
  `eval/eval_agentic.json`: an `answer` row passes only if the answer clears the `Verifier` AND
  contains every expected anchor; a `withhold` row (invented contract / mis-binding lure) passes
  only if the agent correctly withholds. It needs Ollama and is separate from the recall gate.
- **Full question set, through the service.** `eval/runs/stream_sim.py` replays all 441
  `docs/queries.txt` questions as SSE client streams against a running `service/` (PART 1 as real
  sessions) and `eval/runs/judge_stream.py` scores them: 407/441 = 92.3% (R10, 2026-10-06;
  single-run noise ~±7); 408/441 with the 2026-10-08 judge, which scores a withhold like a decline
  (PLAN.md Phase H; only #68 moves). The calc route removed R9's 5 wrong model sums and lost 4 correct ones;
  R10 predates 168191b, so it stated no total (R9: 415/441). ~4 h per run on two servers.
- **Verify gate:** `.claude/verify.sh` (a Stop hook) validates `database/` OCR integrity on
  every turn. It must exit 0. It only checks JSON structure — it never re-OCRs.
  `.claude/` is gitignored, so this gate is local-only: a fresh clone has no verify.sh or hook.
- Match existing style; keep changes surgical. `PYTHONWARNINGS=ignore` and `PYTHONUTF8=1` (Windows defaults to cp1252) are set by the scripts.
