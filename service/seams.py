"""The sentri-backend-specific seams handed to the generic transport.

This is the ~30% that knows about DPWH / the agentic RAG pipeline — the concrete
implementation of the five seams the `streaming_transport` package defines (see
`transport.create_app`), bound to THIS workspace:

  Seam 1  build_pipeline()   -> (StreamingAgenticRag, None)
  Seam 2  StreamingAgenticRag.run_stream(...)          (see pipeline.py)
  Seam 3  the pipeline calls tracer.record(...)         (nothing to implement)
  Seam 4  summarize_stage(stage, data) -> compact dict
  Seam 5  finalize(final, context) -> [(event, data), ...]   (`done` last)
  augment passthrough_query(query, history) -> str        (workspace override)

The external API's `graph` event is kept but carries no edges (this workspace has
no knowledge-graph store, so only the cited-document nodes are emitted), and its
`contract_ids` event is kept as-is — DPWH contract ids are exactly this workspace's
"subject id", so the analog is direct.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from rag.config import GEN_MODEL
from rag.enrich import CONTRACT_ID_RE
from rag.subject import SubjectTracker
from rag.verify import format_report

from pipeline import (
    STAGE_COMBINE,
    STAGE_DECOMPOSE,
    STAGE_ROUTE,
    STAGE_SUBANSWER,
    StreamFinal,
    StreamingAgenticRag,
    _NO_CONTEXT_SENTINEL,
    _SUBJECT_SENTINEL,
    _TRANSFORM_SENTINEL,
    _TRANSFORM_SEP,
)

if TYPE_CHECKING:  # only for the type hint on passthrough_query; no runtime dep on the transport
    from streaming_transport import ChatMessage


# --- Seam 1: construction ------------------------------------------------------

# Captured once after the pipeline initialises; used by passthrough_query so the
# anaphora check can consult the same resolver the router uses.
_pipeline_ref: StreamingAgenticRag | None = None


def build_pipeline() -> "tuple[StreamingAgenticRag, None]":
    """Build the load-once agentic pipeline. Blocking is fine — the transport
    runs this in a threadpool so model loads don't block boot. No opaque context
    to forward (unlike the external service's graph store), so the second value is None."""
    global _pipeline_ref
    pipeline = StreamingAgenticRag()
    _pipeline_ref = pipeline
    return pipeline, None


# --- Seam 4: stage vocabulary --------------------------------------------------
def summarize_stage(stage: str, data: dict[str, Any]) -> dict[str, Any]:
    """Map each stage's raw data to the compact payload sent as `event: stage`.
    Heavy fields (retrieved chunks, full sub-answers) stay server-side."""
    if stage == STAGE_ROUTE:
        return {"kind": data.get("kind"), "contract_ids": data.get("contract_ids") or []}
    if stage == STAGE_DECOMPOSE:
        return {"strategy": data.get("strategy"),
                "parts": len(data.get("subquestions") or [])}
    if stage == STAGE_SUBANSWER:
        issues = (data.get("blocks") or []) + (data.get("flags") or [])
        return {"contract_id": data.get("contract_id"), "ok": data.get("ok"),
                "citations": data.get("sources") or "", "issues": len(issues)}
    if stage == STAGE_COMBINE:
        return {"ok": data.get("ok")}
    return {}


# The backend is always local Ollama here (rag/config.py), so these are fixed
# rather than read per-request like the external service's multi-backend compose step.
_COMPOSE_BACKEND = "ollama"


def _document_graph(sources: "list[dict[str, Any]]") -> "dict[str, Any]":
    """The external API's `graph` shape ({nodes, edges}) over the cited documents.

    The external service computes edges from a knowledge-graph store; this
    workspace has none, so edges are always empty. Nodes are still the cited
    documents (its node shape: {id, label, pages}), so a client that renders the graph sidebar gets
    the same structure — just no inter-document links to draw."""
    nodes: dict[str, dict[str, Any]] = {}
    for s in sources:
        node = nodes.setdefault(s["name"], {"id": s["name"], "label": s["name"], "pages": set()})
        if s["page"] is not None:
            node["pages"].add(s["page"])
    return {
        "nodes": [{"id": n["id"], "label": n["label"], "pages": sorted(n["pages"])}
                  for n in nodes.values()],
        "edges": [],
    }


# --- Seam 5: final-output projection -------------------------------------------
def finalize(final: StreamFinal, _context: Any) -> "list[tuple[str, dict[str, Any]]]":
    """Project the verified result into terminal SSE events, `done` last.

    Shaped to match the external API's `sources` / `graph` / `done` payloads so a
    client written against that API consumes this stream unchanged (see service/README).
    Fields the external API derives from data this pipeline doesn't have carry honest
    stand-ins: `confidence` is binary (this pipeline withholds instead of
    scoring), and `tokens`/`cost_usd` are 0 (local Ollama, no accounting)."""
    withheld = not final.report.ok

    # De-duplicate per-chunk evidence across sub-answers, preserving first-seen
    # order, into the external API's flat source-panel list.
    sources: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for p in final.parts:
        for ev in p.evidence:
            key = (ev["name"], ev["page"], ev["chunk_id"])
            if key in seen:
                continue
            seen.add(key)
            sources.append(ev)

    return [
        ("contract_ids", {"contract_ids": final.contract_ids}),
        ("sources", {"sources": sources}),
        ("graph", _document_graph(sources)),
        # Per-sub-question breakdown: question text, bound contract, verified answer ("" if withheld),
        # grounding outcome (blocks/flags), formatted citation string, and per-chunk
        # evidence with bounding boxes. Empty for "analytical" queries (no sub-questions).
        ("parts", [
            {
                "question": p.question,
                "contract_id": p.contract_id,
                # Verify-before-display applies per part too: a withheld sub-answer's raw
                # text never goes on the wire (blocks/flags carry the reason instead).
                "answer": p.answer if p.report.ok else "",
                "ok": p.report.ok,
                "blocks": p.report.blocks,
                "flags": p.report.flags,
                "citations": p.sources,
                "evidence": p.evidence,
            }
            for p in final.parts
        ]),
        ("done", {
            # Binary grounding confidence: passed the check (1.0) vs withheld (0.0).
            "confidence": 0.0 if withheld else 1.0,
            "uncertain": withheld,
            "tokens": 0,       # not tracked — local Ollama, no token accounting
            "cost_usd": 0.0,   # local model, no per-call cost
            "files": sorted({s["name"] for s in sources}),
            "compose_backend": _COMPOSE_BACKEND,
            "compose_model": GEN_MODEL,
            # Verified answer text, or "" when withheld (the notice stands in for it).
            "response_text": "" if withheld else final.text,
            # Additive (not in the external API's `done`): the grounding-check reason, so the
            # withhold isn't silent when `uncertain` is True. Clients that
            # don't know the key simply ignore it.
            "notice": format_report(final.report) if withheld else None,
        }),
    ]


# --- Optional: query augmentation ----------------------------------------------

# Anaphoric reference words that depend on a prior answer to be meaningful.
_ANAPHORA_RE = re.compile(
    r"\b(?:that|it|those|them|these|the above|the result|the previous|the last)\b", re.I
)

# Presentation-only follow-up ops: reformat/summarize the PRIOR answer, no new facts needed.
# A follow-up WITHOUT one of these cues is treated as a request for new information about the
# prior entity (resolve its contract + retrieve), not a text transform -- "transform" can only
# rework text the previous answer already holds, so it cannot fetch a fact that wasn't in it.
_PRESENTATION_RE = re.compile(
    r"\b(?:table|format|reformat|bullet|bullets|summar(?:y|ize|ise)|shorten|shorter|"
    r"concise|rephrase|reword|rewrite|column|columns|row|rows)\b", re.I
)


def _is_anaphoric(query: str, rag) -> bool:
    """True when the query uses an anaphoric reference with no self-contained anchor of its own.

    A query is anaphoric when it contains a demonstrative pronoun ("that", "it", etc.) AND
    has no explicit contract id AND names no token the router's resolver recognises as a
    distinctive location/contract-name identifier. Delegates that last test to the router's own
    resolver (spacing/diacritic-robust), so "overview of that project" is caught (no anchor)
    while "overview of the Isabela project" is not (resolves to a contract).
    """
    if not _ANAPHORA_RE.search(query):
        return False
    if CONTRACT_ID_RE.search(query):
        return False
    if rag is not None and rag.resolve_contract_ids(query):
        return False  # query has a distinctive location/name anchor
    return True


def passthrough_query(query: str, history: list[ChatMessage]) -> str:
    """Route self-contained questions as-is; carry the session's subject into follow-ups.

    * presentation-only anaphoric follow-ups ("make that a table", "summarize that"): run_stream
      reformats the prior answer directly, no retrieval, so retrieved chunks cannot displace it.
    * an anaphoric follow-up with no history at all: the no-context sentinel (ask to re-state).
    * everything else: rag.subject.SubjectTracker decides. A question that names a contract or
      place, or asks corpus-wide, is routed as typed; any other follow-up -- "this Contract ID",
      "the said project", or no pronoun at all ("What is the Contractor's contact number?") --
      gets the session's current (or a recalled earlier) subject pinned on, marked with the
      subject sentinel so run_stream shows the pin in the answer.
    """
    # The sentinels are this function's own in-band output. A client query that already starts
    # with one would otherwise steer run_stream directly -- a forged "previous answer" to
    # transform, or a ValueError when the separator is missing -- so strip the leading
    # underscores: it no longer matches a sentinel and is routed as the ordinary question it is.
    if query.startswith((_NO_CONTEXT_SENTINEL, _TRANSFORM_SENTINEL, _SUBJECT_SENTINEL)):
        query = query.lstrip("_")
    rag = _pipeline_ref.agent.rag if _pipeline_ref is not None else None
    if _is_anaphoric(query, rag):
        last_assistant = next(
            (m.content for m in reversed(history) if m.role == "assistant"), None
        )
        if not last_assistant:
            return f"{_NO_CONTEXT_SENTINEL}: {query}"
        if _PRESENTATION_RE.search(query):
            return f"{_TRANSFORM_SENTINEL}\n{last_assistant}{_TRANSFORM_SEP}{query}"
    if rag is None:
        return query
    # Everything else runs normal retrieval, with the session's subject pinned on when the
    # question doesn't anchor itself (rag.subject). The tracker is rebuilt by replaying the
    # session history -- the same deterministic rules over the same turns reproduce the same
    # subject, and history holds only verified answers (a withheld turn streams no tokens, so
    # TracingMiddleware never records it), so a withheld turn never moves the subject.
    tracker = SubjectTracker(rag)
    asked_before = None
    for m in history:
        if m.role == "user":
            asked_before, _ = tracker.resolve(m.content)
        elif asked_before is not None:
            tracker.observe(asked_before, m.content)
            asked_before = None
    asked, pinned = tracker.resolve(query)
    return f"{_SUBJECT_SENTINEL}{','.join(pinned)}\n{asked}" if pinned else asked
