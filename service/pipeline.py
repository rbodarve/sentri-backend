"""Seam 2 (streaming protocol) for the sentri-backend agentic RAG pipeline.

`StreamingAgenticRag` wraps `rag.agent.AgenticRag` in the blocking-generator
contract the transport expects: `run_stream(query, tracer, cancel_check)`
yielding `("thinking"|"token"|"final_output", payload)` and calling
`tracer.record(stage, data)` as each stage completes (Seam 3).

There is ONE orchestration: run_stream calls `AgenticRag.answer()` itself --
the same code path as `make agent` -- so the streamed answer cannot drift from
the CLI. Streaming is an aspect layered on through answer()'s `on_stage` hook:
each completed stage is recorded to the tracer (Seam 3), and a client cancel
aborts the run by raising out of that hook. Citations come from each Part's
retrieval `response`.

This pipeline VERIFIES BEFORE IT DISPLAYS. `AgenticRag` withholds any answer
that fails the manifest grounding check. Streaming raw LLM tokens as they
generate would put un-verified (possibly withheld) text on the wire, so we do
NOT stream generation live. We run the full route -> decompose -> verify loop,
emitting per-stage progress, and only replay the *verified* answer as `token`
events. A withheld answer streams no tokens — `done` carries the notice instead
(a fan-out whose failed parts were withheld on their own still streams its
verified parts).

The only service-only paths are the sentinels from `passthrough_query`: the two
anaphora sentinels (no prior context / reformat the previous answer) precede the
agent entirely, and the subject pin runs the agent on the pinned question and
only prefixes its verified answer with rag.subject.pin_note.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from llama_index.core import Settings

from rag.agent import AgenticRag, Part
from rag.enrich import CONTRACT_ID_RE
from rag.generate import format_sources
from rag.subject import pin_note
from rag.verify import Report

# Seam 4 stage vocabulary — the stage names this pipeline reports.
STAGE_ROUTE = "route"

# Returned by passthrough_query when an anaphoric follow-up has no session history.
_NO_CONTEXT_SENTINEL = "__NO_PRIOR_CONTEXT__"

# Returned by passthrough_query when an anaphoric follow-up has history to draw on.
# Payload format: _TRANSFORM_SENTINEL + previous_answer + _TRANSFORM_SEP + follow_up
# run_stream handles this as a direct LLM transformation (no retrieval) so that
# retrieved document chunks can't displace the previous answer as the LLM's context.
_TRANSFORM_SENTINEL = "__PRIOR_CONTEXT_TRANSFORM__"
_TRANSFORM_SEP = "\n__FOLLOWUP__\n"

# Returned by passthrough_query when the session's subject was pinned onto a follow-up.
# Payload format: _SUBJECT_SENTINEL + comma-joined pinned ids + "\n" + pinned question.
# run_stream strips it, answers the pinned question, and prefixes the verified answer with
# rag.subject.pin_note so a wrongly carried subject is visible on the same turn.
_SUBJECT_SENTINEL = "__SUBJECT_PIN__"

_TRANSFORM_TEMPLATE = (
    "Reformat or summarize the previous answer as the user requests.\n"
    "Use ONLY information from the previous answer — add no new facts.\n"
    "If the requested information is not in the previous answer, say so explicitly.\n"
    "\nPrevious answer:\n{previous}\n"
    "\nUser request: {request}\n"
    "\nResponse:"
)
STAGE_DECOMPOSE = "decompose"
STAGE_SUBANSWER = "subanswer"
STAGE_COMBINE = "combine"


@dataclass
class StreamPart:
    """One sub-question, its verified outcome, and the citations behind it."""

    question: str
    answer: str
    report: Report
    contract_id: str | None
    sources: str  # "contract/doc_type pN, ..." from rag.generate.format_sources
    # Per-chunk evidence in the external API's source-panel shape ({name, page,
    # score, chunk_id} + bbox), extracted from the retrieval Response so `finalize` can
    # emit the same `sources` payload the external API does.
    evidence: list[dict] = field(default_factory=list)


def _node_evidence(response: Any, docstore=None) -> list[dict]:
    """Project a retrieval Response's source nodes into the external API's
    per-chunk source shape. Manifest nodes (the injected contract list) are skipped —
    they're context scaffolding, not citable evidence.

    When a signature_summary node is encountered and a docstore is provided, the
    summary is expanded to its constituent signature chunks (stored in metadata as
    source_node_ids at build time) so the client receives real bbox coordinates
    instead of the summary's null coordinate."""
    if response is None:
        return []
    out: list[dict] = []
    for n in response.source_nodes:
        md = n.metadata
        if md.get("is_manifest"):
            continue
        if md.get("category") == "signature_summary" and docstore is not None:
            for node_id in md.get("source_node_ids", []):
                sig_node = docstore.get_node(node_id, raise_error=False)
                if sig_node is None:
                    continue
                sig_md = sig_node.metadata
                sig_page = sig_md.get("pdf_page")
                sig_page = int(sig_page) if isinstance(sig_page, str) and sig_page.isdigit() else sig_page
                out.append({
                    "name": Path(sig_md.get("pdf_source", "")).stem,
                    "page": sig_page,
                    "score": float(n.score) if n.score is not None else None,
                    "chunk_id": node_id,
                    "bbox": sig_md.get("coordinate"),
                })
            continue
        # pdf_page is stored as a string in the DB; emit an int to match the
        # external API's source-panel shape (page: int). Leave non-numeric values untouched.
        page = md.get("pdf_page")
        page = int(page) if isinstance(page, str) and page.isdigit() else page
        out.append({
            # Match the external API's `name`: its doc_id is the source-file stem
            # (e.g. "24aj0052_contract_agreement"), which this workspace carries as
            # each chunk's pdf_source. Emit that stem rather than a synthetic
            # "<CONTRACT_ID>/<DOC_TYPE>" pair, so a client keyed on the external
            # API's filename-style name binds unchanged.
            "name": Path(md.get("pdf_source", "")).stem,
            "page": page,
            # The reranker returns a numpy float32; coerce to a builtin float so
            # the SSE layer's json.dumps can serialize it.
            "score": float(n.score) if n.score is not None else None,
            "chunk_id": n.node.node_id,
            # [x, y, width, height] in page-image pixels (origin top-left, y down -- see
            # rag/relationships.py _reading_order_key), straight from the OCR database.
            # None for synthetic nodes with no page position (manifest nodes are skipped above).
            "bbox": md.get("coordinate"),
        })
    return out


@dataclass
class StreamFinal:
    """The terminal object handed to `finalize` (Seam 5)."""

    question: str
    kind: str                    # an agent route (simple|fanout|semantic|analytical|enumerate|rank|aggregate|filter),
                                 # or "transform" | "no_context" for the anaphora sentinels
    contract_ids: list[str]      # contract ids the user typed (resolved/pinned ids are not echoed)
    text: str                    # combined answer (displayable only if report.ok)
    report: Report               # final grounding outcome
    parts: list[StreamPart] = field(default_factory=list)


def _word_tokens(text: str) -> Iterator[str]:
    """Split into whitespace-preserving chunks so the client can re-join them
    back into the exact answer text. Used to replay the verified answer."""
    return iter(re.findall(r"\S+\s*", text))


class _Cancelled(Exception):
    """Raised from the on_stage hook to abort AgenticRag.answer() on client cancel."""


class StreamingAgenticRag:
    """Streaming adapter over AgenticRag. Built once at startup (Seam 1)."""

    def __init__(self) -> None:
        # Blocking, load-once: index + reranker + LLM handle (sets Settings.llm).
        # The transport already runs build_pipeline in a threadpool — do NOT wrap.
        self.agent = AgenticRag()
        self._docstore = self.agent.rag.docstore

    def _stream_part(self, p: Part) -> StreamPart:
        """Project an agent Part into its cited StreamPart (citations from its own Response)."""
        return StreamPart(p.question, p.answer, p.report, p.contract_id,
                          format_sources(p.response) if p.response is not None else "",
                          _node_evidence(p.response, self._docstore))

    def run_stream(
        self,
        query: str,
        tracer,
        cancel_check: Callable[[], bool],
    ) -> Iterator[tuple[str, Any]]:
        if query.startswith(_NO_CONTEXT_SENTINEL):
            # Anaphoric query with no session history — ask the user to rephrase.
            original = query[len(_NO_CONTEXT_SENTINEL) + 2:]
            tracer.record(STAGE_ROUTE, {"kind": "no_context", "contract_ids": []})
            result = StreamFinal(
                original, "no_context", [],
                "I need more context — please re-state your full question.",
                Report(blocks=[], flags=[]), [],
            )
        elif query.startswith(_TRANSFORM_SENTINEL):
            # Anaphoric follow-up with resolved history (e.g. "make that a table").
            # Run a direct LLM transformation on the previous answer — no retrieval —
            # so retrieved document chunks can't displace the prior answer as context.
            rest = query[len(_TRANSFORM_SENTINEL) + 1:]  # strip sentinel + newline
            sep_idx = rest.index(_TRANSFORM_SEP)
            previous = rest[:sep_idx]
            follow_up = rest[sep_idx + len(_TRANSFORM_SEP):]
            tracer.record(STAGE_ROUTE, {"kind": "transform", "contract_ids": []})
            text = str(Settings.llm.complete(
                _TRANSFORM_TEMPLATE.format(previous=previous, request=follow_up)
            )).strip()
            # check_bindings=False: this is a reformatting of an already-verified
            # answer, not a fresh retrieval, so location mis-binding flags don't apply.
            report = self.agent.rag.verifier.check(text, check_bindings=False)
            result = StreamFinal(follow_up, "transform", [], text, report, [])
        else:
            pinned: list[str] = []
            if query.startswith(_SUBJECT_SENTINEL):
                head, _, query = query.partition("\n")
                pinned = head[len(_SUBJECT_SENTINEL):].split(",")
            # Only emit contract IDs that the user explicitly typed in the query.
            # Location-resolved IDs (inferred from place names like "La Union") are used
            # internally for routing but must not appear in the output stream — the client
            # didn't ask by ID and shouldn't receive one it never mentioned. A pinned
            # subject id is announced by the pin note in the answer text instead.
            explicit_ids = [i for i in dict.fromkeys(
                m.upper() for m in CONTRACT_ID_RE.findall(query)
            ) if i not in pinned]

            def on_stage(stage: str, data: dict) -> None:
                # Record each completed agent stage (Seam 3), then poll for cancel so a
                # disconnected client stops the run before the next sub-question.
                if stage == STAGE_ROUTE:
                    data = {"kind": data["kind"], "contract_ids": explicit_ids}
                elif stage == STAGE_SUBANSWER:
                    part = data["part"]
                    data = {
                        "question": part.question, "route": part.route,
                        "contract_id": part.contract_id, "ok": part.report.ok,
                        "blocks": part.report.blocks, "flags": part.report.flags,
                        "sources": self._stream_part(part).sources,
                    }
                tracer.record(stage, data)
                if cancel_check():
                    raise _Cancelled

            try:
                ar = self.agent.answer(query, on_stage=on_stage)
            except _Cancelled:
                return
            text = f"{pin_note(pinned)}\n\n{ar.text}" if pinned and ar.ok else ar.text
            result = StreamFinal(query, ar.kind, explicit_ids, text, ar.report,
                                 [self._stream_part(p) for p in ar.parts])

        # Replay the verified answer as tokens. A withheld answer (failed the
        # grounding check) streams nothing — the withheld notice rides on `done`.
        if result.report.ok:
            for token in _word_tokens(result.text):
                if cancel_check():
                    return
                yield ("token", token)

        yield ("final_output", result)
