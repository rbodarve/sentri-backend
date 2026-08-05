"""Seam 2 (streaming protocol) for the ragtest agentic RAG pipeline.

`StreamingAgenticRag` wraps `rag.agent.AgenticRag` in the blocking-generator
contract the transport expects: `run_stream(query, tracer, cancel_check)`
yielding `("thinking"|"token"|"final_output", payload)` and calling
`tracer.record(stage, data)` as each stage completes (Seam 3).

Two workspace facts shape this adapter and are why it re-drives the agent's own
helper methods instead of just calling `AgenticRag.answer()`:

  * This pipeline VERIFIES BEFORE IT DISPLAYS. `AgenticRag` withholds any answer
    that fails the manifest grounding check. Streaming raw LLM tokens as they
    generate would put un-verified (possibly withheld) text on the wire, so we
    do NOT stream generation live. We run the full route -> decompose -> verify
    loop, emitting per-stage progress, and only replay the *verified* answer as
    `token` events. A withheld answer streams no tokens — `done` carries the
    notice instead.
  * The agent keeps only `(text, report)` per sub-answer and discards the
    retrieval `Response` (with its `source_nodes`). We shim `answer_once` on our
    private `RagAnswerer` instance to capture that `Response` so `finalize` can
    project real citations. `rag/` is left untouched.

The orchestration below mirrors `AgenticRag.answer()` step for step (same
router, same fan-out/decompose, same combine prompt, same verifier), so the
answer matches `make agent` exactly; the only additions are the stage records,
the token replay, and the cancel-check polling the transport requires.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from llama_index.core import Settings

# `_COMBINE_TEMPLATE` is module-private in rag.agent, but reusing it verbatim is
# what keeps the semantic-combine answer identical to the CLI agent — worth the
# private import rather than duplicating (and drifting from) the prompt.
from rag.agent import AgenticRag, _COMBINE_TEMPLATE
from rag.generate import format_sources
from rag.verify import Report

# Seam 4 stage vocabulary — the stage names this pipeline reports.
STAGE_ROUTE = "route"
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
    # Per-chunk evidence in Sentri's source-panel shape ({name, page, score,
    # chunk_id}), extracted from the retrieval Response so `finalize` can emit
    # the same `sources` payload Sentri does.
    evidence: list[dict] = field(default_factory=list)


def _node_evidence(response: Any) -> list[dict]:
    """Project a retrieval Response's source nodes into Sentri's per-chunk
    source shape. Manifest nodes (the injected contract list) are skipped —
    they're context scaffolding, not citable evidence."""
    if response is None:
        return []
    out: list[dict] = []
    for n in response.source_nodes:
        md = n.metadata
        if md.get("is_manifest"):
            continue
        # pdf_page is stored as a string in the DB; emit an int to match Sentri's
        # source-panel shape (page: int). Leave non-numeric values untouched.
        page = md.get("pdf_page")
        page = int(page) if isinstance(page, str) and page.isdigit() else page
        out.append({
            "name": f"{md.get('contract_id')}/{md.get('doc_type')}",
            "page": page,
            # The reranker returns a numpy float32; coerce to a builtin float so
            # the SSE layer's json.dumps can serialize it.
            "score": float(n.score) if n.score is not None else None,
            "chunk_id": n.node.node_id,
        })
    return out


@dataclass
class StreamFinal:
    """The terminal object handed to `finalize` (Seam 5)."""

    question: str
    kind: str                    # "simple" | "fanout" | "semantic"
    contract_ids: list[str]      # contracts the router bound the question to
    text: str                    # combined answer (displayable only if report.ok)
    report: Report               # final grounding outcome
    parts: list[StreamPart] = field(default_factory=list)


def _word_tokens(text: str) -> Iterator[str]:
    """Split into whitespace-preserving chunks so the client can re-join them
    back into the exact answer text. Used to replay the verified answer."""
    return iter(re.findall(r"\S+\s*", text))


class StreamingAgenticRag:
    """Streaming adapter over AgenticRag. Built once at startup (Seam 1)."""

    def __init__(self) -> None:
        # Blocking, load-once: index + reranker + LLM handle (sets Settings.llm).
        # The transport already runs build_pipeline in a threadpool — do NOT wrap.
        self.agent = AgenticRag()
        self._rag = self.agent._rag  # the shared RagAnswerer the agent answers through

        # Adapter: capture the Response of each answer_once call so we can emit
        # citations (the agent itself keeps only text + report). Shimming the
        # bound attribute on our own instance leaves rag/ untouched; the accepted
        # attempt is always the last call before the retry ladder breaks.
        self._last_response: Any = None
        _original = self._rag.answer_once

        def _capturing_answer_once(question, contract_id, search_query,
                                   reranker=None, trace=None):
            resp = _original(question, contract_id, search_query,
                             reranker=reranker, trace=trace)
            self._last_response = resp
            return resp

        self._rag.answer_once = _capturing_answer_once

    def _answer_part(self, subq: str, tracer) -> StreamPart:
        """Route + self-correct one sub-question (reusing the agent's ladder),
        then record the stage with its citations."""
        contract_id, search_query, route = self._rag.route(subq)
        answer, report = self.agent._answer_verified(subq, contract_id, search_query)
        resp = self._last_response
        sources = format_sources(resp) if resp is not None else ""
        evidence = _node_evidence(resp)
        tracer.record(STAGE_SUBANSWER, {
            "question": subq, "route": route, "contract_id": contract_id,
            "ok": report.ok, "blocks": report.blocks, "flags": report.flags,
            "sources": sources,
        })
        return StreamPart(subq, answer, report, contract_id, sources, evidence)

    def run_stream(
        self,
        query: str,
        tracer,
        cancel_check: Callable[[], bool],
    ) -> Iterator[tuple[str, Any]]:
        agent = self.agent

        kind, ids = agent.classify(query)
        tracer.record(STAGE_ROUTE, {"kind": kind, "contract_ids": ids})
        if cancel_check():
            return

        if kind == "simple":
            part = self._answer_part(query, tracer)
            result = StreamFinal(query, kind, ids, part.answer, part.report, [part])
        else:
            subqs = (agent._fanout_subquestions(query, ids) if kind == "fanout"
                     else agent._decompose(query))
            tracer.record(STAGE_DECOMPOSE, {"strategy": kind, "subquestions": subqs})

            parts: list[StreamPart] = []
            for subq in subqs:
                if cancel_check():
                    return
                parts.append(self._answer_part(subq, tracer))

            if kind == "fanout":
                # Per-contract listing: combined report is ok only if every part cleared.
                text = "\n".join(f"- {p.answer}" for p in parts)
                report = Report(blocks=[b for p in parts for b in p.report.blocks],
                                flags=[f for p in parts for f in p.report.flags])
                result = StreamFinal(query, kind, ids, text, report, parts)
            else:
                # semantic: combine the (already verified) sub-answers, then verify the join.
                context = "\n\n".join(
                    f"Sub-question: {p.question}\nAnswer: {p.answer}" for p in parts
                )
                combined = str(Settings.llm.complete(
                    _COMBINE_TEMPLATE.format(context_str=context, query_str=query)
                )).strip()
                report = agent._verifier.check(combined)
                tracer.record(STAGE_COMBINE, {"ok": report.ok})
                result = StreamFinal(query, kind, ids, combined, report, parts)

        # Replay the verified answer as tokens. A withheld answer (failed the
        # grounding check) streams nothing — the withheld notice rides on `done`.
        if result.report.ok:
            for token in _word_tokens(result.text):
                if cancel_check():
                    return
                yield ("token", token)

        yield ("final_output", result)
