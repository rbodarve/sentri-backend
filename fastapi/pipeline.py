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
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from llama_index.core import Settings

# `_COMBINE_TEMPLATE` is module-private in rag.agent, but reusing it verbatim is
# what keeps the semantic-combine answer identical to the CLI agent — worth the
# private import rather than duplicating (and drifting from) the prompt.
from rag.agent import AgenticRag, _ANALYTICAL_TEMPLATE, _COMBINE_TEMPLATE
from rag.enrich import CONTRACT_ID_RE
from rag.generate import format_sources
from rag.manifest import load_manifest
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
    # score, chunk_id}), extracted from the retrieval Response so `finalize` can
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
            # [x, y, width, height] in PDF-space pixels, straight from the OCR database.
            # None for manifest-injected nodes (they have no coordinate).
            "bbox": md.get("coordinate"),
        })
    return out


@dataclass
class StreamFinal:
    """The terminal object handed to `finalize` (Seam 5)."""

    question: str
    kind: str                    # "simple" | "fanout" | "semantic" | "analytical"
    contract_ids: list[str]      # contracts the router bound the question to
    text: str                    # combined answer (displayable only if report.ok)
    report: Report               # final grounding outcome
    parts: list[StreamPart] = field(default_factory=list)


def _humanize_ids(text: str, id_to_name: dict[str, str]) -> str:
    """Swap each contract id in the DISPLAYED answer for its manifest short_name so the token
    stream / response_text read naturally (e.g. '24cc0265_roa.pdf' -> '<name>_roa.pdf'). Matches
    both cases (the regex char class covers them); an id with no short_name is left as-is.
    Presentation only -- the `contract_ids`/`sources` metadata stays keyed by id, and this runs
    after verification, so grounding is unaffected."""
    return CONTRACT_ID_RE.sub(lambda m: id_to_name.get(m.group(0).upper(), m.group(0)), text)


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
        # contract_id -> short_name, for humanizing ids in the displayed answer (see run_stream).
        self._id_to_name = {r["contract_id"].upper(): r.get("short_name") or r["contract_id"]
                            for r in load_manifest()}

        # Adapter: capture the Response of each answer_once call so we can emit
        # citations (the agent itself keeps only text + report). Shimming the
        # bound attribute on our own instance leaves rag/ untouched; the accepted
        # attempt is always the last call before the retry ladder breaks.
        #
        # This instance is a startup singleton the transport drives from a
        # threadpool, so concurrent /query requests share it. The captured
        # Response therefore lives in thread-local storage -- otherwise request A
        # could read the chunks request B just wrote and cite the wrong sources.
        # The whole route -> answer_once -> read runs in one worker thread, so a
        # thread-local slot isolates each request cleanly.
        self._docstore = self._rag._index.storage_context.docstore
        self._captured = threading.local()
        _original = self._rag.answer_once

        def _capturing_answer_once(question, contract_id, search_query,
                                   reranker=None, trace=None):
            resp = _original(question, contract_id, search_query,
                             reranker=reranker, trace=trace)
            self._captured.response = resp
            return resp

        self._rag.answer_once = _capturing_answer_once

    def _answer_part(self, subq: str, tracer) -> StreamPart:
        """Route + self-correct one sub-question (reusing the agent's ladder),
        then record the stage with its citations."""
        contract_id, search_query, route = self._rag.route(subq)
        answer, report = self.agent._answer_verified(subq, contract_id, search_query)
        resp = getattr(self._captured, "response", None)
        sources = format_sources(resp) if resp is not None else ""
        evidence = _node_evidence(resp, self._docstore)
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
            report = agent._verifier.check(text, check_bindings=False)
            result = StreamFinal(follow_up, "transform", [], text, report, [])
        else:
            kind, ids = agent.classify(query)
            # Only emit contract IDs that the user explicitly typed in the query.
            # Location-resolved IDs (inferred from place names like "La Union") are used
            # internally for routing but must not appear in the output stream — the client
            # didn't ask by ID and shouldn't receive one it never mentioned.
            explicit_ids = list(dict.fromkeys(
                m.upper() for m in CONTRACT_ID_RE.findall(query)
            ))
            tracer.record(STAGE_ROUTE, {"kind": kind, "contract_ids": explicit_ids})
            if cancel_check():
                return

            if kind == "simple":
                part = self._answer_part(query, tracer)
                result = StreamFinal(query, kind, explicit_ids, part.answer, part.report, [part])
            elif kind == "analytical":
                # Corpus-wide pattern/anomaly: one pass over the COMPLETE manifest -- no retrieval,
                # no sub-questions, so there are no per-chunk citations. Verified with the BLOCK
                # tier only (check_bindings=False) so a legitimate multi-contract synthesis isn't
                # false-withheld. Mirrors AgenticRag.answer()'s analytical branch.
                manifest_str = self._rag._manifest_node.node.text
                text = str(Settings.llm.complete(
                    _ANALYTICAL_TEMPLATE.format(manifest_str=manifest_str, query_str=query)
                )).strip()
                report = agent._verifier.check(text, check_bindings=False)
                tracer.record(STAGE_COMBINE, {"strategy": "analytical", "ok": report.ok})
                result = StreamFinal(query, kind, explicit_ids, text, report, [])
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
                    ok_parts = [p for p in parts if p.report.ok]
                    if not ok_parts:
                        # Every sub-answer failed grounding — withhold entirely.
                        report = Report(blocks=[b for p in parts for b in p.report.blocks],
                                        flags=[f for p in parts for f in p.report.flags])
                        result = StreamFinal(query, kind, explicit_ids, "", report, parts)
                    else:
                        # Synthesize from verified parts only, then re-verify the join.
                        # This prevents failed sub-answers (e.g. a hallucinated location)
                        # from poisoning the composed text and triggering false Verifier flags.
                        context = "\n\n".join(
                            f"Sub-question: {p.question}\nAnswer: {p.answer}" for p in ok_parts
                        )
                        combined = str(Settings.llm.complete(
                            _COMBINE_TEMPLATE.format(context_str=context, query_str=query)
                        )).strip()
                        report = agent._verifier.check(combined)
                        tracer.record(STAGE_COMBINE, {"ok": report.ok})
                        result = StreamFinal(query, kind, explicit_ids, combined, report, parts)
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
                    result = StreamFinal(query, kind, explicit_ids, combined, report, parts)

        # Humanize contract ids in the DISPLAYED answer only (the token stream + response_text
        # read `finalize` off this same text). The `contract_ids`/`sources` metadata stays keyed
        # by id. This runs after verification, so it can't affect grounding.
        result.text = _humanize_ids(result.text, self._id_to_name)

        # Replay the verified answer as tokens. A withheld answer (failed the
        # grounding check) streams nothing — the withheld notice rides on `done`.
        if result.report.ok:
            for token in _word_tokens(result.text):
                if cancel_check():
                    return
                yield ("token", token)

        yield ("final_output", result)
