"""Agentic controller over the single-pass RAG (rag.generate.RagAnswerer).

Composes three behaviors on top of the deterministic pipeline, without replacing it:

  router       -- classify a question: "simple" | "fanout" | "semantic" | "analytical" |
                  "enumerate" (a corpus-wide "list all projects/locations" answered straight
                  from the manifest, deterministically -- no LLM, no retrieval).
  fan-out      -- a multi-contract question asking one intent of several contracts -- an
                  explicit id list, or a distributive "each/all projects" over the whole corpus
                  -- is split, by rule, into one single-contract sub-question each (no LLM).
  semantic     -- a genuine multi-hop/compound question is split into sub-questions by the LLM
                  (the ONLY place the LLM drives control), then the sub-answers are combined.
  analytical   -- a corpus-wide pattern/anomaly/commonality question reasons over the COMPLETE
                  manifest in one pass (not per-contract, no retrieval added -- the manifest is
                  already the whole corpus), the substrate top-k retrieval structurally lacks.
  self-correct -- every sub-answer is checked by the manifest Verifier; on a block/flag (or an
                  unhelpful "I don't know") it retries with a widened k, then a dropped contract
                  filter, and otherwise the answer is withheld.

Design constraints this module honors (see CLAUDE.md / the plan):
  * Recall stays 1.000: it reuses RagAnswerer.answer_once unchanged; widening only keeps more
    already-retrieved chunks. It never adds a new retrieval path.
  * No local arithmetic: the combine prompt forbids sums/totals -- aggregation stays the job of
    the stronger handoff model, exactly as the manifest/verifier already assume.
  * A bad LLM split cannot emit an unverified claim: every sub-answer passes the deterministic
    route + Verifier before it is combined.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate

from rag.config import AGENT_MAX_ATTEMPTS, AGENT_WIDE_TOP_N
from rag.enrich import CONTRACT_ID_RE
from rag.generate import RagAnswerer
from rag.manifest import format_enumeration
from rag.rerank import get_reranker
from rag.verify import Report

_ID = CONTRACT_ID_RE.pattern
# A contiguous list of >=2 contract ids ("contracts X, Y and Z" / "for X and Y"): the signal
# that one intent is being asked of several contracts -> deterministic fan-out. Ids split across
# separate clauses (different verbs) never match this, so they fall through to semantic.
_MULTI_ID_RE = re.compile(
    rf"(?:(?:of|for)\s+)?(?:contracts?\s+)?((?:{_ID})(?:\s*(?:,|and)\s*{_ID})+)", re.I
)
# Comparison/superlative cues make even a contiguous id list semantic (needs cross-contract
# reasoning), not a plain per-contract fan-out.
_COMPARISON_RE = re.compile(
    r"\b(larger|largest|bigger|biggest|smaller|smallest|highest|lowest|greatest|most|least|"
    r"more|less|than|compare|difference|which of)\b", re.I
)
# A distributive "each/every/all/per project(s)/contract(s)" phrase names no ids but asks one
# intent of the whole corpus ("list each project's location"). It fans out over every known
# contract; a leading "for"/"of" is left outside the match so the rewrite reads naturally.
_EACH_RE = re.compile(r"\b(?:each|every|all|per)\s+(?:of\s+the\s+)?(?:projects?|contracts?)\b", re.I)
# Corpus-wide sweep cue ("... recorded across the documents in the database"): one field asked of
# every contract. A single unfiltered pass hands the model 10 of ~400 chunks, so it can only ever
# list the few instances that ranked; it fans out over every contract instead, like _EACH_RE.
_CORPUS_RE = re.compile(
    r"\bacross\b.{0,40}\b(?:documents|database|corpus|contracts|projects|files)\b|"
    r"\b(?:in|from|of) the (?:database|corpus)\b", re.I
)
# ...but "list ONE project in the database" asks for a single example, not a sweep.
_SINGULAR_RE = re.compile(r"\b(?:one|a|an|single)\s+(?:\w+\s+){0,2}(?:project|contract)\b", re.I)
_ACROSS_RE = re.compile(r"\bacross\s+(?:all\s+)?(?:the\s+)?", re.I)
_IN_DB_RE = re.compile(r"\s*\b(?:found\s+)?(?:in|from|of) the (?:database|corpus)\b", re.I)
# Corpus-wide enumeration cue: a listing verb over a corpus-level noun. Combined with "no explicit
# id AND no resolvable location" in classify(), this catches "list all projects" / "list the
# locations you know about" -- questions whose complete answer IS the whole manifest, so they are
# answered by deterministic enumeration instead of lossy summarization or a whole-corpus fan-out.
_LIST_CORPUS_RE = re.compile(
    r"\b(?:list|show|give|name|enumerate|tell)\b.{0,40}?"
    r"\b(?:projects|contracts|locations|places)\b", re.I  # plural: a corpus-wide set, not "that project"
)
# Cross-corpus analytical cues: a pattern/anomaly/commonality question that must reason over the
# whole corpus at once, not fetch one contract's spans. Routed to the manifest-wide analytical
# pass (see answer()). A question that instead compares >=2 *named* contracts stays semantic.
_ANALYTICAL_RE = re.compile(
    r"\b(?:patterns?|anomal(?:y|ies|ous)|outliers?|trends?|unusual|irregular(?:ities)?|"
    r"recurring|commonalit(?:y|ies)|similarit(?:y|ies)|stands?\s+out)\b|\bin\s+common\b", re.I
)

_DECOMPOSE_TEMPLATE = PromptTemplate(
    "Break the question into the minimal list of independent, self-contained sub-questions "
    "needed to answer it. Each sub-question must name its own contract id or location so it "
    "stands alone. Do NOT ask for sums or totals. Return ONLY a JSON array of strings.\n"
    "Question: {query_str}\nJSON: "
)
_COMBINE_TEMPLATE = PromptTemplate(
    "You are answering a multi-part question about DPWH procurement documents.\n"
    "Below are answers to its sub-questions. Compose ONE concise, direct answer to the original "
    "question using ONLY these sub-answers -- add no new facts, and do NOT compute sums or "
    "totals. State each fact in its own sentence that names the contract id it belongs to, so "
    "no fact reads as belonging to another contract. Name the source document(s).\n"
    "---------------------\n{context_str}\n---------------------\n"
    "Original question: {query_str}\nAnswer: "
)
# Analytical answers reason over the COMPLETE manifest (the whole corpus in one context), the one
# place a cross-corpus pattern/anomaly is even visible. Grounded strictly in the manifest rows, no
# sums/totals (aggregation stays the handoff model's job), honest when the manifest can't support it.
_ANALYTICAL_TEMPLATE = PromptTemplate(
    "You analyze a COMPLETE set of DPWH infrastructure-procurement contracts.\n"
    "The CORPUS MANIFEST below is the authoritative, complete list of every contract and its "
    "fields. Answer by reasoning ONLY over these rows -- identify patterns, commonalities, or "
    "anomalies ACROSS the contracts. Ground every observation in specific contract ids and field "
    "values from the manifest; add no facts beyond it, and do NOT compute sums or totals. If the "
    "manifest does not support an observation, say so.\n"
    "---------------------\n{manifest_str}\n---------------------\n"
    "Question: {query_str}\nAnswer: "
)


@dataclass
class Part:
    """One sub-question and its self-corrected outcome."""

    question: str
    answer: str
    report: Report
    contract_id: str | None = None   # the contract the sub-question was routed (filtered) to
    route: str = ""                  # how it was routed: "explicit_id" | "resolved_token" | "none"
    response: Any = None             # the accepted attempt's retrieval Response (source_nodes -> citations)


@dataclass
class AgentResult:
    """What the controller returns: the final answer plus its per-sub-question trail.

    CONTRACT: ``text`` is always the raw model output, even when the answer was WITHHELD -- it is
    NOT blanked on a grounding failure. A consumer MUST gate display on ``.ok`` (report.ok) and
    show a withhold notice instead of ``text`` when it is False. Internal callers (format_result,
    evaluate_agentic, the service/ API) already do; any new renderer must too."""

    question: str
    kind: str
    text: str          # RAW model output -- display only when .ok is True (see CONTRACT above)
    report: Report     # final grounding outcome; report.ok => text is displayable
    parts: list[Part]

    @property
    def ok(self) -> bool:
        return self.report.ok


class AgenticRag:
    """Router + fan-out + LLM decomposition + verifier-driven self-correction over RagAnswerer."""

    def __init__(self, max_attempts: int = AGENT_MAX_ATTEMPTS):
        self._rag = RagAnswerer()               # loads index + reranker + LLM once; sets Settings.llm
        self._verifier = self._rag.verifier     # the same manifest oracle the pipeline verifies against
        self._known = self._rag.contract_ids    # real contract ids -- guards against filtering on an invented one
        self._wide_reranker = get_reranker(AGENT_WIDE_TOP_N)
        self._max_attempts = max_attempts

    @property
    def rag(self) -> RagAnswerer:
        """The shared RagAnswerer the agent answers through (public for the service/ API)."""
        return self._rag

    # -- routing -------------------------------------------------------------------------
    def classify(self, question: str) -> tuple[str, list[str]]:
        """(kind, ordered contract ids). kind in {"simple","fanout","semantic","analytical",
        "enumerate"}."""
        explicit = list(dict.fromkeys(m.upper() for m in CONTRACT_ID_RE.findall(question)))
        resolved = self._rag.resolve_contract_ids(question)  # resolve once, reuse (was up to 3x)
        # Corpus-wide enumeration: a "list the projects/contracts/locations" question with no
        # explicit id and no resolvable location -- its complete answer IS the manifest, so answer
        # by deterministic enumeration (see answer()), not lossy/nondeterministic summarization or
        # a whole-corpus fan-out. A location-scoped "list all projects in X" resolves an id here
        # and so falls through to the scoped simple/fan-out route below.
        if not explicit and _LIST_CORPUS_RE.search(question) and not resolved:
            return "enumerate", sorted(self._known)
        # Analytical (corpus-wide pattern/anomaly) reasons over the whole manifest at once, so it
        # precedes the fan-out/each rules. A question comparing >=2 *named* contracts is left to
        # the semantic route (cross-contract reasoning over specific ids), not analytical.
        if _ANALYTICAL_RE.search(question) and len(set(explicit)) < 2:
            return "analytical", sorted(self._known)
        if (not explicit and not resolved and _CORPUS_RE.search(question)
                and not _SINGULAR_RE.search(question)):
            return "fanout", sorted(self._known)
        if not explicit and _EACH_RE.search(question):
            # "list/for each project ...": one intent asked of some or all contracts.
            # If the question also names a location, restrict to matching contracts so a
            # location-scoped "all projects in X" doesn't fan out over the whole corpus.
            scope = sorted(resolved) if resolved else sorted(self._known)
            return ("simple" if len(scope) <= 1 else "fanout"), scope
        ids = explicit or sorted(resolved)
        if len(set(ids)) <= 1:
            return "simple", ids
        m = _MULTI_ID_RE.search(question)
        listed = {i.upper() for i in re.findall(_ID, m.group(1))} if m else set()
        # Fan-out only when every named contract sits in one contiguous list and nothing asks to
        # compare them; otherwise it is a true multi-hop question for the LLM to decompose.
        if listed == set(explicit) and explicit and not _COMPARISON_RE.search(question):
            return "fanout", explicit
        return "semantic", ids

    # -- self-correction -----------------------------------------------------------------
    def _answer_verified(self, question: str, contract_id: str | None,
                         search_query: str) -> tuple[str, Report, Any]:
        """answer_once wrapped in the retry ladder.

        Invented id (not in the corpus): withhold before retrieval. Answering it unfiltered let
        another contract's evidence answer it, and the Verifier only blocks an invented id the
        answer text repeats -- "BAYANI R. GAJES" for a 24AF0094 question sailed through.

        Known id: stay filtered on EVERY rung. Dropping the filter would let another contract's
        evidence answer a contract-scoped question -- a cross-contract mis-binding (fix A). And a
        filtered pass is authoritative: when it legitimately finds no value the model punts
        ("not stated"), which is the correct answer, so a punt is accepted rather than retried
        into the wider corpus (fix B) -- only a grounding failure warrants the wider retry."""
        if contract_id is not None and contract_id not in self._known:
            return "", Report(blocks=[f"names contract {contract_id}, which is not in the corpus"],
                              flags=[]), None
        known = contract_id is not None
        if known:
            strategies = [(contract_id, None), (contract_id, self._wide_reranker)]
        else:
            strategies = [(contract_id, None), (contract_id, self._wide_reranker),
                          (None, self._wide_reranker)]
        strategies = strategies[: self._max_attempts + 1]
        resp = report = None
        for cid, reranker in strategies:
            resp = self._rag.answer_once(question, cid, search_query, reranker=reranker)
            # Pass the routed scope so the Verifier can test bindings even when the answer text
            # omits the id (fix C).
            report = self._verifier.check(str(resp), contract_id=contract_id)
            # Known-id filtered pass: retry with wider k if the model punts (the info may
            # exist but ranked below the standard top-k cut), or on a grounding failure.
            # Accept the result when grounded and non-punt, or when already on the wide
            # reranker (no further strategies remain). Never drop the contract filter for a
            # known id -- that would let another contract's evidence answer this question.
            if known:
                if report.ok and not _needs_retry(resp, report):
                    break
                if reranker is not None:   # already on the wide reranker — accept as-is
                    break
            elif not _needs_retry(resp, report):
                break
        return str(resp).strip(), report, resp

    def _answer_subquestion(self, subq: str, stage: Callable[[str, dict], None]) -> Part:
        contract_id, search_query, route = self._rag.route(subq)
        answer, report, resp = self._answer_verified(subq, contract_id, search_query)
        part = Part(subq, answer, report, contract_id, route, resp)
        stage("subanswer", {"part": part})
        return part

    # -- decomposition -------------------------------------------------------------------
    def _fanout_subquestions(self, question: str, ids: list[str]) -> list[str]:
        """Rewrite a multi-contract question into one single-contract question per id: either an
        explicit id list ("contracts X, Y and Z") or a distributive "each project" phrasing."""
        if _MULTI_ID_RE.search(question):
            return [_MULTI_ID_RE.sub(f"for contract {cid}", question, count=1) for cid in ids]
        if _EACH_RE.search(question):
            return [_EACH_RE.sub(f"contract {cid}", question, count=1) for cid in ids]
        # Corpus-wide sweep: "across the X documents in the database" -> "in contract C's X documents"
        base = question
        if _ACROSS_RE.search(base):
            base = _IN_DB_RE.sub("", base)
            return [_ACROSS_RE.sub(f"in contract {cid}'s ", base, count=1) for cid in ids]
        return [_IN_DB_RE.sub(f" for contract {cid}", base, count=1) for cid in ids]

    def _decompose(self, question: str) -> list[str]:
        """Ask the LLM for sub-questions (JSON array). Fail-open to the original question."""
        raw = str(Settings.llm.complete(_DECOMPOSE_TEMPLATE.format(query_str=question))).strip()
        match = re.search(r"\[.*\]", raw, re.S)
        if not match:
            return [question]
        try:
            subs = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [question]
        subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
        return subs or [question]

    # -- top-level -----------------------------------------------------------------------
    def answer(self, question: str,
               on_stage: Callable[[str, dict], None] | None = None) -> AgentResult:
        """Route, answer, verify. ``on_stage(stage, data)`` is an optional observer called as each
        stage completes ("route" | "decompose" | "subanswer" | "combine") -- the service/ API
        streams progress through it, so there is ONE orchestration shared by the CLI and the API.
        It may raise to abort (the service does so on client cancel); the agent never catches."""
        stage = on_stage or (lambda name, data: None)
        kind, ids = self.classify(question)
        stage("route", {"kind": kind, "contract_ids": ids})
        if kind == "enumerate":
            # The complete answer IS the manifest: enumerate it deterministically (no LLM, no
            # retrieval), so the answer is complete and identical every run. Grounded by
            # construction -- built straight from the oracle -- so the report is clean.
            text = format_enumeration(self._rag.manifest)
            stage("combine", {"strategy": "enumerate", "ok": True})
            return AgentResult(question, kind, text, Report(blocks=[], flags=[]), [])

        if kind == "simple":
            part = self._answer_subquestion(question, stage)
            return AgentResult(question, kind, part.answer, part.report, [part])

        if kind == "analytical":
            # One pass over the COMPLETE manifest (already built, whole corpus) -- no retrieval, so
            # the retry ladder (widen k / drop filter) doesn't apply. Verified with the DERIVED
            # BLOCK tier only: the per-contract mis-binding FLAG heuristic mis-reads a legitimate
            # multi-contract synthesis, so the analytical route relaxes it (check_bindings=False)
            # while still withholding any answer that invents a contract.
            manifest_str = self._rag.manifest_text
            text = str(Settings.llm.complete(
                _ANALYTICAL_TEMPLATE.format(manifest_str=manifest_str, query_str=question))).strip()
            report = self._verifier.check(text, check_bindings=False)
            stage("combine", {"strategy": "analytical", "ok": report.ok})
            return AgentResult(question, kind, text, report, [])

        subqs = self._fanout_subquestions(question, ids) if kind == "fanout" \
            else self._decompose(question)
        stage("decompose", {"strategy": kind, "subquestions": subqs})
        parts = [self._answer_subquestion(s, stage) for s in subqs]

        if kind == "fanout":
            # Per-contract listing: each part was verified on its own, so one failed part withholds
            # only itself (its text never shows; the reason stays on the part) -- withholding the
            # whole list threw away every verified part (a corpus sweep lost 5 of 6 over one
            # invented id). All parts failed -> withhold the answer, as before.
            text = "\n".join(f"- {p.answer}" if p.report.ok else
                             f"- {p.contract_id}: answer withheld (it failed the grounding check)"
                             for p in parts)
            report = Report() if any(p.report.ok for p in parts) else Report(
                blocks=[b for p in parts for b in p.report.blocks],
                flags=[f for p in parts for f in p.report.flags])
            stage("combine", {"strategy": "fanout", "ok": report.ok})
            return AgentResult(question, kind, text, report, parts)

        # semantic: combine the (already verified) sub-answers with the LLM, then verify the join.
        context = "\n\n".join(f"Sub-question: {p.question}\nAnswer: {p.answer}" for p in parts)
        combined = str(Settings.llm.complete(
            _COMBINE_TEMPLATE.format(context_str=context, query_str=question))).strip()
        report = self._verifier.check(combined)
        stage("combine", {"strategy": "semantic", "ok": report.ok})
        return AgentResult(question, kind, combined, report, parts)


# Phrasings the small model uses to punt -- treated as a non-answer that should trigger a retry
# (widen k, then drop the filter) before it is accepted. Retrying a genuine "no such value"
# question only costs latency; it never changes a correct answer.
_PUNT_MARKERS = (
    "don't know", "do not know", "not in the context", "not specified", "not provided",
    "not mentioned", "not stated", "not available", "not found", "cannot find",
    "could not find", "no information",
)


def _needs_retry(response, report: Report) -> bool:
    """Retry when grounding failed or the model punted (empty / 'not specified' / ...)."""
    if not report.ok:
        return True
    low = str(response).lower()
    return not low.strip() or any(m in low for m in _PUNT_MARKERS)


def format_result(result: AgentResult) -> str:
    from rag.verify import format_report
    head = f"[{result.kind}] "
    if not result.ok:
        return head + format_report(result.report)
    return f"{head}\n{result.text}\n{format_report(result.report)}"


if __name__ == "__main__":
    import sys

    agent = AgenticRag()
    questions = sys.argv[1:] or [
        "Who is the District Engineer for contracts 24BJ0005 and 24CC0265?",
        "Which contractor was awarded contract 24AJ0052?",
    ]
    for q in questions:
        result = agent.answer(q)
        print(f"\nQ: {q}")
        print(format_result(result))
