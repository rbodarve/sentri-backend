"""Agentic controller over the single-pass RAG (rag.generate.RagAnswerer).

Composes three behaviors on top of the deterministic pipeline, without replacing it:

  router       -- classify a question: "simple" | "fanout" | "semantic" | "analytical" |
                  "enumerate" (a corpus-wide "list all projects/locations" answered straight
                  from the manifest, deterministically -- no LLM, no retrieval) | "rank" (a
                  corpus-wide "highest / rank by amount", the manifest amounts sorted in Decimal) |
                  "aggregate" (a corpus-wide total/average amount, computed in Decimal) |
                  "filter" (a corpus-wide "amount above / below X", compared in Decimal).
  fan-out      -- a multi-contract question asking one intent of several contracts -- an
                  explicit id list, a distributive "each/all projects", or a corpus-wide sweep
                  ("... across the documents in the database") -- is split, by rule, into one
                  single-contract sub-question each (no LLM). A part that fails verification is
                  withheld on its own; the answer is withheld only if every part fails.
  semantic     -- a genuine multi-hop/compound question is split into sub-questions by the LLM
                  (the ONLY place the LLM drives control), then the sub-answers are combined.
  analytical   -- a corpus-wide pattern/anomaly/commonality question reasons over the COMPLETE
                  manifest in one pass (not per-contract, no retrieval added -- the manifest is
                  already the whole corpus), the substrate top-k retrieval structurally lacks.
  self-correct -- every sub-answer is checked by the manifest Verifier; on a block/flag (or an
                  unhelpful "I don't know") it retries with a widened k. A known contract id stays
                  filtered on every rung; only an unfiltered question may drop to the whole corpus.
                  An id that is not in the corpus is withheld before retrieval.

Design constraints this module honors (see CLAUDE.md / the plan):
  * Recall stays 1.000: it reuses RagAnswerer.answer_once unchanged; widening only keeps more
    already-retrieved chunks. It never adds a new retrieval path.
  * No LLM arithmetic: the decompose/combine prompts forbid sums/totals. The agreed replacement
    (a cited calculation request evaluated in Decimal, never by a model) is not wired yet.
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
from rag.manifest import (format_aggregate, format_enumeration, format_filter, format_ranking,
                          has_threshold, parse_threshold)
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
# Corpus-wide ranking cue: an order word AND a ranked manifest field, with no contract named
# ("which contract has the highest awarded amount", "rank the contracts by amount"). Answered by
# comparing the manifest's amounts in Decimal (see answer()) -- no LLM orders numbers.
# The order word is a superlative or a list verb. The first superlative sets the direction
# ("from smallest to largest" -> ascending); a list verb (rank/sort/"order by"/"in ... order")
# or superlatives of both directions ask for the full order, else the top contract. A bare
# "order" is not a list verb: "the change order amount" names a document.
_SUPERLATIVE_RE = re.compile(r"\b(highest|largest|biggest|lowest|smallest)\b", re.I)
_HIGH_WORDS = ("highest", "largest", "biggest")
_RANK_LIST_RE = re.compile(
    r"\b(?:rank(?:ed|ing)?|sort(?:ed)?|order(?:ed)?\s+by|"
    r"in\s+(?:(?:ascending|descending|increasing|decreasing|reverse)\s+)?order)\b", re.I
)
_RANK_FIELD_RE = re.compile(r"\b(?:amounts?|price|cost)\b", re.I)
# ...and it must range over contracts: "which contract", "the contracts", "the corpus". Without
# this, a per-contract follow-up ("which bidder had the lowest bid amount", "the change order")
# would rank the whole corpus and lose its session pin.
_RANK_SCOPE_RE = re.compile(r"\b(?:which|what)\s+(?:contract|project)s?\b|\b(?:contracts|projects|corpus)\b", re.I)


def is_rank_question(question: str) -> bool:
    """The corpus-wide ranking cue (order word AND ranked field AND corpus scope), shared with
    rag.subject so a session never pins its contract onto a question the router ranks."""
    return bool((_SUPERLATIVE_RE.search(question) or _RANK_LIST_RE.search(question))
                and _RANK_FIELD_RE.search(question) and _RANK_SCOPE_RE.search(question))

# Corpus-wide aggregate cue: a total/average word AND the summed field named AND corpus scope
# ("the combined total of all the awarded contract amounts", "the total amount of all contracts").
# Answered by summing the manifest's contract amounts in Decimal (see answer()) -- no LLM adds
# numbers. The field cue is positive: only the contract amount is summed, so "the sum of the bid
# amounts" or "a total amount above 100 million" (a filter) never gets the contract-amount sum.
_TOTAL_RE = re.compile(r"\b(?:total|sum|combined|aggregate)\b", re.I)
_AVERAGE_RE = re.compile(r"\b(?:average|mean)\b", re.I)
_AGGREGATE_FIELD_RE = re.compile(
    r"\b(?:awarded|total)\s+contract\s+(?:amounts?|prices?|values?)\b|"
    r"\bamounts?\s+of\s+(?:all\s+)?(?:the\s+)?(?:contracts|projects)\b", re.I
)
# ...but "Total Contract Amount" is also a per-contract field name: "the total contract amount for
# each of the contracts" / "list the total contract amounts" asks one value per contract (fan-out
# or enumerate), not their sum. Not _EACH_RE: its "all contracts" is the aggregate's own scope.
# And a "which" question selects contracts ("which contracts have a total contract amount above
# 100 million"); an aggregate answers with a value.
_PER_CONTRACT_RE = re.compile(
    r"\b(?:each|every|per)\b|\b(?:list|enumerate)\b|\btotal\s+(?:contract\s+)?amounts\b|"
    r"^\s*which\b", re.I
)


def is_aggregate_question(question: str) -> bool:
    """The corpus-wide total/average cue, shared with rag.subject like is_rank_question. Not when
    the question states a threshold ("the average ... over X"): summing every amount would drop
    the condition."""
    return bool((_TOTAL_RE.search(question) or _AVERAGE_RE.search(question))
                and _AGGREGATE_FIELD_RE.search(question) and _RANK_SCOPE_RE.search(question)
                and not _PER_CONTRACT_RE.search(question) and not has_threshold(question))


# Corpus-wide amount filter: a threshold ("above 100 million", see parse_threshold) AND the
# contract amount named AND corpus scope ("which contracts have an amount above X"). Answered by
# comparing the manifest amounts in Decimal (see answer()) -- no LLM compares numbers. The field
# cue is positive: "a bid amount above X" is not the manifest's contract amount.
_FILTER_FIELD_RE = re.compile(r"\b(?:an|the|contract|awarded|total)\s+amounts?\b", re.I)


def is_filter_question(question: str) -> bool:
    """The corpus-wide amount-threshold cue, shared with rag.subject like is_rank_question."""
    return bool(_FILTER_FIELD_RE.search(question) and _RANK_SCOPE_RE.search(question)
                and parse_threshold(question))


def is_corpus_wide(question: str) -> bool:
    """Every corpus-wide route cue, text-only. classify() takes a corpus-wide route only when this
    matches, and rag.subject never pins a session contract onto it -- one list, so a new route
    cannot be added to one and missed in the other."""
    return bool(_LIST_CORPUS_RE.search(question) or _EACH_RE.search(question)
                or _ANALYTICAL_RE.search(question) or _CORPUS_RE.search(question)
                or is_rank_question(question) or is_aggregate_question(question)
                or is_filter_question(question))

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
# sums/totals (no model computes -- see the module docstring), honest when the manifest can't support it.
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

    CONTRACT: ``text`` is the raw model output, even when the answer was WITHHELD -- it is NOT
    blanked on a grounding failure. (Exception: a fan-out already replaces each FAILED part's text
    with a "withheld" line, so a partly withheld fan-out is displayable as is.) A consumer MUST gate display on ``.ok`` (report.ok) and
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
        "enumerate","rank","aggregate","filter"}."""
        explicit = list(dict.fromkeys(m.upper() for m in CONTRACT_ID_RE.findall(question)))
        resolved = self._rag.resolve_contract_ids(question)  # resolve once, reuse (was up to 3x)
        # A corpus-wide route needs its cue in is_corpus_wide (shared with rag.subject's pin rule).
        if is_corpus_wide(question):
            if not explicit and not resolved:
                # Corpus-wide amount filter ("which contracts have an amount above X"): each amount
                # is compared in Decimal from the manifest (see answer()). First: a threshold is the
                # stronger cue, so "list / rank the contracts above X" filters, never lists all.
                if is_filter_question(question):
                    return "filter", sorted(self._known)
                # Corpus-wide ranking ("which contract has the highest amount"): every contract's
                # value is needed and the comparison must not be the LLM's, so it is answered by
                # sorting the manifest (see answer()). Precedes enumerate so "list the contracts by
                # amount" ranks.
                if is_rank_question(question):
                    return "rank", sorted(self._known)
                # Corpus-wide total/average: every amount is an operand, so it is summed in Decimal
                # from the manifest (see answer()), never by the LLM through a fan-out combine.
                # After rank, so "the highest total amount" ranks.
                if is_aggregate_question(question):
                    return "aggregate", sorted(self._known)
            # Corpus-wide enumeration: a "list the projects/contracts/locations" question with no
            # explicit id and no resolvable location -- its complete answer IS the manifest, so
            # answer by deterministic enumeration (see answer()), not lossy/nondeterministic
            # summarization or a whole-corpus fan-out. A location-scoped "list all projects in X"
            # resolves an id here and so falls through to the scoped simple/fan-out route below.
            if not explicit and _LIST_CORPUS_RE.search(question) and not resolved:
                return "enumerate", sorted(self._known)
            # Analytical (corpus-wide pattern/anomaly) reasons over the whole manifest at once, so
            # it precedes the fan-out/each rules. A question comparing >=2 *named* contracts is left
            # to the semantic route (cross-contract reasoning over specific ids), not analytical.
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
        """Rewrite a multi-contract question into one single-contract question per id: an explicit
        id list ("contracts X, Y and Z"), a distributive "each project" phrasing, or a corpus-wide
        sweep ("... across the documents in the database")."""
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
        if kind in ("enumerate", "rank", "aggregate", "filter"):
            # The complete answer IS the manifest: enumerate it, or rank / total its amounts in
            # Decimal, deterministically (no LLM, no retrieval), so the answer is complete and
            # identical every run. Grounded by construction -- built straight from the oracle -- so
            # the report is clean. The LLM only extracted the verbatim strings at build time.
            if kind == "enumerate":
                text = format_enumeration(self._rag.manifest)
            elif kind == "aggregate":
                # "the total and the average": both lines (the route matched at least one word).
                ops = tuple(op for op, cue in (("total", _TOTAL_RE), ("average", _AVERAGE_RE))
                            if cue.search(question))
                text = format_aggregate(self._rag.manifest, ops)
            elif kind == "filter":
                text = format_filter(self._rag.manifest, *parse_threshold(question))
            else:
                highs = [w.lower() in _HIGH_WORDS for w in _SUPERLATIVE_RE.findall(question)]
                descending = not highs or highs[0]
                both_ends = len(set(highs)) == 2  # "highest and the lowest": show the full order
                text = format_ranking(self._rag.manifest, descending,
                                      top_only=not (both_ends or _RANK_LIST_RE.search(question)))
            stage("combine", {"strategy": kind, "ok": True})
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


def manifest_router(manifest: list[dict]) -> Callable[[str], tuple[str, list[str]]]:
    """AgenticRag.classify on a manifest-only stub: the production router without loading the
    index or the LLM. It needs only the known ids and the manifest-backed resolver, so the eval
    gates (rag.evaluate, rag.evaluate_agentic --routes) route exactly as production does."""
    from types import SimpleNamespace
    from rag.generate import _resolver_index, resolve_contract_ids
    resolver, phrases = _resolver_index(manifest)
    router = SimpleNamespace(
        _known={r["contract_id"] for r in manifest},
        _rag=SimpleNamespace(resolve_contract_ids=lambda q: resolve_contract_ids(q, resolver, phrases)))
    return lambda question: AgenticRag.classify(router, question)


# Phrasings the small model uses to punt -- treated as a non-answer that should trigger a retry
# (widen k; an unfiltered question may then also drop to the whole corpus -- a known contract id
# never does) before it is accepted. Retrying a genuine "no such value"
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
