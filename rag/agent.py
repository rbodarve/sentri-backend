"""Agentic controller over the single-pass RAG (rag.generate.RagAnswerer).

Composes three behaviors on top of the deterministic pipeline, without replacing it:

  router       -- classify a question: "simple" | "fanout" | "semantic" | "analytical" |
                  "enumerate" (a corpus-wide "list all projects/locations" answered straight
                  from the manifest, deterministically -- no LLM, no retrieval) | "rank" (a
                  corpus-wide "highest / rank by amount", the manifest amounts sorted in Decimal) |
                  "aggregate" (a corpus-wide total/average amount, computed in Decimal) |
                  "filter" (a corpus-wide "amount above / below X", compared in Decimal) |
                  "calc" (arithmetic over one contract's figures; out of scope -> withheld).
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
  * No LLM arithmetic: the decompose/combine prompts forbid sums/totals. The calc route asks only
    for cited operands + one op (rag.calc); deterministic guards check them and Decimal computes.
  * A bad LLM split cannot emit an unverified claim: every sub-answer passes the deterministic
    route + Verifier before it is combined.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable

from llama_index.core import Settings
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts import PromptTemplate

from rag import calc
from rag.config import AGENT_MAX_ATTEMPTS, AGENT_WIDE_TOP_N
from rag.enrich import CONTRACT_ID_RE
from rag.generate import RagAnswerer
from rag.manifest import (format_aggregate, format_enumeration, format_filter, format_ranking,
                          has_threshold, parse_threshold, threshold_ends_clause)
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
    """The corpus-wide total/average cue, shared with rag.subject like is_rank_question. A stated
    threshold ("the average ... over X") must be fully parsed, as answer() totals only the
    contracts it selects; else (negated, a range) summing every amount would drop the condition."""
    return bool((_TOTAL_RE.search(question) or _AVERAGE_RE.search(question))
                and _AGGREGATE_FIELD_RE.search(question) and _RANK_SCOPE_RE.search(question)
                and not _PER_CONTRACT_RE.search(question)
                and (not has_threshold(question) or contract_threshold(question)))


# Corpus-wide amount filter: a threshold ("above 100 million", see parse_threshold) AND the
# contract amount named AND corpus scope ("which contracts have an amount above X"). Answered by
# comparing the manifest amounts in Decimal (see answer()) -- no LLM compares numbers. The field
# cue is positive: "a bid amount above X" is not the manifest's contract amount.
_FILTER_FIELD_RE = re.compile(r"\b(?:an|the|contract|awarded|total|whose)\s+amounts?\b", re.I)
# ...or a count of contracts ("how many contracts are above X"): the filter header states the count.
_COUNT_RE = re.compile(r"\bhow\s+many\s+(?:contracts|projects)\b", re.I)
# ...or a selection of contracts ("which contracts are above 100 million", "list the contracts
# above PHP X"). No amount noun binds either cue (count or select), so the figure must carry its own money cue (see
# contract_threshold) and end its clause: "... above 100 million in bid amount" names another field.
_SELECT_RE = re.compile(
    r"\b(?:which|what)\s+(?:contracts|projects)\b|\blist\s+(?:all\s+)?(?:the\s+)?(?:contracts|projects)\b",
    re.I
)
# ...and the threshold must bind to the contracts or their amount: the one threshold
# parse_threshold matched (comparison word + figure) must follow "contracts" /
# "an|the|contract|awarded|total|whose amount", with at most a short bridge ("an amount of over X",
# "whose total contract amount is below X", "contracts are above X"). So "with a bid amount above X" never filters or
# totals the contract amounts, and "contracts under review" is no threshold at all.
_AMOUNT_NOUN = r"\b(?:an|the|contract|awarded|total|whose)\s+amounts?"
_BRIDGE = r"(?:\s+(?:of|is|are|that\s+(?:is|are)|which\s+(?:is|are)))?\s*$"
_THRESHOLD_SUBJECT_RE = re.compile(rf"(?:\b(?:contracts|projects)|{_AMOUNT_NOUN}){_BRIDGE}", re.I)
# A figure with no money cue (peso marker, scale word, "pesos") is pesos only when it binds an
# amount noun ("an amount above 50,000,000") or -- in a question naming the amount -- bare
# "contracts" when the figure ends its clause ("the total contract amount of all contracts above
# 100,000,000?"). So "how many contracts are over 300 days", "the average amount of the contracts
# that are over 200 days" and "the average amount of the contracts over 200 days" do not parse.
_AMOUNT_PROOF_RE = re.compile(_AMOUNT_NOUN + _BRIDGE, re.I)
_BARE_SUBJECT_RE = re.compile(r"\b(?:contracts|projects)\s*$", re.I)


def contract_threshold(question: str) -> tuple[str, Decimal] | None:
    """parse_threshold's (op, X), only when the threshold binds to the contract amount."""
    names_amount = _FILTER_FIELD_RE.search(question) or _AGGREGATE_FIELD_RE.search(question)
    return parse_threshold(question, after=_THRESHOLD_SUBJECT_RE, proof=_AMOUNT_PROOF_RE,
                           end_proof=_BARE_SUBJECT_RE if names_amount else None)


def is_filter_question(question: str) -> bool:
    """The corpus-wide amount-threshold cue, shared with rag.subject like is_rank_question."""
    return bool((_FILTER_FIELD_RE.search(question)
                 or ((_COUNT_RE.search(question) or _SELECT_RE.search(question))
                     and threshold_ends_clause(question)))
                and _RANK_SCOPE_RE.search(question)
                and contract_threshold(question))


def is_corpus_wide(question: str) -> bool:
    """Every corpus-wide route cue, text-only. classify() takes a corpus-wide route only when this
    matches, and rag.subject never pins a session contract onto it -- one list, so a new route
    cannot be added to one and missed in the other."""
    return bool(_LIST_CORPUS_RE.search(question) or _EACH_RE.search(question)
                or _ANALYTICAL_RE.search(question) or _CORPUS_RE.search(question)
                or is_rank_question(question) or is_aggregate_question(question)
                or is_filter_question(question))


# Calculation cue (E3): arithmetic over ONE contract's figures. Positive cues only: "total" alone is
# a lookup ("the total contract amount"), so it never makes a calc. An add, a subtract, or a
# multi-step word (average, percentage) is a cue; the multi-step ones only so they are withheld.
_CALC_ADD_RE = re.compile(
    r"\b(?:combined|sum\s+of|add(?:ed)?\s+(?:up|together))\b|"
    r"\btotal\s+(?:amount|cost|price|value)\s+of\b[^?]*\bitems\b", re.I  # "the total amount of the Part A items"
)
_CALC_SUB_RE = re.compile(
    r"\b(?:difference|subtract|minus)\b|"
    r"\bhow\s+much\s+(?:lower|less|larger|more|higher|bigger|smaller|greater)\b[^,?]*?\bthan\b", re.I
)  # "how much more time was granted" has no "than": a lookup
# Average over a stated set ("the average X of the three bidders"); a share or percentage of a
# stated whole. "The average daily output" and "the percentage of completion" are lookups.
_CALC_MULTI_RE = re.compile(
    r"\baverage\b[^?]*?\bof\s+(?:the|all|both|two|three|four|five|\d+)\b|"
    r"\bwhat\s+share\b|\bpercent(?:age)?\s+of\s+the\b", re.I
)
# "per" is no cue ("the unit price per meter" is a lookup), but inside a calc part it is a second step.
_CALC_PER_RE = re.compile(r"\bper\b", re.I)
# "all N items": the clause names a set, not the items, so no operand label can be bound (E4-C).
_CALC_ALL_N_RE = re.compile(r"\ball (?:\d+|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)
# The strict subtract pattern fixes the order: "how much lower/less is A than B" -> B - A,
# "how much larger/more/higher is A than B" -> A - B. Any other subtract phrasing is withheld.
_CALC_SUB_STRICT_RE = re.compile(
    r"\bhow\s+much\s+(lower|less|larger|more|higher)\s+is\s+(.+?)\s+than\s+(.+?)[\s?.]*$", re.I
)
# A compound question joins its parts with ", and <wh-word>" ("..., and on what date was ...").
_CLAUSE_RE = re.compile(
    r",?\s+and\s+(?=(?:(?:on|in|at|by|for|from)\s+)?(?:what|which|who|whom|whose|when|where|how)\b)", re.I
)


def _calc_cue(clause: str) -> bool:
    return bool(_CALC_ADD_RE.search(clause) or _CALC_SUB_RE.search(clause)
                or _CALC_MULTI_RE.search(clause))


def is_calc_question(question: str) -> bool:
    """A calculation cue and no corpus-wide cue. classify() also needs exactly one contract;
    a corpus-wide sum stays aggregate. Not in is_corpus_wide: a follow-up sum keeps its pin."""
    return _calc_cue(question) and not is_corpus_wide(question)


@dataclass
class CalcScope:
    """What calc_scope() found. ``op`` is "" when the question is out of scope (``reason`` says why)."""

    op: str                     # "add" | "subtract" | "" (out of scope: withhold)
    reason: str = ""            # why it is out of scope
    calc: str = ""              # the calc part of the question
    facts: tuple[str, ...] = ()  # a compound question's fact parts, each answered on its own
    order: tuple[str, ...] = ()  # subtract: (minuend, subtrahend), fixed by the strict pattern


def calc_scope(question: str) -> CalcScope:
    """Split a calc question into its calc part and fact parts, and decide whether one operation
    covers the calc part. Text only; "all N items" is checked after the request (E4)."""
    clauses = _CLAUSE_RE.split(question)
    calcs = [c for c in clauses if _calc_cue(c)]
    if len(calcs) != 1:
        return CalcScope("", "more than one calculation in one question")
    calc = calcs[0]
    facts = tuple(c for c in clauses if c is not calc)
    multi = _CALC_MULTI_RE.search(calc) or _CALC_PER_RE.search(calc)
    if multi:
        word = multi.group(0).lower()
        name = "average" if word.startswith("average") else "per" if word == "per" else "percentage"
        return CalcScope("", f"multi-step calculation ({name})", calc, facts)
    if _CALC_ALL_N_RE.search(calc):
        return CalcScope("", "the question names no items to cite", calc, facts)
    if _CALC_ADD_RE.search(calc) and _CALC_SUB_RE.search(calc):
        return CalcScope("", "more than one calculation in one question", calc, facts)
    if _CALC_SUB_RE.search(calc):
        m = _CALC_SUB_STRICT_RE.search(calc)
        if not m:
            return CalcScope("", "subtract outside the strict pattern "
                                 "'how much lower/less/larger/more/higher is A than B'", calc, facts)
        word, a, b = m.groups()
        order = (b, a) if word.lower() in ("lower", "less") else (a, b)
        return CalcScope("subtract", "", calc, facts, order)
    return CalcScope("add", "", calc, facts)


def _join_parts(parts: list[Part], show_reason: bool = False) -> tuple[str, Report]:
    """One line per verified part; a failed part withholds only itself (its reason stays on the
    part, and shows in the line with ``show_reason``). All parts failed -> withhold the answer."""
    def why(p: Part) -> str:
        return "; ".join(p.report.blocks) if show_reason else "it failed the grounding check"
    text = "\n".join(f"- {p.answer}" if p.report.ok else
                     f"- {p.contract_id}: answer withheld ({why(p)})" for p in parts)
    report = Report() if any(p.report.ok for p in parts) else Report(
        blocks=[b for p in parts for b in p.report.blocks],
        flags=[f for p in parts for f in p.report.flags])
    return text, report


_LEAD_AND_RE = re.compile(r"^[\s,]*(?:and\b)?\s*", re.I)


def fact_question(fact: str, contract_id: str) -> str:
    """A compound question's fact part as a standalone sub-question, filtered to the route's
    contract id (the fact clause alone names no contract)."""
    return f"For contract {contract_id}, {_LEAD_AND_RE.sub('', fact).strip()}"


_CALC_TEMPLATE = PromptTemplate(
    "Pick the operands of this calculation from the document chunks below. Do NOT compute.\n"
    "For each operand give: label = the item or bidder name as written in its table row or "
    "sentence; value = the number copied exactly as written in the chunk (the whole cell), in "
    "digits, commas and a point only -- never the amount in words; chunk = the handle (C1, C2, ...) "
    "of the chunk the value is copied from. Then op = the one operation the calculation asks for."
    "\nCalculation: {query_str}\n\n{chunks_str}\n"
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
        "enumerate","rank","aggregate","filter","calc"}."""
        explicit = list(dict.fromkeys(m.upper() for m in CONTRACT_ID_RE.findall(question)))
        resolved = self._rag.resolve_contract_ids(question)  # resolve once, reuse (was up to 3x)
        # A corpus-wide route needs its cue in is_corpus_wide (shared with rag.subject's pin rule).
        if is_corpus_wide(question):
            if not explicit and not resolved:
                # "The total of the contracts above X": the threshold selects the summed amounts
                # (see answer()). Before filter, which would only list them; not a rank question,
                # so "the highest total amount of the contracts above X" ranks the subset.
                if (contract_threshold(question) and is_aggregate_question(question)
                        and not is_rank_question(question)):
                    return "aggregate", sorted(self._known)
                # Corpus-wide ranking ("which contract has the highest amount"): every contract's
                # value is needed and the comparison must not be the LLM's, so it is answered by
                # sorting the manifest (see answer()). Precedes enumerate so "list the contracts by
                # amount" ranks. A parsed threshold ranks only the contracts it selects ("the lowest
                # amount among the contracts above X"); so before filter, which would only list them.
                # A threshold that does not parse ("... with a bid amount above X") falls through:
                # ranking every contract would drop it. Here, not in is_rank_question: the aggregate
                # rule above and rag.subject's no-pin rule still need the rank cue.
                if is_rank_question(question) and (not has_threshold(question)
                                                   or contract_threshold(question)):
                    return "rank", sorted(self._known)
                # Corpus-wide amount filter ("which contracts have an amount above X"): each amount
                # is compared in Decimal from the manifest (see answer()). Before enumerate: a
                # threshold is the stronger cue, so "list the contracts above X" filters, never lists all.
                if is_filter_question(question):
                    return "filter", sorted(self._known)
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
        # Arithmetic over one contract's figures: computed in Decimal, never by the LLM (see answer()).
        if len(set(ids)) == 1 and is_calc_question(question):
            return "calc", ids
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

    def _answer_calc(self, scope: CalcScope, contract_id: str,
                     stage: Callable[[str, dict], None]) -> Part:
        """Option C (E4-C): one schema-constrained request for operands + op over this pass's
        document chunks; every guard checks each operand; the answer lists them with citations and
        computes nothing. Any failed check withholds, with no retry (the same chunks again)."""
        _, search_query, route = self._rag.route(scope.calc)
        chunks = calc.handles(calc.doc_chunks(
            self._rag.context(scope.calc, contract_id, search_query)))
        text, reason, picked = "", "no document chunk retrieved", []
        if chunks:
            prompt = _CALC_TEMPLATE.format(query_str=scope.calc, chunks_str="\n\n".join(
                f"[{h}]\n{n.node.text}" for h, n in chunks.items()))
            raw = Settings.llm.chat([ChatMessage(role="user", content=prompt)],
                                    format=calc.schema(list(chunks))).message.content
            try:
                request = json.loads(raw, parse_float=Decimal)
            except json.JSONDecodeError:
                request = {}
            picked, reason = calc.check(request, chunks, scope.calc, scope.op, scope.order)
            text = calc.cite(picked) if picked else ""
        report = self._verifier.check(text) if text else Report(
            blocks=[f"calculation withheld: {reason}"], flags=[])
        part = Part(scope.calc, text, report, contract_id, route,
                    Response(response=text, source_nodes=[n for _, _, n in picked]))
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
                text = format_aggregate(self._rag.manifest, ops, contract_threshold(question))
            elif kind == "filter":
                text = format_filter(self._rag.manifest, *contract_threshold(question))
            else:
                highs = [w.lower() in _HIGH_WORDS for w in _SUPERLATIVE_RE.findall(question)]
                descending = not highs or highs[0]
                both_ends = len(set(highs)) == 2  # "highest and the lowest": show the full order
                text = format_ranking(self._rag.manifest, descending,
                                      top_only=not (both_ends or _RANK_LIST_RE.search(question)),
                                      threshold=contract_threshold(question))
            stage("combine", {"strategy": kind, "ok": True})
            return AgentResult(question, kind, text, Report(blocks=[], flags=[]), [])

        if kind == "calc":
            scope = calc_scope(question)
            if not scope.op:  # out of scope: withhold with the reason, never compute
                stage("combine", {"strategy": kind, "ok": False})
                return AgentResult(question, kind, "", Report(
                    blocks=[f"calculation out of scope: {scope.reason}"], flags=[]), [])
            part = self._answer_calc(scope, ids[0], stage)
            if not scope.facts:
                return AgentResult(question, kind, part.answer, part.report, [part])
            # Compound: each fact part is its own filtered, verified sub-question (the fan-out join).
            parts = [part] + [self._answer_subquestion(fact_question(f, ids[0]), stage)
                              for f in scope.facts]
            text, report = _join_parts(parts, show_reason=True)
            stage("combine", {"strategy": kind, "ok": report.ok})
            return AgentResult(question, kind, text, report, parts)
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
            text, report = _join_parts(parts)
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


def _self_check() -> None:
    """Model-free calc_scope pins on the E1 held-out DEV rows (`python -m rag.agent --check`).
    Sealed rows are never read before E5."""
    rows = [r for r in json.loads(open("eval/calc_heldout.json", encoding="utf-8").read())
            if r["split"] == "dev"]
    # kind -> (op, reason fragment); a subtract row also pins (minuend, subtrahend) fragments.
    expected = {"add2": ("add", ""), "add3": ("add", ""), "all_n": ("", "names no items to cite"),
                "split_table": ("add", ""), "compound": ("add", ""), "sub_in": ("subtract", ""),
                "sub_out": ("", "strict pattern"), "average": ("", "average"),
                "percent": ("", "percentage"), "multi_calc": ("", "more than one calculation")}
    order = {"E08": ("5 Jewel", "Amethyst"), "E09": ("Approved Budget", "contract price"),
             "E10": ("Total Calculated Bid", "Total Bid as Read")}
    assert {r["kind"] for r in rows} == set(expected), "an E1 kind has no calc_scope pin"
    for r in rows:
        q, (op, reason) = r["question"], expected[r["kind"]]
        scope = calc_scope(q)
        assert is_calc_question(q), f"{r['id']}: no calc cue: {q!r}"
        assert scope.op == op and reason in scope.reason, f"{r['id']}: {scope}"
        if r["kind"] == "compound":
            assert len(scope.facts) == 1 and "Notice of Award" in scope.facts[0], f"{r['id']}: {scope}"
        elif op:
            assert not scope.facts, f"{r['id']}: a fact part split off a plain calc: {scope}"
        if r["id"] in order:
            assert all(k in s for k, s in zip(order[r["id"]], scope.order)), f"{r['id']}: {scope.order}"
    for q in ("What is the total contract amount of 24CC0265?",
              "What is the unit price per linear meter of the sheet piles in 24CC0265?"):
        assert not is_calc_question(q), f"a lookup has a calc cue: {q!r}"
    print(f"OK: calc_scope pins ({len(rows)} E1 dev rows, {len(expected)} kinds)")
    failed = [name for name, ok in _calc_pins(rows) if not ok]
    assert not failed, f"calc pins failed: {failed}"


def _calc_pins(rows: list[dict]) -> list[tuple[str, bool]]:
    """E4 guard + evaluator pins, model-free, on database/ chunks and E1 DEV rows only (expected
    values from calc_heldout.json). Prints one line per pin; an exception counts as a fail."""
    from llama_index.core.schema import NodeWithScore, TextNode

    from rag.loader import load_nodes

    nodes = {n.node_id: NodeWithScore(node=n, score=1.0) for n in load_nodes()}
    record = NodeWithScore(node=TextNode(  # the contract record node: is_manifest, never citable
        text="CONTRACT RECORD (the contract in question): Contract 24AJ0052 | contract price: "
             "19,109,972.23", metadata={"is_manifest": True}), score=1.0)
    by_id = {r["id"]: r for r in rows}
    boq = "e3cc4331-3b05-4d96-82e9-a7da425aece8"

    def outcome(r: dict, operands: list[dict], extra: tuple = (), question: str = "",
                compute: bool = False) -> tuple[str, str]:
        """(text, reason) of the option-C path (calc.check + calc.cite) on these operands, in a
        context of their chunks + the record; ``compute`` uses the dead calc.run path instead (the
        evaluator pins). ``question`` replaces the row's question (a pin's variant)."""
        scope = calc_scope(question or r["question"])
        ids = dict.fromkeys([o["chunk"] for o in operands if o["chunk"] in nodes] + list(extra))
        context = [nodes[i] for i in ids] + [record]
        chunks = calc.handles(calc.doc_chunks(context))
        # Cite by handle, as the model does; an id with no handle (the record) stays as it is.
        to_handle = {n.node.node_id: h for h, n in chunks.items()}
        request = {"operands": [{**o, "chunk": to_handle.get(o["chunk"], o["chunk"])}
                                for o in operands], "op": scope.op}
        if compute:
            tape, reason, _ = calc.run(request, chunks, scope.calc, scope.op, scope.order)
            return tape, reason
        picked, reason = calc.check(request, chunks, scope.calc, scope.op, scope.order)
        return (calc.cite(picked) if picked else ""), reason

    def gold(r: dict) -> list[dict]:
        return [{"label": o["label"], "value": o["cell"], "chunk": o["chunk"]} for o in r["operands"]]

    def answers(r: dict, operands: list[dict]) -> bool:
        """Option C: lists every gold cell, says the total is not computed, and shows no total."""
        text, _ = outcome(r, operands)
        return (all(o["cell"] in text for o in r["operands"]) and "The total is not computed." in text
                and f"{Decimal(r['expected']):,.2f}" not in text)

    def computes(r: dict, operands: list[dict]) -> bool:
        """The dead evaluator path (calc.run): the tape ends in the expected value."""
        tape, _ = outcome(r, operands, compute=True)
        return tape.endswith(f"= {Decimal(r['expected']):,.2f}")

    def no_evaluate() -> bool:
        """The option-C path never calls evaluate(): E01 gold still answers when it raises."""
        real = calc.evaluate
        calc.evaluate = lambda op, values: 1 / 0
        try:
            return answers(e01, gold(e01))
        finally:
            calc.evaluate = real

    def withholds(r: dict, operands: list[dict], extra: tuple = (), question: str = "") -> bool:
        tape, reason = outcome(r, operands, extra, question)
        return not tape and bool(reason)

    e01, e04, e05 = by_id["E01"], by_id["E04"], by_id["E05"]
    # E05 asks Rodekom + Grace + R.U. Aquino; this variant drops Grace from the question.
    no_grace = e05["question"].replace("Rodekom General Construction and Enterprise, Grace "
                                       "Construction Corporation and", "Rodekom General "
                                       "Construction and Enterprise and")
    # E04 lists three items without a count; this variant states it ("the three items").
    three_items = e04["question"].replace("combined Total Amount of", "combined Total Amount of "
                                          "the three items")
    # E01 with the OSH program in place of the Furnished item, so its label is a named item.
    osh = e01["question"].replace("the two Structural Steel Sheet Piles items (Furnished and Driven)",
                                  "Occupational Safety and Health Program and Structural Steel "
                                  "Sheet Piles, Driven")
    furnished = {"label": "Sheet Piles, Furnished", "chunk": boq}
    # The BAC-resolution sentence "... Rodekom ... total Calculated Bid ... (Php140,274,481.48)".
    e05_text = next(i for i in nodes if i.startswith("94353284"))

    def e20_try2_withholds() -> bool:
        """E20's logged try-2 request (item-number labels), checked as an add over the BOQ. E20 is
        out of scope in calc_scope, so this calls calc.check directly: binding alone must stop it."""
        q = by_id["E20"]["question"]
        chunks = calc.handles(calc.doc_chunks([nodes[boq]]))
        request = {"op": "add", "operands": [
            {"label": lab, "value": val, "chunk": "C1"} for lab, val in
            (("801(6)", "467,614.56"), ("1700(1)", "9,861.75"), ("1704(1)b", "992,249.94"),
             ("1701(4)", "444,635.77"))]}
        picked, reason = calc.check(request, chunks, q, "add")
        return not picked and "not an item the question names" in reason

    pins = {
        # False-withhold check (option C): the gold operands of every in-scope dev answer row are
        # listed with citations and no total. Evaluator pins (dead path): a subtract row in both
        # orders computes the expected value (a sign flip or wrong minuend fails).
        **{f"gold {r['id']}": (lambda r=r: answers(r, gold(r)))
           for r in rows if r["expect"] == "answer" and calc_scope(r["question"]).op},
        **{f"gold reversed {i} (evaluator)": (lambda i=i: computes(by_id[i], gold(by_id[i])[::-1]))
           for i in ("E08", "E09", "E10")},
        "C path never calls evaluate()": no_evaluate,
        "text field: E05 Rodekom from the 'Calculated Bid' sentence on the As Read question":
            lambda: withholds(e05, [{"label": "Rodekom", "value": "140,274,481.48",
                                     "chunk": e05_text}] + gold(e05)[1:]),
        "binding, no 'all N' exemption: E20's try-2 request -> withhold": e20_try2_withholds,
        "forged value": lambda: withholds(e01, [{**furnished, "value": "44,816,185.72"},
                                                gold(e01)[1]]),
        "value from another chunk": lambda: withholds(  # an E03 cell of 3611e9c0, cited as the BOQ
            e01, [{**furnished, "value": "143,617,975.00"}, gold(e01)[1]],
            extra=(by_id["E03"]["operands"][0]["chunk"],)),
        "partial cell 448,700 of 448,700.36": lambda: "Occupational" in osh and withholds(e01, [
            {"label": "Occupational Safety and Health Program", "value": "448,700", "chunk": boq},
            gold(e01)[1]], question=osh),
        "wrong column (Unit Cost, not Total Amount)": lambda: withholds(
            e01, [{**furnished, "value": "3,787.71"}, gold(e01)[1]]),
        "wrong row (OSH Total Amount as Sheet Piles)": lambda: withholds(
            e01, [{**furnished, "value": "448,700.36"}, gold(e01)[1]]),
        "value cited from the record node": lambda: withholds(by_id["E09"], [  # E09 names it
            {"label": "contract price", "value": "19,109,972.23", "chunk": record.node.node_id},
            gold(by_id["E09"])[0]]),
        "question binding: Grace on a Rodekom + R.U. Aquino question": lambda: (
            "Grace" not in no_grace and withholds(e05, gold(e05), question=no_grace)),
        "duplicate operand (#256 shape: one cell twice)": lambda: withholds(
            e01, [gold(e01)[0], gold(e01)[0]]),
        "stated count: 'the three items', 2 operands (E04 variant)": lambda: (
            "the three items" in three_items and withholds(e04, gold(e04)[:2], question=three_items)),
        "divide by zero": lambda: calc.evaluate("divide", [Decimal("5"), Decimal("0")]) is None,
        "handles are C1..Cn in context order": lambda: list(calc.handles(calc.doc_chunks(
            [nodes[boq], record, nodes[by_id["E03"]["operands"][0]["chunk"]]]))) == ["C1", "C2"],
        "unknown handle C99 -> withhold": lambda: withholds(e01, [{**gold(e01)[0], "chunk": "C99"},
                                                                  gold(e01)[1]]),
        "compound join shows the calc withhold reason": lambda: "two operands cite the same cell" in
            _join_parts([Part("calc", "", Report(blocks=["calculation withheld: two operands cite "
                                                         "the same cell"], flags=[]), "24A00153"),
                         Part("fact", "MAR 05 2024", Report(), "24A00153")], show_reason=True)[0],
        "E24 fact part routed to 24A00153": lambda: (
            m := CONTRACT_ID_RE.search(fact_question(calc_scope(by_id["E24"]["question"]).facts[0],
                                                     "24A00153"))) is not None
            and m.group(0).upper() == "24A00153",
    }
    results = []
    for name, pin in pins.items():
        try:
            ok = bool(pin())
        except Exception as exc:  # noqa: BLE001 -- a crash is a failed pin, never a pass
            ok, name = False, f"{name} ({type(exc).__name__}: {exc})"
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        results.append((name, ok))
    return results


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _self_check()
        sys.exit()
    agent = AgenticRag()
    questions = sys.argv[1:] or [
        "Who is the District Engineer for contracts 24BJ0005 and 24CC0265?",
        "Which contractor was awarded contract 24AJ0052?",
    ]
    for q in questions:
        result = agent.answer(q)
        print(f"\nQ: {q}")
        print(format_result(result))
