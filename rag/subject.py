"""Conversation subject: which contract(s) a follow-up question is about.

The agent answers each question on its own, so a follow-up that names no contract ("Who are the
bidders for this Contract ID?", "What is the Contractor's contact number?") ran unfiltered and was
answered about whichever contract ranked first -- 15 of 105 PART 1 follow-ups in docs/queries.txt
came back about the wrong contract, which the Verifier can't see (each answer is internally
consistent). SubjectTracker carries the conversation's subject forward and pins it onto such a
question, deterministically -- no LLM, the router's own id/place resolver decides.

Rules, in order, for resolve(question):
  reset       "new question" / "another project" / "any project"  -> clear the subject, no pin
  anchored    names a contract id or a place/name the resolver binds  -> ask as-is (the subject
              moves in observe()); if it also compares against "it/this/that", pin the union
  contractor  names a contractor ("the Rodekom contract") -> pin its contract (the router's
              resolver knows places and project names, not contractors)
  corpus-wide "list all projects", "across the documents", "in the database", "which contract
              has the highest amount", "the total amount of all contracts", "which contracts/
              projects ..." (unless it points back: "its", "the first project") -> ask as-is
  earlier     "the first/second/previous/other project" -> pin that earlier subject
  follow-up   anything else, when a subject exists -> pin the current subject
A pin of several contracts reads "for contracts A, B and C", which the router fans out.

observe(question, answer) is called only for a VERIFIED answer: a withheld turn never moves the
subject, and an invented id never becomes one (only manifest ids are tracked). The subject is the
contract(s) the question was routed to, else -- for an unanchored question like "List one project
in the database" -- the contract(s) its answer names.
"""

from __future__ import annotations

import re

from rag.agent import (_ANALYTICAL_RE, _CORPUS_RE, _EACH_RE, _LIST_CORPUS_RE, is_aggregate_question,
                       is_rank_question)
from rag.enrich import CONTRACT_ID_RE

_RESET_RE = re.compile(
    r"\b(?:new question|start over|another (?:project|contract)|different (?:project|contract)|"
    r"any (?:project|contract))\b", re.I
)
_COMPARE_RE = re.compile(r"\b(?:compare|compared|versus|vs\.?|difference|same as|similar to)\b", re.I)
_ANAPHOR_RE = re.compile(r"\b(?:it|this|that|these|those|them|its)\b", re.I)
_WHICH_PLURAL_RE = re.compile(r"^\s*which\s+(?:contracts|projects)\b", re.I)  # selects across all
_ORDINAL_RE = re.compile(
    r"\b(first|second|third|previous|earlier|other)\s+(?:one|project|contract)\b", re.I
)
_ORDINAL_INDEX = {"first": 0, "second": 1, "third": 2}
_COMPANY_STOP = {
    "and", "gen", "construction", "supply", "general", "enterprise", "enterprises", "builders",
    "development", "contractor", "corp", "corporation", "company", "trading", "services",
}


def _ids(text: str) -> list[str]:
    return list(dict.fromkeys(m.upper() for m in CONTRACT_ID_RE.findall(text)))


def pin(question: str, ids: list[str]) -> str:
    """Attach the subject in the router's own phrasing (stripped again from the search text)."""
    if len(ids) == 1:
        return f"{question.rstrip()} for contract {ids[0]}"
    return f"{question.rstrip()} for contracts {', '.join(ids[:-1])} and {ids[-1]}"


def pin_note(ids: list[str]) -> str:
    """The user-visible line on a pinned answer, so a wrongly carried subject is seen at once."""
    which = f"contract {ids[0]}" if len(ids) == 1 else f"contracts {', '.join(ids)}"
    return (f"(Answering about {which}, taken from this conversation. "
            "Name a contract or place to change it.)")


class SubjectTracker:
    """Per-conversation subject: the current contract(s) plus every earlier subject, in order."""

    def __init__(self, rag):
        self._rag = rag                  # RagAnswerer: resolver + the manifest's real ids
        self._known = rag.contract_ids
        # Contractor-name words that identify exactly one contract ("rodekom", "jwu"), so "the
        # Rodekom contract" anchors a question. Shared or generic company words ("construction",
        # "supply", "general" as in "General Manager") never match.
        owners: dict[str, set[str]] = {}
        for r in rag.manifest:
            if r.get("contractor", "").lower().startswith("not stated"):
                continue
            for w in set(re.findall(r"[a-z]{3,}", r["contractor"].lower())) - _COMPANY_STOP:
                owners.setdefault(w, set()).add(r["contract_id"])
        self._contractor_word = {w: next(iter(c)) for w, c in owners.items() if len(c) == 1}
        self.current: list[str] = []
        self.past: list[list[str]] = []  # distinct subjects in first-seen order

    def _anchor(self, question: str) -> list[str]:
        """The contracts a question names itself: explicit ids (even invented ones -- the agent
        withholds those), else what the place/name resolver binds."""
        return _ids(question) or sorted(self._rag.resolve_contract_ids(question))

    def _earlier(self, question: str) -> list[str] | None:
        m = _ORDINAL_RE.search(question)
        if m:
            word = m.group(1).lower()
            if word in _ORDINAL_INDEX:
                i = _ORDINAL_INDEX[word]
                return self.past[i] if i < len(self.past) else None
            before = [s for s in self.past if s != self.current]
            return before[-1] if before else None
        return None

    def _contractor_named(self, question: str) -> list[str]:
        return sorted({self._contractor_word[w] for w in re.findall(r"[a-z]{3,}", question.lower())
                       if w in self._contractor_word})

    def resolve(self, question: str) -> tuple[str, list[str]]:
        """(question to ask, contracts pinned onto it -- [] when asked as-is)."""
        if _RESET_RE.search(question):
            self.current = []
            return question, []
        anchor = self._anchor(question)
        if anchor:
            if self.current and _COMPARE_RE.search(question) and _ANAPHOR_RE.search(question):
                union = list(dict.fromkeys(self.current + [a for a in anchor if a in self._known]))
                if len(union) > len(anchor):
                    return pin(question, union), union
            return question, []
        named = self._contractor_named(question)
        if named:
            return pin(question, named), named
        if (_LIST_CORPUS_RE.search(question) or _EACH_RE.search(question)
                or _ANALYTICAL_RE.search(question) or _CORPUS_RE.search(question)
                or is_rank_question(question) or is_aggregate_question(question)
                or (_WHICH_PLURAL_RE.search(question) and not _ANAPHOR_RE.search(question)
                    and not _ORDINAL_RE.search(question))):
            return question, []
        subject = self._earlier(question) or self.current
        return (pin(question, subject), subject) if subject else (question, [])

    def observe(self, asked: str, answer: str) -> None:
        """Move the subject after a VERIFIED answer to ``asked`` (the question as resolved)."""
        ids = [i for i in self._anchor(asked) if i in self._known]
        if not ids:
            ids = [i for i in _ids(answer) if i in self._known]
        if not ids:
            return
        self.current = ids
        if ids not in self.past:
            self.past.append(ids)
