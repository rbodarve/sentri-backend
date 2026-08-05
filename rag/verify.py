"""Step 10 of the RAG pipeline: a final grounding check on the generated answer.

Runs after generation, before the answer reaches the terminal. It does NOT re-ask the
model or do any arithmetic; it checks the answer's *relational claims* against the corpus
manifest (rag.manifest), which is the authoritative structured record of every contract.

Either tier withholds the answer: on any failure the caller drops the generated text and shows
a generic "the model appears to be hallucinating" notice in its place, with the specific reason
appended for diagnosis. The two tiers differ only in confidence, not in action:

- BLOCK (DERIVED, zero false positives): the answer names a contract_id that does not exist
  in the corpus. An invented contract is unambiguously hallucinated.
- FLAG (EXTRACTED): an attribution unit -- a blank-line-separated block (so a "<location>:"
  heading and the contract listed under it are read together) or a single sentence within it --
  binds one contract to *another* contract's location or contractor (e.g. "24A00153 ... in
  Bulacan", where Bulacan belongs to 24CC0265, or a "Projects in Olongapo City:" heading over a
  contract that is really in Bulacan). The manifest value is itself LLM-extracted, so this tier
  can carry a false positive; per project policy the answer is withheld anyway, accepting a rare
  suppressed-correct-answer over ever displaying a mis-bound one.

Only the CORE relations that the manifest holds (contract existence, location, contractor) are
checked; open-ended (TAIL) claims -- materials, clauses, counts -- are left unverified here and
are the job of a stronger entailment check on the handoff hardware. No values are ever summed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.enrich import CONTRACT_ID_RE

# Generic place / boilerplate words that carry no contract identity: excluded from the
# location token index so they never trigger a spurious mis-binding flag. "luzon" is the
# island (and a barangay name), far too common to attribute to one contract.
_GEO_STOP = {
    "barangay", "brgy", "city", "phase", "north", "south", "east", "west", "norte", "sur",
    "luzon", "street", "road", "river", "creek", "along", "construction", "rehabilitation",
    "flood", "mitigation", "control", "structure", "protection", "slope", "reinforced",
    "concrete", "facilities", "within", "major", "basins", "principal", "rivers", "project",
}


def _sentences(text: str) -> list[str]:
    """Split an answer into sentences/lines; attribution is checked per sentence."""
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _attribution_units(text: str) -> list[str]:
    """Units the mis-binding check reads: each blank-line-separated block (so a location
    heading and the contract listed beneath it are seen together) plus every sentence within.
    Checking both scopes catches a foreign token whether it shares a line with the contract id
    or sits on a nearby heading, while the per-sentence scope still isolates individual
    contracts inside a dense block. Duplicate flags across scopes are de-duplicated in check()."""
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    units = list(blocks)
    for b in blocks:
        units.extend(_sentences(b))
    return units


def _location_index(manifest: list[dict]) -> dict[str, set[str]]:
    """Distinctive location token (lowercased) -> set of contract_ids it belongs to."""
    idx: dict[str, set[str]] = {}
    for r in manifest:
        for tok in re.findall(r"[a-zñ]+", (r.get("location") or "").lower()):
            if len(tok) < 4 or tok in _GEO_STOP:
                continue
            idx.setdefault(tok, set()).add(r["contract_id"])
    return idx


def _contractor_index(manifest: list[dict]) -> dict[str, set[str]]:
    """Distinctive contractor head-token (e.g. 'amethyst') -> set of contract_ids."""
    idx: dict[str, set[str]] = {}
    for r in manifest:
        m = re.search(r"[a-zñ]{3,}", (r.get("contractor") or "").lower())
        if not m or m.group(0) == "not":  # skip "not stated"
            continue
        idx.setdefault(m.group(0), set()).add(r["contract_id"])
    return idx


@dataclass
class Report:
    """Outcome of the pre-send check. ``blocked`` withholds the answer; ``flags`` warn."""

    blocks: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the answer cleared the grounding check; any block or flag withholds it."""
        return not self.blocks and not self.flags


class Verifier:
    """Checks an answer's contract bindings against the manifest (the structured oracle)."""

    def __init__(self, manifest: list[dict]):
        self._contracts = {r["contract_id"] for r in manifest}
        self._loc = _location_index(manifest)
        self._con = _contractor_index(manifest)
        self._loc_value = {r["contract_id"]: r.get("location", "") for r in manifest}
        self._con_value = {r["contract_id"]: r.get("contractor", "") for r in manifest}

    def check(self, answer: str) -> Report:
        report = Report()
        seen: set[str] = set()  # flag messages already recorded, so block/sentence scopes don't double-report

        # BLOCK: any contract_id in the answer that the corpus does not contain is invented.
        mentioned = {m.upper() for m in CONTRACT_ID_RE.findall(answer)}
        for unknown in sorted(mentioned - self._contracts):
            report.blocks.append(f"names contract {unknown}, which is not in the corpus")

        # FLAG: an attribution unit about exactly one contract that cites another contract's binding.
        for unit in _attribution_units(answer):
            ids = {m.upper() for m in CONTRACT_ID_RE.findall(unit)} & self._contracts
            if len(ids) != 1:
                continue  # 0 or >1 contracts -> attribution is ambiguous, skip
            (cid,) = tuple(ids)
            low = unit.lower()
            self._flag_foreign(report, seen, low, cid, self._loc, self._loc_value,
                               "location", cid_desc=" is in")
            self._flag_foreign(report, seen, low, cid, self._con, self._con_value,
                               "contractor", cid_desc="'s contractor is")
        return report

    @staticmethod
    def _flag_foreign(report, seen, low, cid, index, values, label, cid_desc):
        for token, owners in index.items():
            if cid in owners:
                continue
            if re.search(rf"\b{re.escape(token)}\b", low):
                msg = (
                    f"answer ties {cid} to {label} '{token}' (belongs to "
                    f"{', '.join(sorted(owners))}); manifest says {cid}{cid_desc} "
                    f"'{values[cid]}'"
                )
                if msg not in seen:
                    seen.add(msg)
                    report.flags.append(msg)


def format_report(report: Report) -> str:
    """Render the check outcome as a compact banner for the terminal. Any grounding failure --
    an invented contract or a mis-bound location/contractor -- yields a generic "the model
    appears to be hallucinating" notice that stands in for the withheld answer; the specific
    reason(s) follow so the failure can be diagnosed."""
    if report.ok:
        return ("✓ grounding check passed -- but this ONLY confirms contract IDs, locations & "
                "contractors against the manifest.\n"
                "  ⚠ NOT verified here: amounts/figures, district engineers, offices, dates and "
                "every other detail -- treat those as unconfirmed and check the source documents.")
    lines = ["⛔ verification: the model appears to be hallucinating -- answer withheld "
             "(failed the manifest grounding check):"]
    lines += [f"    - {r}" for r in report.blocks + report.flags]
    return "\n".join(lines)
