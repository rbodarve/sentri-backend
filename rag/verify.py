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
import sys
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

# All-caps multi-word sequences (no line-break crossing) that may be person names.
_ALLCAPS_NAME_RE = re.compile(r'\b([A-Z][A-Z.]+(?:[^\S\n]+[A-Z.]+){1,})\b')

# Words that appear in all-caps legal/government phrases but are NOT person-name tokens.
_NAME_STOP_WORDS = frozenset({
    # common English words
    "and", "the", "for", "but", "nor", "yet", "not", "its", "our", "all", "any", "few",
    "now", "this", "that", "with", "from", "upon", "into", "over", "under", "been", "have",
    "done", "made", "make", "said", "each", "both", "when", "then", "than", "thus", "also",
    "once", "only", "same", "some", "such", "here", "there", "where", "which", "what", "whom",
    "will", "shall", "may", "can", "must", "duly", "until", "unto", "very", "said", "fore",
    # legal/document boilerplate
    "agreement", "witnesseth", "follows", "contract", "pursuant", "herein", "hereby",
    "thereof", "thereto", "whereas", "sealed", "delivered", "executed", "aforesaid",
    "foregoing", "witnessed", "above", "notary", "public", "roll", "commission", "valid",
    # government/institutional terms
    "department", "works", "highways", "republic", "philippines", "government", "general",
    "corporation", "enterprise", "construction", "supply", "infrastructure", "district",
    "engineering", "office", "region", "provincial", "authority", "institute", "foundation",
    "association", "center", "bureau", "notice", "award", "certificate", "acceptance",
    "completion", "scope", "terms", "conditions", "article", "section", "clause",
    "office", "national", "regional", "implementing",
    # role/rank labels printed in all caps beside signatures ("PROCURING ENTITY", "BAC MEMBER",
    # "GEN MANAGER", "CESO III"), and the notary's city -- not person names
    "procuring", "entity", "member", "manager", "ceso", "city",
})


def _extract_sig_names(text: str) -> set[str]:
    """Extract person-name-like tokens from an all-caps signature chunk.

    Matches all-caps multi-word sequences on a single line, requires ≥2 words
    with ≥3 alpha chars that are not legal/government stop-words. Used to build
    the corpus-wide signatory map for the grounding check."""
    names: set[str] = set()
    for m in _ALLCAPS_NAME_RE.finditer(text):
        candidate = m.group(0).strip()
        qualified = [
            w for w in candidate.split()
            if len(re.sub(r"[^A-Za-z]", "", w)) >= 3 and w.lower() not in _NAME_STOP_WORDS
        ]
        if len(qualified) >= 2:
            names.add(candidate.lower())
    return names


def _sentences(text: str) -> list[str]:
    """Split an answer into sentences/lines; attribution is checked per sentence.
    Requires ≥2 letter chars before the sentence-ending punctuation so single-letter
    initials like 'D.' in 'Brandy D. Abeya' are not treated as sentence boundaries -- or a
    digit, so a sentence ending in a contract id ('...contract 24BG0272. The ...') still ends
    there instead of merging into the next sentence's attribution scope. Decimals ('428.64')
    are unaffected: the split needs whitespace after the punctuation."""
    return [s for s in re.split(r"(?:(?<=[A-Za-z]{2}[.!?])|(?<=\d[.!?]))\s+|\n+", text)
            if s.strip()]


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

    def __init__(self, manifest: list[dict],
                 person_names: dict[str, set[str]] | None = None):
        self._contracts = {r["contract_id"] for r in manifest}
        self._loc = _location_index(manifest)
        self._con = _contractor_index(manifest)
        self._loc_value = {r["contract_id"]: r.get("location", "") for r in manifest}
        self._con_value = {r["contract_id"]: r.get("contractor", "") for r in manifest}
        # Inverse map: person name (lowercase) → set of contract_ids whose signature
        # corpus contains that name. Built from signature chunks at startup; empty when
        # the caller does not provide the map (e.g. tests that construct Verifier directly).
        # None (not provided) silently disables the whole person-fabrication tier, so signal it;
        # a caller that means "no person checking" passes an explicit {} to stay quiet.
        if person_names is None:
            print("Verifier: no person_names map -- the person-fabrication grounding tier is "
                  "DISABLED (pass {} explicitly to silence this).", file=sys.stderr)
        self._person_owners: dict[str, set[str]] = {}
        for cid, names in (person_names or {}).items():
            for name in names:
                self._person_owners.setdefault(name, set()).add(cid)

    def check(self, answer: str, *, check_bindings: bool = True,
              contract_id: str | None = None) -> Report:
        report = Report()
        seen: set[str] = set()  # flag messages already recorded, so block/sentence scopes don't double-report

        # BLOCK: any contract_id in the answer that the corpus does not contain is invented.
        mentioned = {m.upper() for m in CONTRACT_ID_RE.findall(answer)}
        for unknown in sorted(mentioned - self._contracts):
            report.blocks.append(f"names contract {unknown}, which is not in the corpus")

        # The mis-binding FLAG heuristic assumes each attribution unit is about exactly ONE
        # contract. A cross-corpus analytical answer (rag.agent's "analytical" route) legitimately
        # enumerates many contracts with their locations/contractors together, which the heuristic
        # mis-reads as a mis-binding and would false-withhold. Such callers pass
        # check_bindings=False to verify with only the DERIVED, zero-false-positive BLOCK tier
        # above; single-contract callers keep the full check (the default).
        if not check_bindings:
            return report

        # FLAG: an attribution unit about exactly one contract that cites another contract's binding.
        for unit in _attribution_units(answer):
            ids = {m.upper() for m in CONTRACT_ID_RE.findall(unit)} & self._contracts
            if len(ids) == 1:
                (cid,) = tuple(ids)
            elif len(ids) == 0 and contract_id in self._contracts:
                # The answer names no contract, but the caller routed this (sub)answer to a known
                # contract -- bind the unit to that scope so a borrowed contractor/location from
                # another contract is still caught (e.g. a scoped answer that omits the id).
                cid = contract_id
            else:
                continue  # >1 contracts, or 0 with no routed scope -> ambiguous, skip
            low = unit.lower()
            self._flag_foreign(report, seen, low, cid, self._loc, self._loc_value,
                               "location", cid_desc=" is in")
            self._flag_foreign(report, seen, low, cid, self._con, self._con_value,
                               "contractor", cid_desc="'s contractor is")

        # FLAG: person name grounding check.
        # A person found in the answer alongside a contract's location/contractor token must
        # appear in that contract's signature corpus. Catches the LLM using manifest metadata
        # (contractor/location fields) to fabricate person-contract associations not in any
        # retrieved chunk — e.g. claiming a signatory from contract A signed contract B.
        if self._person_owners:
            for sentence in _sentences(answer):
                slow = sentence.lower()
                # Contracts this sentence implies via distinctive location/contractor tokens.
                sentence_contracts: set[str] = set()
                for token, owners in self._loc.items():
                    if re.search(rf"\b{re.escape(token)}\b", slow):
                        sentence_contracts |= owners
                for token, owners in self._con.items():
                    if re.search(rf"\b{re.escape(token)}\b", slow):
                        sentence_contracts |= owners
                if not sentence_contracts:
                    continue
                for name, known_cids in self._person_owners.items():
                    if not re.search(rf"\b{re.escape(name)}\b", slow):
                        continue
                    for cid in sentence_contracts:
                        if cid not in known_cids:
                            msg = (
                                f"person '{name}' appears in answer with contract {cid}'s "
                                f"location/contractor but has no record in {cid}'s signature "
                                f"corpus (known to: {', '.join(sorted(known_cids))})"
                            )
                            if msg not in seen:
                                seen.add(msg)
                                report.flags.append(msg)

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


if __name__ == "__main__":
    # Pin the two independent signatory parsers together (Partition-2 F1): the summary BUILDER
    # (rag.index._person_from_chunk, via _signatory_summaries) chooses which names appear in each
    # per-contract signatory_summary; the answer POLICER here (_extract_sig_names, fed by
    # generate._build_person_names) decides which names an answer may legitimately contain. They
    # share no code. If the policer cannot recover a name the builder emitted, the verifier can
    # reject a correct signatory answer -- so assert every listed name is recoverable, over the
    # real database/ signature chunks.
    from rag.enrich import enrich_nodes
    from rag.index import _signatory_summaries
    from rag.loader import load_nodes

    nodes = load_nodes()
    enrich_nodes(nodes)
    summaries = _signatory_summaries(nodes)
    bullets = 0
    for node in summaries:
        cid = node.metadata["contract_id"]
        for line in node.text.splitlines():
            if not line.startswith("- "):
                continue
            bullets += 1
            assert _extract_sig_names(line), (
                f"signatory parser drift ({cid}): the verifier cannot recover a name from "
                f"summary line {line!r} that the builder emitted"
            )
    print(f"OK: {len(summaries)} signatory summaries, {bullets} listed names all recoverable "
          "by the verifier's parser (index._person_from_chunk <-> verify._extract_sig_names)")
