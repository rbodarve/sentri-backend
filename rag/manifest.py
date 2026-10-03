"""Step 9 of the RAG pipeline: a corpus-level manifest -- one structured record per contract.

Semantic top-k retrieval answers span-level questions ("who won 24CC0265?") but structurally
cannot answer *global* ones ("how many projects? list every location? add the amounts"): those
need the complete set of contracts in context, which retrieval never fetches (it returns the
few chunks nearest the query, not the whole corpus). This module materializes that complete
set as a small structured manifest (one row per contract), built once and persisted next to
the index; rag.generate injects it into every unfiltered answer context (a contract-filtered
pass gets only its own row), so no query-intent guessing is needed.

Two field tiers:
- free fields (contract_id, doc_types) are aggregated from the enriched nodes -- no model call.
- text fields (contract_name, location, implementing_office, contractor, amount) are read from
  each contract's highest-signal chunks by one LLM extraction call per contract. ``content`` is
  read verbatim; nothing about the OCR text is altered (the DB is the authoritative transcription).

``contractor`` can be *present in the OCR but not locally extractable*: on an NTP-only contract
(24CM0001) the firm is named only as the addressee/signatory -- there is no "winning bidder"
label -- and the small local model does not classify that as the contractor. Rather than a narrow
local parser that special-cases the one failing document shape, it is left ``"not stated"`` and
deferred to the stronger handoff model, the same division of labor as the TAIL checks in
rag.verify: the local pipeline records what it can reliably extract and stays honest about the
rest; the handoff model, which reads the full source, fills the gap.

``amount`` is absent from the OCR for 24CM0001 (NTP-only in the DB, carries no award figure);
it is rendered ``"not stated"`` so global sums stay honest instead of silently under-counting --
the same confident-but-incomplete failure the manifest exists to fix.
"""

from __future__ import annotations

import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from rag.config import PERSIST_DIR, get_llm

MANIFEST_PATH = Path(PERSIST_DIR) / "manifest.json"

# Fields the LLM extracts, in a fixed order. contract_id/doc_types are added for free (metadata).
_EXTRACT_FIELDS = ("contract_name", "location", "implementing_office", "contractor", "amount")

# Feed the extractor the identity-bearing docs first (name/location/office/contractor/amount all
# live on their front pages); ROA/ADS carry no new identity, so they come last.
_DOC_PRIORITY = {"NTP": 0, "CONTRACT_AGREEMENT": 1, "COA": 2, "NOA": 3, "ROA": 4, "ADS": 5}
_EXTRACT_CHAR_BUDGET = 3000  # ~750 tokens: keeps each extraction well inside the small num_ctx

_EXTRACT_PROMPT = (
    "You extract structured facts about a SINGLE DPWH infrastructure contract from OCR text.\n"
    "Return ONLY a JSON object with exactly these keys: "
    "contract_name, location, implementing_office, contractor, amount.\n"
    "Rules:\n"
    "- Use ONLY the text provided. If a field is not present, use the string \"not stated\".\n"
    "- contract_name: the value following 'Contract Name:'.\n"
    "- location: the project location (e.g. 'Bulakan, Bulacan'); it may follow "
    "'Location of the Contract:' or appear inside the contract name.\n"
    "- implementing_office: the DPWH District or Regional Engineering Office.\n"
    "- contractor: the winning-bidder COMPANY name (NOT the individual signatory or General Manager).\n"
    "- amount: the total contract cost as the peso figure (e.g. 'P69,479,428.64').\n"
    "Text:\n---\n{text}\n---\n"
    "JSON:"
)


def _extraction_input(nodes) -> str:
    """Identity-bearing text for one contract: prioritized docs, deduped lines, budget-capped."""
    ordered = sorted(nodes, key=lambda n: (_DOC_PRIORITY.get(n.metadata["doc_type"], 9),
                                           int(n.metadata["pdf_page"])))
    seen: set[str] = set()
    parts: list[str] = []
    total = 0
    for node in ordered:
        line = node.text.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        parts.append(line)
        total += len(line)
        if total >= _EXTRACT_CHAR_BUDGET:
            break
    return "\n".join(parts)


# A display-only short label for each contract (used by format_enumeration's listing). It is
# DERIVED deterministically from fields already extracted -- never invented: an LLM asked to
# "name the structure" hallucinated a "Dam" / "Coastal Barrier" for the two contracts whose name
# is generic, which is exactly the confident fabrication the rest of the pipeline guards against.
# So project type comes from keywords actually present in the name, the waterway only if the name
# states one, and the location verbatim. It never feeds retrieval or the verifier -- presentation only.
_PROJECT_TYPES = (
    ("slope protection", "Slope Protection Project"),
    ("flood control", "Flood Control Project"),
    ("flood mitigation", "Flood Mitigation Project"),
    ("flood management", "Flood Mitigation Project"),
    ("flood", "Flood Control Project"),
)
_WATERWAY_RE = re.compile(r"\b(along|at)\s+([A-ZÑ][\w.]*(?:\s+[A-ZÑ][\w.]*)*\s+(?:River|Creek))\b")


def _derive_short_name(contract_name: str, location: str) -> str:
    """A concise, grounded label for display substitution -- project type + any named waterway +
    location, all read from the record; no paraphrase, no invention."""
    low = contract_name.lower()
    ptype = next((label for key, label in _PROJECT_TYPES if key in low), "Project")
    m = _WATERWAY_RE.search(contract_name)
    waterway = f" {m.group(1)} {m.group(2)}" if m else ""
    loc = location if location and location != "not stated" else ""
    return f"{ptype}{waterway}" + (f", {loc}" if loc else "")


def _parse_extraction(raw: str) -> tuple[dict, bool]:
    """Pull the JSON object out of the LLM reply. Returns (record, parsed_ok): every field falls
    back to 'not stated', and parsed_ok is False when the reply carried no parseable JSON object.
    The caller surfaces that failure distinctly, because a parse failure is a MODEL/parse problem,
    NOT the absent-DATA ceiling coverage.py reports off the 'not stated' sentinel -- conflating the
    two would make coverage's DATA-vs-MODEL diagnosis untrustworthy. Non-greedy so a
    prose+JSON+prose reply captures just the object, not everything up to a trailing brace."""
    record = {field: "not stated" for field in _EXTRACT_FIELDS}
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return record, False
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return record, False
    for field in _EXTRACT_FIELDS:
        value = str(parsed.get(field, "")).strip()
        if value:
            record[field] = value
    return record, True


def build_manifest(nodes=None, persist: bool = True) -> list[dict]:
    """One record per contract: free metadata fields + one LLM extraction call each.

    Resumable: this step runs last and per-contract on Ollama, AFTER the expensive CPU embed, so
    an interruption on contract k must not discard the k-1 done. Records are persisted after each
    extraction, and a PARTIAL manifest.json (fewer contracts than the corpus) is resumed -- its
    completed records are reused and only the missing contracts re-extracted. A COMPLETE manifest
    is ignored, so a normal rebuild still re-extracts everything fresh (prompt/field changes take
    effect); `make clean` or deleting manifest.json also forces a clean rebuild."""
    if nodes is None:
        from rag.index import build_nodes

        nodes = build_nodes()

    by_contract: dict[str, list] = {}
    for node in nodes:
        by_contract.setdefault(node.metadata["contract_id"], []).append(node)

    # Resume only from a partial (interrupted) manifest; a complete one means a normal rebuild.
    done: dict[str, dict] = {}
    if MANIFEST_PATH.exists():
        prev = {r["contract_id"]: r for r in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))}
        if 0 < len(prev) < len(by_contract):
            done = prev
            print(f"resuming manifest: reusing {len(done)} record(s), extracting "
                  f"{len(by_contract) - len(done)} remaining", file=sys.stderr)

    llm = get_llm()
    manifest: list[dict] = []
    parse_failures: list[str] = []
    for contract_id in sorted(by_contract):
        if contract_id in done:
            manifest.append(done[contract_id])
            continue
        group = by_contract[contract_id]
        doc_types = sorted({n.metadata["doc_type"] for n in group})
        reply = llm.complete(_EXTRACT_PROMPT.format(text=_extraction_input(group)))
        fields, parsed_ok = _parse_extraction(str(reply))
        if not parsed_ok:
            parse_failures.append(contract_id)
            print(f"WARNING: manifest extraction for {contract_id} yielded no parseable JSON; "
                  f"all fields set to 'not stated' (PARSE FAILURE, not absent data)", file=sys.stderr)
        record = {"contract_id": contract_id, "doc_types": doc_types, **fields}
        record["short_name"] = _derive_short_name(record["contract_name"], record["location"])
        manifest.append(record)
        if persist:  # persist after each extraction so an interruption keeps completed work
            MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if persist:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if parse_failures:
        print(f"manifest: {len(parse_failures)}/{len(by_contract)} extraction(s) failed to parse: "
              f"{', '.join(parse_failures)}", file=sys.stderr)
    return manifest


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} not found; run `python -m rag.manifest` (or `make build`).")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def format_manifest(manifest: list[dict]) -> str:
    """Render the manifest as a compact, complete text table for context injection."""
    lines = [
        f"CORPUS MANIFEST -- the database contains exactly {len(manifest)} contracts/projects "
        "(this is the complete list; use it for any question about how many, which, or all projects):"
    ]
    for i, r in enumerate(manifest, 1):
        lines.append(f"{i}. {format_manifest_row(r)}")
    return "\n".join(lines)


def format_manifest_row(r: dict) -> str:
    """One contract's manifest record, as it reads in the injected table."""
    return (f"Contract {r['contract_id']}: {r['contract_name']} | "
            f"location: {r['location']} | office: {r['implementing_office']} | "
            f"contractor: {r['contractor']} | amount: {r['amount']} | "
            f"documents: {', '.join(r['doc_types'])}")


def format_enumeration(manifest: list[dict]) -> str:
    """Deterministic answer to a corpus-wide 'list all projects/locations' question: one row per
    contract, read straight from the manifest -- no LLM, so the answer is complete and identical
    every run (the failure the retrieval/summarization route produced). Surfaces the real
    contract_id, which routing knows but generation never showed."""
    lines = [f"The database contains {len(manifest)} contracts/projects:"]
    for i, r in enumerate(manifest, 1):
        lines.append(f"{i}. Contract {r['contract_id']} — {_display_name(r)} — "
                     f"Location: {r.get('location') or 'not stated'}")
    return "\n".join(lines)


def _display_name(r: dict) -> str:
    """A contract's name as the deterministic manifest answers show it."""
    return r.get("short_name") or r.get("contract_name") or "not stated"


def parse_amount(text: str | None) -> Decimal | None:
    """A manifest amount ('Php140,274,481.48', 'P 96,489,983.04', 'PHP140274481.48') as a Decimal;
    None for anything that is not exactly one peso figure ('not stated', a figure with trailing
    text), so an odd string is never guessed into a figure and mis-ranked."""
    m = re.fullmatch(r"\s*(?:Php|P|₱)?\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*", text or "", re.I)
    return Decimal(m.group(1).replace(",", "")) if m else None


def format_ranking(manifest: list[dict], descending: bool, top_only: bool) -> str:
    """Deterministic answer to a corpus-wide 'highest / rank by amount' question: the verbatim
    manifest amounts compared in Decimal -- no LLM orders numbers. A contract without a parseable
    amount is named as not ranked, with its verbatim manifest value, never dropped silently."""
    stated, unstated = _stated_amounts(manifest)
    ranked = [r for _, r in sorted(stated, key=lambda ar: ar[0], reverse=descending)]
    if top_only and ranked:
        word = "highest" if descending else "lowest"
        lines = [f"Of the {len(ranked)} contracts with a stated amount, the {word} is "
                 f"{_amount_row(ranked[0])}."]
    else:
        order = "largest to smallest" if descending else "smallest to largest"
        lines = [f"The {len(ranked)} contracts with a stated amount, {order}:"]
        lines += [f"{i}. {_amount_row(r)}" for i, r in enumerate(ranked, 1)]
    if unstated:
        lines.append(f"Not ranked {unstated}")
    return "\n".join(lines)


def format_aggregate(manifest: list[dict], ops: tuple[str, ...]) -> str:
    """Deterministic answer to a corpus-wide 'total / average amount' question (ops: "total"
    and/or "average", one line each): the verbatim manifest amounts summed in Decimal -- no LLM
    adds numbers. Every addend is listed; a contract without a parseable amount is named as not
    included."""
    stated, unstated = _stated_amounts(manifest)
    if not stated:
        return f"No contract has a stated amount. Not included {unstated}"
    total = sum((a for a, _ in stated), Decimal(0))
    values = {"total": total,
              "average": (total / len(stated)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)}
    lines = [f"The {op} amount of the {len(stated)} contracts with a stated amount is "
             f"Php{values[op]:,.2f}." for op in ops]
    lines.append("Stated amounts:")
    lines += [f"- {_amount_row(r)}" for _, r in stated]
    if unstated:
        lines.append(f"Not included {unstated}")
    return "\n".join(lines)


def _stated_amounts(manifest: list[dict]) -> tuple[list[tuple[Decimal, dict]], str]:
    """(every (Decimal amount, row) the manifest states, in manifest order; the tail naming each
    contract without one with its verbatim value, '' if none) -- shared by rank and aggregate."""
    parsed = [(parse_amount(r.get("amount")), r) for r in manifest]
    missing = [f"Contract {r['contract_id']} ({r.get('amount') or 'not stated'})"
               for a, r in parsed if a is None]
    tail = f"(no single peso amount in the manifest): {', '.join(missing)}." if missing else ""
    return [(a, r) for a, r in parsed if a is not None], tail


def _amount_row(r: dict) -> str:
    """One contract's line in the deterministic amount answers (rank, aggregate)."""
    return f"Contract {r['contract_id']} — {_display_name(r)} — Amount: {r['amount']}"


def _self_check() -> None:
    """Model-free pins for the rank route's amount handling (`python -m rag.manifest --check`)."""
    assert parse_amount("PHP140274481.48") == Decimal("140274481.48"), "no-comma amount not parsed"
    vat = "Php69,479,428.64 (VAT inclusive)"
    assert parse_amount(vat) is None, "trailing text must not be guessed into a figure"
    text = format_ranking([{"contract_id": "X1", "contract_name": "n", "amount": vat}], True, False)
    assert "not stated" not in text.split("Not ranked")[-1].split(":")[0], \
        f"an unparsed amount is labelled 'not stated': {text!r}"
    assert vat in text.split("Not ranked")[-1], f"the not-ranked line hides the verbatim value: {text!r}"
    print("OK: parse_amount / format_ranking pins")
    rows = [{"contract_id": "X1", "contract_name": "a", "amount": "P1,000.00"},
            {"contract_id": "X2", "contract_name": "b", "amount": "Php2,000.50"},
            {"contract_id": "X3", "contract_name": "c", "amount": "not stated"}]
    text = format_aggregate(rows, ("total",))
    assert "3,000.50" in text, f"total is not the Decimal sum: {text!r}"
    assert "of the 2 contracts with a stated amount" in text, f"total hides its n: {text!r}"
    assert "P1,000.00" in text and "Php2,000.50" in text, f"total hides an addend: {text!r}"
    assert "Contract X3 (not stated)" in text and "(no single peso amount in the manifest)" in text, \
        f"total hides an excluded contract: {text!r}"
    # mean 0.025: ROUND_HALF_UP -> 0.03 (ROUND_HALF_EVEN would give 0.02); no addend contains "0.03"
    text = format_aggregate([{"contract_id": "Y1", "amount": "P0.01"},
                             {"contract_id": "Y2", "amount": "P0.04"}], ("average",))
    assert "0.03" in text and "0.025" not in text, f"average not rounded ROUND_HALF_UP: {text!r}"
    text = format_aggregate(rows, ("total", "average"))  # "the total and the average": both lines
    assert "total amount" in text and "3,000.50" in text and "average amount" in text \
        and "1,500.25" in text and text.index("3,000.50") < text.index("1,500.25"), \
        f"a total-and-average question drops a value: {text!r}"
    print("OK: format_aggregate pins")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _self_check()
        sys.exit()
    manifest = build_manifest()
    print(f"built manifest for {len(manifest)} contracts -> {MANIFEST_PATH}\n")
    print(format_manifest(manifest))
