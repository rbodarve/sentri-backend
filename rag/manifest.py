"""Step 9 of the RAG pipeline: a corpus-level manifest -- one structured record per contract.

Semantic top-k retrieval answers span-level questions ("who won 24CC0265?") but structurally
cannot answer *global* ones ("how many projects? list every location? add the amounts"): those
need the complete set of contracts in context, which retrieval never fetches (it returns the
few chunks nearest the query, not the whole corpus). This module materializes that complete
set as a small structured manifest (one row per contract), built once and persisted next to
the index; rag.generate always injects it into the answer context, so no query-intent guessing
is needed.

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
        lines.append(
            f"{i}. Contract {r['contract_id']}: {r['contract_name']} | "
            f"location: {r['location']} | office: {r['implementing_office']} | "
            f"contractor: {r['contractor']} | amount: {r['amount']} | "
            f"documents: {', '.join(r['doc_types'])}"
        )
    return "\n".join(lines)


def format_enumeration(manifest: list[dict]) -> str:
    """Deterministic answer to a corpus-wide 'list all projects/locations' question: one row per
    contract, read straight from the manifest -- no LLM, so the answer is complete and identical
    every run (the failure the retrieval/summarization route produced). Surfaces the real
    contract_id, which routing knows but generation never showed."""
    lines = [f"The database contains {len(manifest)} contracts/projects:"]
    for i, r in enumerate(manifest, 1):
        name = r.get("short_name") or r.get("contract_name") or "not stated"
        lines.append(f"{i}. Contract {r['contract_id']} — {name} — "
                     f"Location: {r.get('location') or 'not stated'}")
    return "\n".join(lines)


if __name__ == "__main__":
    manifest = build_manifest()
    print(f"built manifest for {len(manifest)} contracts -> {MANIFEST_PATH}\n")
    print(format_manifest(manifest))
