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
                                           n.metadata["pdf_page"]))
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


# A display-only label for humanizing contract ids in answer text (the fastapi checker). It is
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


def _parse_extraction(raw: str) -> dict:
    """Pull the JSON object out of the LLM reply; every field falls back to 'not stated'."""
    record = {field: "not stated" for field in _EXTRACT_FIELDS}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            for field in _EXTRACT_FIELDS:
                value = str(parsed.get(field, "")).strip()
                if value:
                    record[field] = value
        except json.JSONDecodeError:
            pass
    return record


def build_manifest(nodes=None, persist: bool = True) -> list[dict]:
    """One record per contract: free metadata fields + one LLM extraction call each."""
    if nodes is None:
        from rag.index import build_nodes

        nodes = build_nodes()

    by_contract: dict[str, list] = {}
    for node in nodes:
        by_contract.setdefault(node.metadata["contract_id"], []).append(node)

    llm = get_llm()
    manifest: list[dict] = []
    for contract_id in sorted(by_contract):
        group = by_contract[contract_id]
        doc_types = sorted({n.metadata["doc_type"] for n in group})
        reply = llm.complete(_EXTRACT_PROMPT.format(text=_extraction_input(group)))
        record = {"contract_id": contract_id, "doc_types": doc_types, **_parse_extraction(str(reply))}
        record["short_name"] = _derive_short_name(record["contract_name"], record["location"])
        manifest.append(record)

    if persist:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
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


if __name__ == "__main__":
    manifest = build_manifest()
    print(f"built manifest for {len(manifest)} contracts -> {MANIFEST_PATH}\n")
    print(format_manifest(manifest))
