"""Step 2 of the RAG pipeline: enrich Step-1 nodes for cross-contract disambiguation.

~90% of the OCR chunks do not name their own contract (a signature is just a name, an
amount is just a number), and the 6 contracts share the same document templates. So
identical content collides across contracts at retrieval time.

This step derives ``contract_id`` and ``doc_type`` from each node's ``pdf_source`` and
stores them in metadata, where they power the retrieval-time contract filter (see
rag.evaluate) and LLM citations. ``node.text`` (the authoritative OCR content) is left
completely untouched.

The embedding text is deliberately kept as *pure content*: injecting contract_id/doc_type
into the embedded vector was measured to homogenize vectors and let short boilerplate
chunks out-rank real answers for any contract-mentioning query. Disambiguation is instead
done structurally by the metadata filter. Per-node visibility:

- embedding text  = content only            (all metadata excluded from the vector)
- LLM text        = everything except ``coordinate``   (kept for citations)
"""

from __future__ import annotations

import re

from llama_index.core.schema import TextNode

# Contract IDs look like 24CC0265, 24AJ0052, or 24A00153 (24 + 1-2 letters + 4-5 digits).
CONTRACT_ID_RE = re.compile(r"24[A-Za-z]{1,2}\d{4,5}")

# When retrieval is scoped by a contract filter, the contract id in the query text is
# redundant and biases ranking toward header chunks. Strip it so the semantic query is
# intent-only (parse the id for the filter, search on the rest) -- used by rag.generate.
_CONTRACT_PHRASE_RE = re.compile(r"\s*(of\s+|for\s+)?contract\s+24[A-Za-z]{1,2}\d{4,5}", re.I)

# Ordered (filename keyword -> canonical doc type); first match wins. Covers every
# pdf_source present in database/. Full-form keys are load-bearing: e.g. "notice_of_award"
# contains no "noa" substring, so it needs its own entry.
DOC_TYPE_KEYWORDS = [
    ("contract_agreement", "CONTRACT_AGREEMENT"),
    ("roa", "ROA"),
    ("coa", "COA"),
    ("notice_to_proceed", "NTP"),
    ("ntp", "NTP"),
    ("notice_of_award", "NOA"),
    ("noa", "NOA"),
    ("ads", "ADS"),
]

# Embed pure content: every metadata key is hidden from the vector (contract scoping is
# done by the retrieval-time metadata filter, not by biasing the embedding).
_EXCLUDED_EMBED = ["contract_id", "doc_type", "pdf_source", "pdf_page", "coordinate", "category"]
# Never useful to the answer-writing LLM (kept only for Step-3 adjacency).
_EXCLUDED_LLM = ["coordinate"]


def contract_id_from_source(pdf_source: str) -> str:
    match = CONTRACT_ID_RE.search(pdf_source)
    return match.group(0).upper() if match else "UNKNOWN"


def doc_type_from_source(pdf_source: str) -> str:
    lowered = pdf_source.lower()
    for keyword, doc_type in DOC_TYPE_KEYWORDS:
        if keyword in lowered:
            return doc_type
    return "UNKNOWN"


def strip_contract_phrase(query: str) -> str:
    """Remove a 'contract <id>' phrase from a query, leaving the intent-only text."""
    return _CONTRACT_PHRASE_RE.sub("", query).strip()


def enrich_nodes(nodes: list[TextNode]) -> list[TextNode]:
    """Add contract_id/doc_type metadata and set embed/LLM visibility. Mutates in place."""
    for node in nodes:
        pdf_source = node.metadata["pdf_source"]
        node.metadata["contract_id"] = contract_id_from_source(pdf_source)
        node.metadata["doc_type"] = doc_type_from_source(pdf_source)
        node.excluded_embed_metadata_keys = list(_EXCLUDED_EMBED)
        node.excluded_llm_metadata_keys = list(_EXCLUDED_LLM)
    return nodes


if __name__ == "__main__":
    from collections import Counter

    from llama_index.core.schema import MetadataMode

    from rag.loader import load_nodes

    nodes = load_nodes()
    original_text = {n.id_: n.text for n in nodes}
    enrich_nodes(nodes)

    # 1) every node is classified (no gaps in derivation)
    unknown = [
        n.id_ for n in nodes
        if n.metadata["contract_id"] == "UNKNOWN" or n.metadata["doc_type"] == "UNKNOWN"
    ]
    assert not unknown, f"unclassified nodes: {unknown[:5]}"

    # 2) OCR content is verbatim -- enrichment must not alter node.text
    assert all(n.text == original_text[n.id_] for n in nodes), "enrichment altered OCR content"

    pairs = Counter((n.metadata["contract_id"], n.metadata["doc_type"]) for n in nodes)
    contracts = {c for c, _ in pairs}
    print(f"enriched {len(nodes)} nodes | {len(contracts)} contracts | {len(pairs)} contract/doc pairs")
    for (contract_id, doc_type), count in sorted(pairs.items()):
        print(f"  {contract_id:<10} {doc_type:<20} {count}")

    # 3) embedding text is pure content; contract identity lives in metadata (for filtering)
    sig = next(n for n in nodes if n.metadata["category"] == "signature")
    embed_text = sig.get_content(metadata_mode=MetadataMode.EMBED)
    print("\n--- sample signature node ---")
    print("raw .text (verbatim):", repr(sig.text))
    print("contract_id (metadata):", sig.metadata["contract_id"])
    print("EMBED model sees:\n" + embed_text)
    assert embed_text == sig.text, "embed text should be pure content"
    assert sig.metadata["contract_id"] not in embed_text, "contract_id leaked into embed vector"
