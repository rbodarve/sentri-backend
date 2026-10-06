"""Step 4 of the RAG pipeline: embed the enriched nodes and build a persistent index.

Composes Steps 1-3 (load -> enrich -> link) into a node list, embeds them with the
configured model (see rag.config), and stores them in an in-memory SimpleVectorStore
persisted to disk. Embedding is the only step that touches a model; it runs once and is
reloaded from disk thereafter, so the cost is paid a single time.
"""

from __future__ import annotations

import re
from collections import defaultdict

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import TextNode

from rag.config import PERSIST_DIR, get_embed_model
from rag.enrich import _EXCLUDED_LLM, enrich_nodes
from rag.loader import NOTARY_PREFIX, load_nodes
from rag.relationships import link_reading_order, stitch_split_table_headers

# Lines that are context labels, not person names — skip them when parsing chunks.
_CONTEXT_LINE_RE = re.compile(
    r"^(?:represented\s+by|witnessed\s+by|approved\s+by|very\s+truly\s+yours"
    r"|i\s+acknowledge|procuring\s+entity|contractor\s*:|name\s+and\s+sign"
    r"|republic\s+of\s+the|department\s+of|office\s+of\s+the|by\s*:"
    r"|notary\s+public\s*$|position\s*:|date\s+approved|agb\s+legal"
    r"|#\d+|commission\s+no|roll\s+of\s+att|ibp\s+official|professional\s+tax"
    r"|mcle\s+compliance|valid\s+until)",
    re.I,
)


def _person_from_chunk(raw: str) -> tuple[str, str] | None:
    """Return (name, title) for the first recognisable person in a signature chunk,
    or None when the chunk contains only context labels or company boilerplate."""
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        if _CONTEXT_LINE_RE.match(line):
            continue
        # A person-name line: has at least one uppercase letter, a reasonable length,
        # does not end with a colon, and is not a pure number or date token.
        if re.search(r"[A-Z]", line) and 4 <= len(line) <= 70 and not line.endswith(":"):
            name = line
            title = ""
            for j in range(i + 1, min(i + 3, len(lines))):
                nxt = lines[j]
                if not _CONTEXT_LINE_RE.match(nxt) and nxt != name and len(nxt) < 80:
                    title = nxt
                    break
            return name, title
    return None


def _signatory_summaries(nodes: list) -> list[TextNode]:
    """One synthetic node per contract listing each unique signatory exactly once.

    Raw signature chunks repeat the same person under different context labels
    ("Represented by:", "Witnessed by:", …). Aggregating verbatim produces a noisy,
    repetitive block the small LLM can't reliably enumerate. Parsing out unique
    (name, title) pairs and emitting a compact clean list lets the model list every
    signatory without losing track of duplicates."""
    # Collect unique persons per contract, keyed by normalised name.
    # Also track all signature node IDs per contract so the serve layer can expand
    # the summary back to real chunks (with bboxes) for the client source panel.
    by_contract: dict[str, dict[str, str]] = defaultdict(dict)
    source_ids: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        if node.metadata.get("category") != "signature":
            continue
        cid = node.metadata.get("contract_id", "").upper()
        if not cid:
            continue
        source_ids[cid].append(node.node_id)
        raw = node.text
        if raw.startswith("Signatory: "):
            raw = raw[len("Signatory: "):]
        elif raw.startswith(NOTARY_PREFIX):
            raw = raw[len(NOTARY_PREFIX):]
        result = _person_from_chunk(raw.strip())
        if not result:
            continue
        name, title = result
        # Normalise to deduplicate across repeated chunks (e.g. "RONNEL M TAN"
        # vs "RONNEL M. TAN" — strip dots/spaces and upper for the key).
        key = re.sub(r"[\.\s,]+", " ", name).strip().upper()
        if key not in by_contract[cid]:
            by_contract[cid][key] = f"{name}, {title}" if title else name

    summaries = []
    for cid, persons in sorted(by_contract.items()):
        entries = "\n".join(f"- {entry}" for entry in persons.values())
        text = f"All signatories for contract {cid}:\n{entries}"
        summaries.append(TextNode(
            # Deterministic id (not a fresh UUID per build) so these synthetic nodes are
            # byte-reproducible across rebuilds; reuses the same key as their pdf_source.
            id_=f"{cid.lower()}_signatories_summary",
            text=text,
            metadata={
                "category": "signature_summary",
                "contract_id": cid,
                "doc_type": "SIGNATORIES",
                "pdf_source": f"{cid.lower()}_signatories_summary",
                "pdf_page": "0",
                "coordinate": None,
                "source_node_ids": source_ids[cid],
            },
            # Built after enrich_nodes, so set here: hide the UUID list from the LLM (it copied
            # it into answers). Embed text is left as it was.
            excluded_llm_metadata_keys=_EXCLUDED_LLM + ["source_node_ids"],
        ))
    return summaries


def build_nodes():
    """Run Steps 1-3, add signatory summaries, and return the fully prepared node list."""
    nodes = load_nodes()
    enrich_nodes(nodes)
    link_reading_order(nodes)
    stitch_split_table_headers(nodes)
    nodes.extend(_signatory_summaries(nodes))
    return nodes


def build_index(persist_dir: str = PERSIST_DIR) -> VectorStoreIndex:
    """Embed the prepared nodes and persist the index to ``persist_dir``."""
    Settings.embed_model = get_embed_model()
    nodes = build_nodes()
    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(persist_dir)
    return index


def load_index(persist_dir: str = PERSIST_DIR) -> VectorStoreIndex:
    """Reload a persisted index. The same embed model must be set for queries."""
    Settings.embed_model = get_embed_model()
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return load_index_from_storage(storage_context)


if __name__ == "__main__":
    from rag.config import EMBED_MODEL, EMBED_PROVIDER

    print(f"embedding with {EMBED_PROVIDER}:{EMBED_MODEL} -> {PERSIST_DIR}/")
    build_index()

    # reload from disk and smoke-test retrieval (proves persistence + querying work)
    index = load_index()
    hits = index.as_retriever(similarity_top_k=3).retrieve(
        "flood control structure along Tabang River"
    )
    print(f"\nreloaded index | top-{len(hits)} for a Tabang River query:")
    for h in hits:
        m = h.node.metadata
        print(f"  {h.score:.3f}  {m['contract_id']}/{m['doc_type']}  {h.node.text[:60]!r}")

    # NB: this assertion is coupled to the DEFAULT embed model's ranking. It is a persistence
    # smoke test (does a reloaded index retrieve?), NOT the recall gate. Swapping RAG_EMBED_* can
    # legitimately change the top hit -- if this fails after a model swap, confirm with `make eval`
    # (the real recall gate) before treating it as a retrieval regression.
    top = hits[0].node.metadata
    assert top["contract_id"] == "24CC0265", f"expected 24CC0265, got {top['contract_id']}"
    print("\nOK: top hit is from contract 24CC0265")
