"""Step 4 of the RAG pipeline: embed the enriched nodes and build a persistent index.

Composes Steps 1-3 (load -> enrich -> link) into a node list, embeds them with the
configured model (see rag.config), and stores them in an in-memory SimpleVectorStore
persisted to disk. Embedding is the only step that touches a model; it runs once and is
reloaded from disk thereafter, so the cost is paid a single time.
"""

from __future__ import annotations

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

from rag.config import PERSIST_DIR, get_embed_model
from rag.enrich import enrich_nodes
from rag.loader import load_nodes
from rag.relationships import link_reading_order


def build_nodes():
    """Run Steps 1-3 and return the fully prepared node list (no embedding)."""
    nodes = load_nodes()
    enrich_nodes(nodes)
    link_reading_order(nodes)
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

    top = hits[0].node.metadata
    assert top["contract_id"] == "24CC0265", f"expected 24CC0265, got {top['contract_id']}"
    print("\nOK: top hit is from contract 24CC0265")
