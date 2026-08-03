"""Step 3 of the RAG pipeline: wire PREV/NEXT reading-order relationships.

The DB's ``reference_above``/``reference_below`` fields are only ~4% populated, so the
intra-document graph they were meant to hold barely exists. Instead we reconstruct it
deterministically from geometry, which every chunk has: group chunks by document
(``pdf_source``) and order them by ``(pdf_page, y, x)`` -- i.e. top-to-bottom then
left-to-right reading order -- then chain consecutive chunks with PREV/NEXT.

This gives later retrieval a reliable way to expand a hit to its neighbouring blocks.
Pure Python (sort + dict); no model, no VRAM.
"""

from __future__ import annotations

from collections import defaultdict

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode


def _reading_order_key(node: TextNode) -> tuple[int, float, float]:
    x, y = node.metadata["coordinate"][0], node.metadata["coordinate"][1]
    return (int(node.metadata["pdf_page"]), y, x)


def link_reading_order(nodes: list[TextNode]) -> list[TextNode]:
    """Chain each document's chunks in reading order via PREV/NEXT. Mutates in place."""
    by_document: dict[str, list[TextNode]] = defaultdict(list)
    for node in nodes:
        by_document[node.metadata["pdf_source"]].append(node)

    for doc_nodes in by_document.values():
        doc_nodes.sort(key=_reading_order_key)
        for prev, curr in zip(doc_nodes, doc_nodes[1:]):
            prev.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=curr.node_id)
            curr.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=prev.node_id)

    return nodes


if __name__ == "__main__":
    from rag.enrich import enrich_nodes
    from rag.loader import load_nodes

    nodes = load_nodes()
    enrich_nodes(nodes)
    link_reading_order(nodes)

    by_id = {n.node_id: n for n in nodes}
    documents = {n.metadata["pdf_source"] for n in nodes}
    n_next = sum(1 for n in nodes if NodeRelationship.NEXT in n.relationships)
    n_prev = sum(1 for n in nodes if NodeRelationship.PREVIOUS in n.relationships)

    print(f"{len(nodes)} nodes | {len(documents)} documents | NEXT {n_next} | PREV {n_prev}")

    # 1) one connected chain per document => edges == nodes - documents
    assert n_next == n_prev == len(nodes) - len(documents), "unexpected edge count"

    # 2) PREV/NEXT are inverse, and never cross a document boundary
    for node in nodes:
        nxt = node.relationships.get(NodeRelationship.NEXT)
        if nxt:
            target = by_id[nxt.node_id]
            assert target.relationships[NodeRelationship.PREVIOUS].node_id == node.node_id
            assert target.metadata["pdf_source"] == node.metadata["pdf_source"]

    # 3) eyeball reading order for one document
    doc = "24CC0265 ROA.pdf"
    head = next(
        n for n in nodes
        if n.metadata["pdf_source"] == doc and NodeRelationship.PREVIOUS not in n.relationships
    )
    print(f"\n--- reading-order chain: {doc} ---")
    cur, i = head, 0
    while cur and i < 6:
        y = cur.metadata["coordinate"][1]
        print(f"  p{cur.metadata['pdf_page']} y={y:>7.1f}  {cur.text[:55]!r}")
        nxt = cur.relationships.get(NodeRelationship.NEXT)
        cur = by_id[nxt.node_id] if nxt else None
        i += 1
