"""Step 1 of the RAG pipeline: load the authoritative OCR database into TextNodes.

Reads ``database/*.json`` (read-only; the DB is the trusted OCR transcription) and
produces exactly one LlamaIndex ``TextNode`` per OCR chunk:

- ``node.id_``     = the chunk's own UUID (globally unique across the DB)
- ``node.text``    = the chunk's ``content``, verbatim
- ``node.metadata``= raw source fields later steps need, copied as-is (no derivation)

Deliberately does NOT enrich (Step 2) or wire relationships (Step 3).
"""

from __future__ import annotations

import json
from pathlib import Path

from llama_index.core.schema import TextNode

# The four chunk categories in every task_*.json file, all sharing one schema.
SECTIONS = ("text", "table", "image", "signature")

# A notary stamp is not a contract signatory: the generic "Signatory:" label buries
# the block's real role, so retrieval (embed + cross-encoder, which score only the
# chunk text) can't match role-based queries to it. Label it with what it is.
NOTARY_PREFIX = "Notary Public who notarized this document: "


def load_nodes(database_dir: str = "database") -> list[TextNode]:
    """Return one TextNode per OCR chunk found in ``database_dir``."""
    nodes: list[TextNode] = []
    for path in sorted(Path(database_dir).glob("task_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for category in SECTIONS:
            for uuid, chunk in data.get(category, {}).items():
                raw = chunk["content"]
                # Prefix signature chunks with their semantic role so the embedding
                # captures "signatory" rather than just the bare name+title block.
                # "Contract" is deliberately omitted from the prefix — it appears in most
                # manifest-extraction queries and would displace content chunks in that context.
                if category == "signature":
                    prefix = NOTARY_PREFIX if "notary public" in raw.lower() else "Signatory: "
                    text = prefix + raw
                else:
                    text = raw
                nodes.append(
                    TextNode(
                        id_=uuid,
                        text=text,
                        metadata={
                            "category": category,
                            "pdf_source": chunk["pdfSource"],
                            "pdf_page": chunk["pdfPage"],
                            "coordinate": chunk["coordinate"],
                        },
                    )
                )
    return nodes


if __name__ == "__main__":
    from collections import Counter

    nodes = load_nodes()
    ids = {n.id_ for n in nodes}
    by_cat = Counter(n.metadata["category"] for n in nodes)

    print(f"loaded {len(nodes)} nodes | distinct ids {len(ids)}")
    for category in SECTIONS:
        print(f"  {category}: {by_cat[category]}")

    assert len(nodes) == len(ids), "duplicate node ids: UUIDs are not unique"
