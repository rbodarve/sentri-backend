"""Step 1 of the RAG pipeline: load the authoritative OCR database into TextNodes.

Reads ``database/*.json`` (read-only; the DB is the trusted OCR transcription) and
produces exactly one LlamaIndex ``TextNode`` per OCR chunk:

- ``node.id_``     = the chunk's own UUID (globally unique across the DB)
- ``node.text``    = the chunk's ``content``, verbatim -- except for a short label that says what a
                     chunk is when its text alone doesn't: a role prefix on signature chunks
                     ("Signatory: " / NOTARY_PREFIX) and a page-derived label on a few
                     context-free text chunks (see the regexes below). index.py strips the
                     signature prefixes again before parsing names.
- ``node.metadata``= raw source fields later steps need, copied as-is (no derivation)

Deliberately does NOT enrich (Step 2) or wire relationships (Step 3).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from llama_index.core.schema import TextNode

# The four chunk categories in every task_*.json file, all sharing one schema.
SECTIONS = ("text", "table", "image", "signature")

# A notary stamp is not a contract signatory: the generic "Signatory:" label buries
# the block's real role, so retrieval (embed + cross-encoder, which score only the
# chunk text) can't match role-based queries to it. Label it with what it is.
NOTARY_PREFIX = "Notary Public who notarized this document: "

# A notice's issue date is OCR'd as a chunk holding only the stamp ('MAR 05 2024'), printed just
# under the page title. Nothing in its text says what the date is, so a "date of the notice of
# award" question ranked a self-describing date from the same PDF ("RESOLVED ... MAR 04 2024")
# above it. Label such a chunk with the title printed on its own page.
_BARE_DATE_RE = re.compile(r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}")
_NOTICE_TITLES = {"NOTICE OF AWARD": "Notice of Award", "NOTICE TO PROCEED": "Notice to Proceed"}
# The notary's register entry ("Doc. No. 249 / Page No. 51 / Book No. X / Series of 2024") never
# says whose register it is, so "notary Page No." questions read the PDF page number instead.
_REGISTER_RE = re.compile(r"\s*Doc\.? ?No\..*Book No\..*Series", re.S)
# Same for a BAC resolution's closing "RESOLVED, at DPWH-Regional Office I, this [] day of MAR 04 2024".
_RESOLVED_RE = re.compile(r"\s*RESOLVED,?\s+at\b.*?this\s+(.*?\d{4})\s*$", re.S)
# ...and its approval line ("Date Approved MAR 21 2024"), which is a different date (resolved MAR 20,
# approved MAR 21): labelled only on a page carrying a RESOLVED line, so it reads as the resolution's.
_APPROVED_RE = re.compile(r"\s*Date Approved:?\s+(.*?\d{4})\s*", re.I)
# The Contract Agreement's execution date only reads "THIS AGREEMENT, made this 17th day of April,
# 2024 between ..." -- never "executed"/"date" -- so once the notice dates were labelled they
# outranked it for "when was the contract agreement executed?". Label it the same way.
_MADE_THIS_RE = re.compile(r"AGREEMENT,?\s+made this\s+(.*?\d{4})", re.I | re.S)


def load_nodes(database_dir: str = "database") -> list[TextNode]:
    """Return one TextNode per OCR chunk found in ``database_dir``."""
    nodes: list[TextNode] = []
    for path in sorted(Path(database_dir).glob("task_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        titles = {(c["pdfSource"], c["pdfPage"]): _NOTICE_TITLES[c["content"].strip().upper()]
                  for c in data.get("text", {}).values()
                  if c["content"].strip().upper() in _NOTICE_TITLES}
        resolutions = {(c["pdfSource"], c["pdfPage"]) for c in data.get("text", {}).values()
                       if _RESOLVED_RE.match(c["content"])}
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
                elif (category == "text" and _BARE_DATE_RE.fullmatch(raw.strip())
                      and (chunk["pdfSource"], chunk["pdfPage"]) in titles):
                    text = f"Date of this {titles[chunk['pdfSource'], chunk['pdfPage']]}: {raw}"
                elif category == "text" and _REGISTER_RE.match(raw):
                    text = f"Notarial register entry of the Notary Public (notary's own numbering): {raw}"
                elif category == "text" and (resolved := _RESOLVED_RE.match(raw)):
                    text = f"Date this BAC Resolution was resolved: {resolved.group(1)}. {raw}"
                elif (category == "text" and (approved := _APPROVED_RE.fullmatch(raw))
                      and (chunk["pdfSource"], chunk["pdfPage"]) in resolutions):
                    text = f"Date this BAC Resolution was approved: {approved.group(1)}. {raw}"
                elif category == "text" and (made := _MADE_THIS_RE.search(raw)):
                    text = f"Date this Contract Agreement was executed: {made.group(1)}. {raw}"
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
