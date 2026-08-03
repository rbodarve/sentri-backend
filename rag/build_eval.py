"""Step 5 of the RAG pipeline: build the retrieval ground-truth set (query -> chunk UUIDs).

Hand-curated -- no LLM, no VRAM. Each question is anchored to answer-bearing substrings;
``expected_ids`` is computed as EVERY chunk in the target contract whose OCR text contains
any anchor, so retrieving any legitimate copy of the answer counts as a hit (this avoids
false misses when a fact appears in several chunks). The result is written to
``eval/eval_set.json`` -- a static, inspectable artifact for the handoff and for Step 6.

Anchors were chosen by inspecting the corpus (contractor names, district engineers,
amounts) and are kept in the output for traceability.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.enrich import enrich_nodes
from rag.loader import load_nodes

# (intent, contract_id, query, [answer-anchor substrings], case-insensitive)
QUESTION_SPECS: list[tuple[str, str, str, list[str]]] = [
    # 24CC0265 -- Reinforced Concrete Flood Control, Tabang River, Bulacan
    ("contract_name",     "24CC0265", "What is the contract name of contract 24CC0265?", ["Contract Name"]),
    ("winning_bidder",    "24CC0265", "Which contractor was awarded contract 24CC0265?", ["AMETHYST HORIZON"]),
    ("amount",            "24CC0265", "What is the total contract amount for 24CC0265?", ["69,479,428"]),
    ("district_engineer", "24CC0265", "Who is the District Engineer for contract 24CC0265?", ["ALCANTARA"]),

    # 24AJ0052 -- Slope Protection, Pangasinan
    ("contract_name",     "24AJ0052", "What is the contract name of contract 24AJ0052?", ["Contract Name"]),
    ("winning_bidder",    "24AJ0052", "Which contractor was awarded contract 24AJ0052?", ["JHI Construction"]),
    ("amount",            "24AJ0052", "What is the contract amount for 24AJ0052?", ["NINETEEN MILLION"]),
    ("district_engineer", "24AJ0052", "Who is the OIC District Engineer for contract 24AJ0052?", ["GONZALES"]),

    # 24BG0272 -- Flood Mitigation Structure, Isabela
    ("contract_name",     "24BG0272", "What is the contract name of contract 24BG0272?", ["Contract Name"]),
    ("winning_bidder",    "24BG0272", "Which contractor was awarded contract 24BG0272?", ["JWU CONSTRUCTION"]),
    ("district_engineer", "24BG0272", "Who is the District Engineer for contract 24BG0272?", ["ASIS"]),
    ("amount",            "24BG0272", "What is the contract amount for 24BG0272?", ["Pesos"]),

    # 24BJ0005 -- Protect Lives and Properties Against Major Floods
    ("contract_name",     "24BJ0005", "What is the contract name of contract 24BJ0005?", ["Contract Name"]),
    ("winning_bidder",    "24BJ0005", "Which contractor was awarded contract 24BJ0005?", ["BRG CONSTRUCTION"]),
    ("district_engineer", "24BJ0005", "Who is the District Engineer for contract 24BJ0005?", ["INGUILLO"]),
    ("amount",            "24BJ0005", "What is the contract amount for 24BJ0005?", ["Pesos"]),

    # 24CM0001 -- Flood Management, Zambales (NTP only in the DB)
    ("contract_name",     "24CM0001", "What is the contract name of contract 24CM0001?", ["Contract Name"]),
    ("winning_bidder",    "24CM0001", "Which contractor was awarded contract 24CM0001?", ["JAMJLE"]),
    ("district_engineer", "24CM0001", "Who is the District Engineer for contract 24CM0001?", ["DIZON"]),

    # 24A00153 -- Flood Mitigation Facilities, DPWH Region I
    ("contract_name",     "24A00153", "What is the contract name of contract 24A00153?", ["Contract Name"]),
    ("winning_bidder",    "24A00153", "Which contractor was awarded contract 24A00153?", ["Rodekom"]),
    ("bac_official",      "24A00153", "Who signed as Chief of the Construction Division for contract 24A00153?", ["JUCAR"]),
]

OUTPUT_PATH = Path("eval/eval_set.json")


def build_dataset() -> tuple[list[dict], list[dict]]:
    """Return (dataset, zero_match_specs). Each dataset entry is a query with its ground-truth ids."""
    nodes = load_nodes()
    enrich_nodes(nodes)
    by_contract: dict[str, list] = {}
    for node in nodes:
        by_contract.setdefault(node.metadata["contract_id"], []).append(node)

    dataset: list[dict] = []
    zero_match: list[dict] = []
    for intent, contract_id, query, anchors in QUESTION_SPECS:
        lowered_anchors = [a.lower() for a in anchors]
        expected_ids = [
            node.id_
            for node in by_contract.get(contract_id, [])
            if any(a in node.text.lower() for a in lowered_anchors)
        ]
        entry = {
            "query": query,
            "contract_id": contract_id,
            "intent": intent,
            "anchors": anchors,
            "expected_ids": expected_ids,
        }
        dataset.append(entry)
        if not expected_ids:
            zero_match.append(entry)
    return dataset, zero_match


if __name__ == "__main__":
    dataset, zero_match = build_dataset()

    # Fail loud on any question with no ground truth -- a zero-match spec is a broken anchor.
    if zero_match:
        print("ZERO-MATCH questions (fix anchors before saving):")
        for e in zero_match:
            print(f"  [{e['contract_id']}/{e['intent']}] anchors={e['anchors']}")
        raise SystemExit(1)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    total_ids = sum(len(e["expected_ids"]) for e in dataset)
    print(f"wrote {OUTPUT_PATH} | {len(dataset)} questions | {total_ids} total expected ids")
    print(f"{'intent':<18}{'contract':<11}#expected  query")
    for e in dataset:
        print(f"  {e['intent']:<16}{e['contract_id']:<11}{len(e['expected_ids']):>6}    {e['query']}")
