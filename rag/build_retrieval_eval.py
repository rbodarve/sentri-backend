"""Step 5 of the RAG pipeline: build the retrieval ground-truth set (query -> chunk UUIDs).

Hand-curated -- no LLM, no VRAM. Each question is anchored to answer-bearing substrings;
``expected_ids`` is computed as EVERY chunk in scope whose OCR text contains any anchor, so
retrieving any legitimate copy of the answer counts as a hit (this avoids false misses when a
fact appears in several chunks). The result is written to ``eval/eval_retrieval.json`` -- a
static, inspectable artifact for the handoff and for Step 6.

Each question carries a ``type``:

- ``single``  -- one fact, scoped to one contract (the classic recall probe).
- ``broad``   -- an answer that spans many chunks/pages (and, when ``contract_id`` is null,
                 many contracts): e.g. "list every District Engineer across the corpus".
- ``complex`` -- needs computation/logic/comparison on top of retrieval (sums, ranking,
                 differences, percentages). ``expected_ids`` are the evidence chunks the
                 downstream model must retrieve to do the math.
- ``empty``   -- the answer is NOT in the database (invented project, or a fact genuinely
                 absent from a real contract). ``expected_ids`` is ``[]`` by design; these are
                 withhold/negative tests, not recall probes.

Scope: a spec with a ``contract_id`` matches anchors within that contract only; a spec with
``contract_id=None`` matches across the whole corpus (cross-contract broad/complex). For
``empty`` specs, ``anchors`` are read as *absence probes*: the builder asserts they match zero
chunks corpus-wide, proving the question is unanswerable (specs with no probe are absences
verified by inspection -- e.g. contract 24CM0001 holds only a Notice to Proceed, so it carries
no amount, duration, or notary).

Anchors were chosen by inspecting the corpus and are kept in the output for traceability.
Full names are used deliberately (e.g. "ADONIS A. ASIS", not "ASIS") to avoid substring
collisions such as "asis" inside "Rental Basis".
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from rag.enrich import enrich_nodes
from rag.loader import SECTIONS, load_nodes

# Each spec: type, intent, contract_id (None = cross-corpus / no filter), query, search_query,
# anchors. `query` is the human, location/name-based question; `search_query` is the intent-only
# text used once retrieval is contract-filtered (rag.evaluate filtered/rerank mode), where naming
# the contract would just bias ranking toward header chunks. For cross-contract and empty specs
# the filter is off, so `search_query` mirrors `query` (it is not used to strip a contract).
QUESTION_SPECS: list[dict] = [
    # -- 20 SINGLE: one fact, one contract --------------------------------------------------
    dict(type="single", intent="contract_name",     contract_id="24CC0265", anchors=["Tabang River, Barangay Tibig"],
         query="What is the official contract name of the Tabang River flood control project in Bulakan, Bulacan?", search_query="What is the contract name?"),
    dict(type="single", intent="winning_bidder",    contract_id="24CC0265", anchors=["AMETHYST HORIZON"],
         query="Which contractor was awarded the Tabang River flood control project in Bulakan, Bulacan?", search_query="Which contractor was awarded?"),
    dict(type="single", intent="amount",            contract_id="24CC0265", anchors=["69,479,428"],
         query="What is the total contract amount for the Tabang River flood control project in Bulakan, Bulacan?", search_query="What is the total contract amount?"),
    dict(type="single", intent="district_engineer", contract_id="24CC0265", anchors=["HENRY C. ALCANTARA"],
         query="Who is the District Engineer for the Tabang River flood control project in Bulakan, Bulacan?", search_query="Who is the District Engineer?"),

    dict(type="single", intent="winning_bidder",    contract_id="24AJ0052", anchors=["JHI Construction"],
         query="Which contractor was awarded the slope protection project in San Carlos City, Pangasinan?", search_query="Which contractor was awarded?"),
    dict(type="single", intent="amount",            contract_id="24AJ0052", anchors=["19,109,972"],
         query="What is the contract amount for the slope protection project in San Carlos City, Pangasinan?", search_query="What is the contract amount?"),
    dict(type="single", intent="district_engineer", contract_id="24AJ0052", anchors=["MEL HARVEY A. GONZALES"],
         query="Who is the OIC District Engineer for the slope protection project in San Carlos City, Pangasinan?", search_query="Who is the OIC District Engineer?"),
    dict(type="single", intent="duration",          contract_id="24AJ0052", anchors=["163 Calendar Days"],
         query="What is the required completion time for the slope protection project in San Carlos City, Pangasinan?", search_query="What is the completion time in calendar days?"),

    dict(type="single", intent="winning_bidder",    contract_id="24BG0272", anchors=["JWU CONSTRUCTION"],
         query="Which contractor was awarded the Macañao Creek flood control project in Cabatuan, Isabela?", search_query="Which contractor was awarded?"),
    dict(type="single", intent="district_engineer", contract_id="24BG0272", anchors=["ADONIS A. ASIS"],
         query="Who is the District Engineer for the Macañao Creek flood control project in Cabatuan, Isabela?", search_query="Who is the District Engineer?"),
    dict(type="single", intent="amount",            contract_id="24BG0272", anchors=["96,489,983"],
         query="What is the contract amount for the Macañao Creek flood control project in Cabatuan, Isabela?", search_query="What is the contract amount?"),
    dict(type="single", intent="duration",          contract_id="24BG0272", anchors=["Twenty One (321)"],
         query="What is the construction duration for the Macañao Creek flood control project in Cabatuan, Isabela?", search_query="What is the construction duration in calendar days?"),

    dict(type="single", intent="winning_bidder",    contract_id="24BJ0005", anchors=["BRG CONSTRUCTION"],
         query="Which contractor was awarded the flood control project in Sta. Fe, Nueva Vizcaya?", search_query="Which contractor was awarded?"),
    dict(type="single", intent="district_engineer", contract_id="24BJ0005", anchors=["MARLYN G. INGUILLO"],
         query="Who is the District Engineer for the flood control project in Sta. Fe, Nueva Vizcaya?", search_query="Who is the District Engineer?"),
    dict(type="single", intent="amount",            contract_id="24BJ0005", anchors=["93,990,000"],
         query="What is the contract amount for the flood control project in Sta. Fe, Nueva Vizcaya?", search_query="What is the contract amount?"),

    dict(type="single", intent="winning_bidder",    contract_id="24CM0001", anchors=["JAMJLE"],
         query="Which contractor was awarded the flood mitigation project in Olongapo City?", search_query="Which contractor was awarded?"),
    dict(type="single", intent="district_engineer", contract_id="24CM0001", anchors=["ROSBE S. DIZON"],
         query="Who is the District Engineer for the flood mitigation project in Olongapo City?", search_query="Who is the District Engineer?"),

    dict(type="single", intent="winning_bidder",    contract_id="24A00153", anchors=["Rodekom"],
         query="Which contractor was awarded the Aringay River flood control project in Tubao, La Union?", search_query="Which contractor was awarded?"),
    dict(type="single", intent="regional_director", contract_id="24A00153", anchors=["RONNEL M. TAN"],
         query="Who is the Regional Director that awarded the Aringay River flood control project in Tubao, La Union?", search_query="Who is the Regional Director?"),
    dict(type="single", intent="amount",            contract_id="24A00153", anchors=["140,274,481"],
         query="What is the total contract amount for the Aringay River flood control project in Tubao, La Union?", search_query="What is the total contract amount?"),

    # -- 10 BROAD: answer spans many chunks/pages (contract_id=None -> across contracts) -----
    dict(type="broad", intent="all_district_engineers", contract_id=None,
         anchors=["HENRY C. ALCANTARA", "MEL HARVEY A. GONZALES", "ADONIS A. ASIS", "MARLYN G. INGUILLO", "ROSBE S. DIZON", "RONNEL M. TAN"],
         query="List every District Engineer, OIC District Engineer, or Regional Director named across all the contracts in the database.", search_query="List every District Engineer, OIC District Engineer, or Regional Director named across all the contracts in the database."),
    dict(type="broad", intent="all_winning_bidders", contract_id=None,
         anchors=["AMETHYST HORIZON", "JHI Construction", "JWU CONSTRUCTION", "BRG CONSTRUCTION", "JAMJLE", "Rodekom"],
         query="Which contractors were awarded contracts across the corpus?", search_query="Which contractors were awarded contracts across the corpus?"),
    dict(type="broad", intent="all_amounts", contract_id=None,
         anchors=["69,479,428", "19,109,972", "96,489,983", "93,990,000", "140,274,481"],
         query="What are all the awarded contract amounts recorded across the projects?", search_query="What are all the awarded contract amounts recorded across the projects?"),
    dict(type="broad", intent="losing_bidders", contract_id="24CC0265",
         anchors=["Sto. Cristo", "Jewel's Construction"],
         query="List the non-winning bidders for the Tabang River contract in Bulakan, Bulacan.", search_query="Who are the losing bidders?"),
    dict(type="broad", intent="all_bidders", contract_id="24A00153",
         anchors=["R.U. Aquino", "Rodekom", "Grace Construction"],
         query="Who are all the bidders that submitted for the Aringay River project in Tubao, La Union?", search_query="Who are all the bidders that submitted?"),
    dict(type="broad", intent="bac_members", contract_id="24CC0265",
         anchors=["IRENE DC. ONTINGCO", "EVELYN T. DE JESUS", "NORBERTO L. SANTOS", "LORENZO A. PAGTALUNAN", "ERNESTO C. GALANG", "JAYPEE D. MENDOZA"],
         query="Which Bids and Awards Committee (BAC) members signed the resolution for the Tabang River contract?", search_query="Who are the BAC members that signed the resolution?"),
    dict(type="broad", intent="project_locations", contract_id=None,
         anchors=["Bulacan", "Pangasinan", "Isabela", "Nueva Vizcaya", "La Union", "Olongapo"],
         query="In which provinces or cities are the flood-control projects located across the corpus?", search_query="In which provinces or cities are the flood-control projects located across the corpus?"),
    dict(type="broad", intent="notaries", contract_id=None,
         anchors=["AILEIN GRACE", "ALFREDO S. VELASCO", "FERNANDEZ B. TAGART"],
         query="List all the Notary Publics who notarized the contract agreements in the corpus.", search_query="List all the Notary Publics who notarized the contract agreements in the corpus."),
    dict(type="broad", intent="accountants", contract_id=None,
         anchors=["DEXTER D. LOMBOY", "JULIETA C. BACANI", "JAYSON V. ANTONIO", "JONNALYN Q. BARASI", "JUANITO C. MENDOZA"],
         query="Who are the accountants named across the contracts?", search_query="Who are the accountants named across the contracts?"),
    # Known gap (fact_cov 0.80): the BAC Resolution is bundled inside the NOA PDF (doc_type NOA,
    # pp. 2-3), so metadata and the manifest cannot see it; its only evidence is one short title
    # chunk, which this meta question ranks 37/40. Kept as-is, not tuned for (2026-10-03).
    dict(type="broad", intent="document_types", contract_id="24A00153",
         anchors=["NOTICE OF AWARD", "CONTRACT AGREEMENT", "NOTICE TO PROCEED", "BAC RESOLUTION", "ACKNOWLEDGEMENT"],
         query="What document types make up the Aringay River contract file in Tubao, La Union?", search_query="What document types make up this contract file?"),

    # -- 10 COMPLEX: retrieval + computation/logic/comparison -------------------------------
    dict(type="complex", intent="max_amount", contract_id=None,
         anchors=["69,479,428", "19,109,972", "96,489,983", "93,990,000", "140,274,481"],
         query="Which contract has the highest awarded amount, and how much is it?", search_query="Which contract has the highest awarded amount, and how much is it?"),
    dict(type="complex", intent="total_amount", contract_id=None,
         anchors=["69,479,428", "19,109,972", "96,489,983", "93,990,000", "140,274,481"],
         query="What is the combined total value of all the awarded contract amounts in the corpus?", search_query="What is the combined total value of all the awarded contract amounts in the corpus?"),
    dict(type="complex", intent="abc_savings", contract_id="24A00153",
         anchors=["144,750,000", "140,274,481"],
         query="By how much did the winning bid for the Aringay River project come in under its Approved Budget for the Contract (ABC)?", search_query="How much under the Approved Budget for the Contract was the winning bid?"),
    dict(type="complex", intent="bid_gap", contract_id="24CC0265",
         anchors=["71,651,188", "69,479,428"],
         query="For the Tabang River contract, how much more did the second-lowest bidder bid than the winning bid?", search_query="How much more did the second-lowest bidder bid than the winning bid?"),
    dict(type="complex", intent="discount_check", contract_id="24A00153",
         anchors=["143,137,226", "140,274,481"],
         query="Rodekom's bid for the Aringay River project was read at 143,137,226 with a 2% discount offered -- does the discounted amount match the awarded contract price?", search_query="Does the bid after the 2% discount match the awarded contract price?"),
    dict(type="complex", intent="max_duration", contract_id=None,
         anchors=["240 (two hundred forty)", "163 Calendar Days", "Twenty One (321)"],
         query="Which contract has the longest construction duration in calendar days?", search_query="Which contract has the longest construction duration in calendar days?"),
    dict(type="complex", intent="rank_amounts", contract_id=None,
         anchors=["69,479,428", "19,109,972", "96,489,983", "93,990,000", "140,274,481"],
         query="Rank the corpus contracts from smallest to largest awarded amount.", search_query="Rank the corpus contracts from smallest to largest awarded amount."),
    dict(type="complex", intent="bid_spread", contract_id="24CC0265",
         anchors=["69,479,428", "71,868,373"],
         query="How much did the highest bid exceed the lowest bid for the Tabang River contract?", search_query="How much did the highest bid exceed the lowest bid?"),
    dict(type="complex", intent="abc_percentage", contract_id="24AJ0052",
         anchors=["19,600,000", "19,109,972"],
         query="The San Carlos City slope-protection project had an ABC of 19,600,000 -- what percentage of the ABC was the winning bid?", search_query="What percentage of the Approved Budget for the Contract was the winning bid?"),
    dict(type="complex", intent="avg_amount", contract_id=None,
         anchors=["69,479,428", "19,109,972", "96,489,983", "93,990,000", "140,274,481"],
         query="Across all contracts, what is the average awarded contract amount?", search_query="Across all contracts, what is the average awarded contract amount?"),

    # -- 10 EMPTY: answer is NOT in the database (anchors are absence probes; [] = verified) --
    dict(type="empty", intent="absent", contract_id=None, anchors=["Pasig River"],
         query="Which contractor was awarded the Pasig River flood control project in Metro Manila?", search_query="Which contractor was awarded the Pasig River flood control project in Metro Manila?"),
    dict(type="empty", intent="absent", contract_id="24CM0001", anchors=[],
         query="What is the total contract amount for the Kalaklan River flood project in Olongapo City?", search_query="What is the total contract amount for the Kalaklan River flood project in Olongapo City?"),
    dict(type="empty", intent="absent", contract_id="24CM0001", anchors=[],
         query="Who is the Notary Public for the Olongapo City (Kalaklan River) contract?", search_query="Who is the Notary Public for the Olongapo City (Kalaklan River) contract?"),
    dict(type="empty", intent="absent", contract_id=None, anchors=["Boracay"],
         query="What is the total contract amount for the Boracay Circumferential Road project?", search_query="What is the total contract amount for the Boracay Circumferential Road project?"),
    dict(type="empty", intent="absent", contract_id=None, anchors=["Marikina"],
         query="Which company was awarded the Marikina River flood control project?", search_query="Which company was awarded the Marikina River flood control project?"),
    dict(type="empty", intent="absent", contract_id=None, anchors=["Cagayan de Oro"],
         query="Who won the Cagayan de Oro River flood control contract?", search_query="Who won the Cagayan de Oro River flood control contract?"),
    dict(type="empty", intent="absent", contract_id="24CM0001", anchors=[],
         query="How many calendar days is the construction duration for the Olongapo City (Kalaklan River) project?", search_query="How many calendar days is the construction duration for the Olongapo City (Kalaklan River) project?"),
    dict(type="empty", intent="absent", contract_id=None, anchors=["Agno"],
         query="Which contractor was awarded the Agno River flood control project in Pangasinan?", search_query="Which contractor was awarded the Agno River flood control project in Pangasinan?"),
    dict(type="empty", intent="absent", contract_id=None, anchors=["Chico River"],
         query="What is the winning bidder for the Chico River irrigation project?", search_query="What is the winning bidder for the Chico River irrigation project?"),
    dict(type="empty", intent="absent", contract_id=None, anchors=["Megawide"],
         query="What bid amount did Megawide Construction Corporation submit?", search_query="What bid amount did Megawide Construction Corporation submit?"),
]

OUTPUT_PATH = Path("eval/eval_retrieval.json")


def _ocr_content(database_dir: str = "database") -> dict[str, str]:
    """Chunk UUID -> its raw OCR ``content``. Anchors are matched against this, not ``node.text``:
    rag.loader adds labels ("Date of this Notice of Award: ...") to some chunks, and a label must
    never make a chunk ground truth -- the "document types" question's "NOTICE OF AWARD" anchor
    would otherwise match every labelled date stamp."""
    content: dict[str, str] = {}
    for path in sorted(Path(database_dir).glob("task_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for category in SECTIONS:
            for uuid, chunk in data.get(category, {}).items():
                content[uuid] = chunk["content"]
    return content


def match_text(text: str) -> str:
    """Normalize text for anchor matching: the OCR keeps HTML entities in table cells
    ("Jewel&#39;s") and doubled spaces in signatures ("IRENE DC.  ONTINGCO"), so a literal anchor
    would never match them. Shared with rag.evaluate's fact coverage so both match the same way."""
    return re.sub(r"\s+", " ", html.unescape(text)).lower()


def build_dataset() -> tuple[list[dict], list[dict]]:
    """Return (dataset, problems). Each dataset entry is a query with its ground-truth ids.

    ``problems`` lists specs that fail their own contract: a non-empty spec with an anchor that
    matches zero chunks (a broken anchor), or an ``empty`` spec whose absence probe actually matched (so the
    answer is present and the question is not really unanswerable). Either is a build error.
    """
    nodes = load_nodes()
    enrich_nodes(nodes)
    ocr = {i: match_text(t) for i, t in _ocr_content().items()}
    by_contract: dict[str, list] = {}
    for node in nodes:
        by_contract.setdefault(node.metadata["contract_id"], []).append(node)

    dataset: list[dict] = []
    problems: list[dict] = []
    for spec in QUESTION_SPECS:
        scope = by_contract.get(spec["contract_id"], []) if spec["contract_id"] else nodes
        lowered_anchors = [match_text(a) for a in spec["anchors"]]
        matched = [n.id_ for n in scope if any(a in ocr[n.id_] for a in lowered_anchors)]

        if spec["type"] == "empty":
            expected_ids: list[str] = []          # the answer is absent by design
            if matched:                           # an absence probe hit -> not actually empty
                problems.append({**spec, "reason": f"absence probe matched {len(matched)} chunk(s)"})
        else:
            expected_ids = matched
            # Per anchor, not per spec: one dead anchor caps the question's fact coverage silently.
            dead = [a for a, low in zip(spec["anchors"], lowered_anchors)
                    if not any(low in ocr[n.id_] for n in scope)]
            if dead:
                problems.append({**spec, "reason": f"anchor(s) matched no chunk: {dead}"})

        dataset.append({
            "type": spec["type"],
            "query": spec["query"],
            "search_query": spec["search_query"],
            "contract_id": spec["contract_id"],
            "intent": spec["intent"],
            "anchors": spec["anchors"],
            "expected_ids": expected_ids,
        })
    return dataset, problems


if __name__ == "__main__":
    dataset, problems = build_dataset()

    # Fail loud on any broken spec -- a zero-match anchor, or an "empty" whose answer is present.
    if problems:
        print("PROBLEM specs (fix before saving):")
        for e in problems:
            print(f"  [{e['type']}/{e['contract_id']}/{e['intent']}] {e['reason']} anchors={e['anchors']}")
        raise SystemExit(1)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    # ensure_ascii=False so non-ASCII place names (e.g. "Macañao") stay literal, matching the
    # checked-in eval_retrieval.json -- regenerating must be a no-op, not a diff.
    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    from collections import Counter
    by_type = Counter(e["type"] for e in dataset)
    total_ids = sum(len(e["expected_ids"]) for e in dataset)
    print(f"wrote {OUTPUT_PATH} | {len(dataset)} questions "
          f"({', '.join(f'{k}:{by_type[k]}' for k in ('single', 'broad', 'complex', 'empty'))}) "
          f"| {total_ids} total expected ids")
    print(f"  {'type':<8}{'intent':<22}{'contract':<11}#expected  query")
    for e in dataset:
        print(f"  {e['type']:<8}{e['intent']:<22}{str(e['contract_id'] or '-'):<11}{len(e['expected_ids']):>6}    {e['query'][:70]}")
