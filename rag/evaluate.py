"""Step 6 of the RAG pipeline: measure retrieval recall (hit_rate) and MRR.

Runs the ground-truth set (eval/eval_retrieval.json) through the persisted index and reports
hit_rate (= recall@k: did any expected chunk appear in the top-k?) and MRR. Only the
embedding model (and, in rerank mode, the CPU cross-encoder) runs -- no generation LLM -- so
this is zero-VRAM.

Run bare, it uses pure vector retrieval (no metadata filter): the honest test of whether the
Step-2 contract enrichment alone disambiguates. RAG_FILTER_BY_CONTRACT=1 scopes each query to its
contract via MetadataFilters, and RAG_RERANK=1 adds the cross-encoder. `make eval` defaults to
rerank mode, which sets both (scripts/evaluate.sh) -- that is the 1.000 gate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, NamedTuple

from llama_index.core.evaluation import RetrieverEvaluator

from rag.config import RERANK_CANDIDATES, RERANK_TOP_N
from rag.generate import build_retriever  # shared with production so the gate can't drift
from rag.index import load_index

EVAL_PATH = Path("eval/eval_retrieval.json")
_MANIFEST_KINDS = ("rank", "aggregate")  # routes answered from the manifest, not retrieval
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
FILTER_BY_CONTRACT = os.getenv("RAG_FILTER_BY_CONTRACT", "0") == "1"
RERANK = os.getenv("RAG_RERANK", "0") == "1"
_METRIC_NAMES = ["hit_rate", "mrr"]


def evaluate(top_k: int = TOP_K, filter_by_contract: bool = FILTER_BY_CONTRACT, rerank: bool = RERANK):
    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    index = load_index()

    reranker = None
    if rerank:
        from rag.rerank import get_reranker
        reranker = get_reranker(RERANK_TOP_N)

    # With reranking we retrieve a wide candidate set, then narrow to RERANK_TOP_N.
    candidate_k = RERANK_CANDIDATES if rerank else top_k

    # The agent's own router decides which no-id questions fan out, and over which contracts. It
    # needs only the manifest-backed resolver, so call it on a stub instead of loading the LLM.
    route = None
    if filter_by_contract:
        from rag.agent import manifest_router
        from rag.manifest import load_manifest
        route = manifest_router(load_manifest())

    results = []
    metrics = None  # resolve the metric objects once (F7); the evaluator wrapper is per-item
    for item in dataset:                                # because the contract filter is per-item.
        # `empty` questions have no ground-truth chunk (the answer is absent by design); they are
        # withhold/negative tests, not recall probes, and RetrieverEvaluator needs expected_ids.
        if item.get("type") == "empty":
            continue
        # A contract filter only makes sense for single-contract items. Cross-contract broad/
        # complex questions (contract_id is null) span contracts, so never scope them.
        use_filter = filter_by_contract and item["contract_id"]
        contract_id = item["contract_id"] if use_filter else None
        # Same builder production uses (rag.generate.build_retriever), so gate == prod retriever.
        retriever = build_retriever(index, contract_id, candidate_k, reranker)
        # with a contract filter, search on intent only: the item's pre-stripped search_query.
        # (the human-style `query` names the contract, which biases ranking toward header chunks.)
        query_text = item["search_query"] if use_filter else item["query"]
        if metrics is None:
            evaluator = RetrieverEvaluator.from_metric_names(_METRIC_NAMES, retriever=retriever)
            metrics = evaluator.metrics
        else:
            evaluator = RetrieverEvaluator(metrics=metrics, retriever=retriever)
        result = evaluator.evaluate(query=query_text, expected_ids=item["expected_ids"])
        # The chunks production actually sees: a no-id question the router fans out runs one
        # filtered search per contract, so pool those; otherwise the single pass above. A `rank` or
        # `aggregate` question is answered from the manifest, not chunks -- the kind tells main() so.
        seen, kind = set(result.retrieved_ids), ""
        if route and not item["contract_id"]:
            kind, ids = route(item["query"])
            if kind == "fanout":
                seen = {n.node.node_id for cid in ids
                        for n in build_retriever(index, cid, candidate_k, reranker).retrieve(item["search_query"])}
        results.append((item, result, seen, kind))
    return results


def _anchor_coverage(item, result) -> float:
    """Chunk coverage (the old number, kept for comparison with old runs): the fraction of a
    question's expected chunks in the single pass's top-k. Capped below 1 whenever a question has
    more expected chunks than k (project_locations: 85 chunks, so at most 10/85)."""
    expected = set(item["expected_ids"])
    if not expected:
        return 0.0
    return len(expected & set(result.retrieved_ids)) / len(expected)


class _Row(NamedTuple):
    """One scored eval question, computed once and read by both report tables."""
    item: dict
    result: Any
    hit: int
    fcov: float
    ccov: float
    manifest: bool  # answered from the manifest (_MANIFEST_KINDS): fcov kept out of the retrieval mean


def _fact_coverage(item, texts) -> float:
    """Fact coverage: the fraction of a question's anchors (answer facts) found in at least one
    seen text -- the raw OCR of each seen chunk, or the manifest for a `rank` or `aggregate`
    question. Matches the same way rag.build_retrieval_eval picks expected_ids, so a loader label
    can never count as a found fact."""
    from rag.build_retrieval_eval import match_text
    anchors = [match_text(a) for a in item["anchors"]]
    if not anchors:
        return 0.0
    texts = [match_text(t) for t in texts]
    return sum(any(a in t for t in texts) for a in anchors) / len(anchors)


def resolver_check():
    """Gate the NL->contract_id step the production router performs, which the retrieval eval above
    SKIPS by filtering on the ground-truth contract_id (F1). Route each item's human `query`
    through the SAME resolver production uses and compare to the item's contract_id (None for
    cross-contract broad/complex). Model-free: manifest + resolver only, no LLM/Ollama."""
    from rag.generate import _resolver_index, route_contract_id
    from rag.manifest import load_manifest

    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    resolver, phrases = _resolver_index(load_manifest())
    rows = [it for it in dataset if it.get("type") != "empty"]
    mismatches = [(it, route_contract_id(it["query"], resolver, phrases))
                  for it in rows if route_contract_id(it["query"], resolver, phrases) != it["contract_id"]]
    return rows, mismatches


if __name__ == "__main__":
    import sys

    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    n_empty = sum(1 for it in dataset if it.get("type") == "empty")

    # F1 gate: does the production resolver map each human query to the right contract filter? The
    # retrieval eval below uses the ground-truth id, so this covers the NL->filter step it skips.
    rows, mismatches = resolver_check()
    print(f"resolver gate: {len(rows) - len(mismatches)}/{len(rows)} queries -> correct contract "
          f"filter (NL->id, the step the retrieval eval assumes perfect)")
    if mismatches:
        for it, got in mismatches:
            print(f"  MISRESOLVED [{it.get('type')}/{it['intent']}] expected {it['contract_id']} "
                  f"got {got}  {it['query'][:50]!r}")
        sys.exit(f"resolver gate FAILED: {len(mismatches)} query(ies) resolve to the wrong filter")
    print()

    from rag.build_retrieval_eval import _ocr_content
    results = evaluate()
    n = len(results)
    ocr = _ocr_content()
    from rag.manifest import format_manifest, load_manifest
    manifest_text = format_manifest(load_manifest())

    def fact_cov(item, seen, kind) -> float:
        # A `rank`/`aggregate` question is answered from the manifest: score it against what it reads.
        return _fact_coverage(item, [manifest_text] if kind in _MANIFEST_KINDS
                              else [ocr.get(i, "") for i in seen])

    mode = "contract-filtered" if FILTER_BY_CONTRACT else "pure-vector"
    if RERANK:
        mode += f" +rerank({RERANK_CANDIDATES}->{RERANK_TOP_N})"
    k_shown = RERANK_TOP_N if RERANK else TOP_K

    # Per-type recall. `single` is THE gate (CLAUDE.md); broad/complex hit_rate is lenient ("any
    # expected chunk in top-k"), so it reads high by construction -- report FACT coverage (how many
    # answer facts the chunks production sees contain) alongside it, plus the old chunk coverage.
    # `rank`/`aggregate` rows are kept OUT of the fact_cov mean (it stays a retrieval number,
    # comparable with old runs) and get their own manifest line; hit_rate/chunk_cov still score
    # their single pass.
    rows = [_Row(item, r, int(r.metric_vals_dict["hit_rate"]), fact_cov(item, seen, kind),
                 _anchor_coverage(item, r), kind in _MANIFEST_KINDS) for item, r, seen, kind in results]
    by_type: dict[str, list[_Row]] = {}
    for row in rows:
        by_type.setdefault(row.item.get("type", "single"), []).append(row)
    single = by_type.get("single", [])
    single_hit = sum(row.hit for row in single) / len(single) if single else 0.0

    # F5: the GATE is the single-type recall, not the strict+lenient blend.
    print(f"mode={mode} top_k={k_shown} | scored={n} | empty(withheld)={n_empty}")
    print(f"GATE (single recall) hit_rate={single_hit:.3f}  <- the pass/fail number (CLAUDE.md)\n")
    for t in ("single", "broad", "complex"):
        rowsT = by_type.get(t, [])
        if rowsT:
            hr = sum(row.hit for row in rowsT) / len(rowsT)
            ccov = sum(row.ccov for row in rowsT) / len(rowsT)
            retrieved = [row.fcov for row in rowsT if not row.manifest]
            ranked = [row.fcov for row in rowsT if row.manifest]
            fcov = sum(retrieved) / len(retrieved) if retrieved else 0.0
            tag = "" if t == "single" else "  [lenient: any-hit; fact_cov = answer facts found]"
            print(f"  {t:<8} hit_rate={hr:.3f}  fact_cov={fcov:.3f}  ({len(rowsT)}; fact_cov over "
                  f"{len(retrieved)}){tag}")
            print(f"  {'':<8} chunk_cov={ccov:.3f}  [old: expected chunks in one pass's top-k]")
            if ranked:
                print(f"  {'':<8} manifest fact_cov={sum(ranked) / len(ranked):.3f}  ({len(ranked)})  "
                      f"[rank/aggregate routes: answered from the manifest, not retrieval]")
    if FILTER_BY_CONTRACT:
        print("  (fact_cov: a no-id question the agent routes to fanout pools one filtered search per contract;"
              " a rank or aggregate question is scored against the manifest, marked (manifest) below)")
    # F3: the ablation is not equal-k -- baseline/filtered score at TOP_K, rerank at RERANK_TOP_N,
    # so "rerank > filtered" mixes the reorder with a bigger window. For an apples-to-apples reader
    # run filtered at the same k: `RAG_FILTER_BY_CONTRACT=1 RAG_TOP_K=10 make eval MODE=filtered`.
    if RERANK and RERANK_TOP_N != TOP_K:
        print(f"\n  NOTE: rerank scores at k={RERANK_TOP_N} vs baseline/filtered at k={TOP_K} "
              f"(not equal-k). To isolate the reranker, compare filtered at RAG_TOP_K={RERANK_TOP_N}.")
    print()

    print(f"  {'hit':>3} {'fcov':>4} {'ccov':>4} {'mrr':>5}  type     intent/contract                query")
    for item, r, h, fcov, ccov, manifest in rows:
        m = r.metric_vals_dict["mrr"]
        mark = " " if h else "*"
        src = " (manifest)" if manifest else ""
        print(f" {mark}{h:>3} {fcov:>4.2f} {ccov:>4.2f} {m:>5.2f}  {item.get('type', 'single'):<8} {item['intent']:<20} "
              f"{str(item['contract_id'] or '-'):<9} {item['query'][:38]}{src}")

    misses = [(item, r) for item, r, _, _ in results if r.metric_vals_dict["hit_rate"] == 0]
    if misses:
        print(f"\n{len(misses)} MISS(es):")
        for item, r in misses:
            print(f"  [{item.get('type')}/{item['contract_id']}/{item['intent']}] expected {len(item['expected_ids'])} ids")
            print(f"    retrieved: {r.retrieved_ids}")
