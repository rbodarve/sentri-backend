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

from llama_index.core.evaluation import RetrieverEvaluator

from rag.config import RERANK_CANDIDATES, RERANK_TOP_N
from rag.generate import build_retriever  # shared with production so the gate can't drift
from rag.index import load_index

EVAL_PATH = Path("eval/eval_retrieval.json")
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
        results.append((item, result))
    return results


def _anchor_coverage(item, result) -> float:
    """Per-anchor recall: the fraction of a question's expected chunks that were retrieved. For
    broad/complex probes (many anchors across contracts) hit_rate=1 means only ONE anchor showed
    up; coverage says how much of the full corpus-spanning answer was actually retrieved."""
    expected = set(item["expected_ids"])
    if not expected:
        return 0.0
    return len(expected & set(result.retrieved_ids)) / len(expected)


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

    results = evaluate()
    n = len(results)

    mode = "contract-filtered" if FILTER_BY_CONTRACT else "pure-vector"
    if RERANK:
        mode += f" +rerank({RERANK_CANDIDATES}->{RERANK_TOP_N})"
    k_shown = RERANK_TOP_N if RERANK else TOP_K

    # Per-type recall. `single` is THE gate (CLAUDE.md); broad/complex hit_rate is lenient ("any
    # expected chunk in top-k"), so it reads high by construction -- report per-anchor COVERAGE
    # alongside it (how much of the corpus-spanning answer was actually retrieved).
    by_type: dict[str, list] = {}
    for item, r in results:
        by_type.setdefault(item.get("type", "single"), []).append(
            (int(r.metric_vals_dict["hit_rate"]), _anchor_coverage(item, r)))
    single = by_type.get("single", [])
    single_hit = sum(h for h, _ in single) / len(single) if single else 0.0

    # F5: the GATE is the single-type recall, not the strict+lenient blend.
    print(f"mode={mode} top_k={k_shown} | scored={n} | empty(withheld)={n_empty}")
    print(f"GATE (single recall) hit_rate={single_hit:.3f}  <- the pass/fail number (CLAUDE.md)\n")
    for t in ("single", "broad", "complex"):
        rowsT = by_type.get(t, [])
        if rowsT:
            hr = sum(h for h, _ in rowsT) / len(rowsT)
            cov = sum(c for _, c in rowsT) / len(rowsT)
            tag = "" if t == "single" else "  [lenient: any-hit; coverage = full-answer recall]"
            print(f"  {t:<8} hit_rate={hr:.3f}  coverage={cov:.3f}  ({len(rowsT)}){tag}")
    # F3: the ablation is not equal-k -- baseline/filtered score at TOP_K, rerank at RERANK_TOP_N,
    # so "rerank > filtered" mixes the reorder with a bigger window. For an apples-to-apples reader
    # run filtered at the same k: `RAG_FILTER_BY_CONTRACT=1 RAG_TOP_K=10 make eval MODE=filtered`.
    if RERANK and RERANK_TOP_N != TOP_K:
        print(f"\n  NOTE: rerank scores at k={RERANK_TOP_N} vs baseline/filtered at k={TOP_K} "
              f"(not equal-k). To isolate the reranker, compare filtered at RAG_TOP_K={RERANK_TOP_N}.")
    print()

    print(f"  {'hit':>3} {'cov':>4} {'mrr':>5}  type     intent/contract                query")
    for item, r in results:
        h = int(r.metric_vals_dict["hit_rate"])
        m = r.metric_vals_dict["mrr"]
        cov = _anchor_coverage(item, r)
        mark = " " if h else "*"
        print(f" {mark}{h:>3} {cov:>4.2f} {m:>5.2f}  {item.get('type', 'single'):<8} {item['intent']:<20} "
              f"{str(item['contract_id'] or '-'):<9} {item['query'][:38]}")

    misses = [(item, r) for item, r in results if r.metric_vals_dict["hit_rate"] == 0]
    if misses:
        print(f"\n{len(misses)} MISS(es):")
        for item, r in misses:
            print(f"  [{item.get('type')}/{item['contract_id']}/{item['intent']}] expected {len(item['expected_ids'])} ids")
            print(f"    retrieved: {r.retrieved_ids}")
