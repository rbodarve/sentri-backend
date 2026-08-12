"""Step 6 of the RAG pipeline: measure retrieval recall (hit_rate) and MRR.

Runs the ground-truth set (eval/eval_retrieval.json) through the persisted index and reports
hit_rate (= recall@k: did any expected chunk appear in the top-k?) and MRR. Only the
embedding model runs -- no generation LLM -- so this is zero-VRAM.

By default it uses pure vector retrieval (no metadata filter): the honest test of whether
the Step-2 contract enrichment alone disambiguates. Set RAG_FILTER_BY_CONTRACT=1 to scope
each query to its contract via MetadataFilters (the documented fix for cross-contract misses).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from rag.config import RERANK_CANDIDATES, RERANK_TOP_N
from rag.index import load_index

EVAL_PATH = Path("eval/eval_retrieval.json")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
FILTER_BY_CONTRACT = os.getenv("RAG_FILTER_BY_CONTRACT", "0") == "1"
RERANK = os.getenv("RAG_RERANK", "0") == "1"


def evaluate(top_k: int = TOP_K, filter_by_contract: bool = FILTER_BY_CONTRACT, rerank: bool = RERANK):
    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    index = load_index()

    reranker = None
    if rerank:
        from rag.rerank import RerankingRetriever, get_reranker
        reranker = get_reranker(RERANK_TOP_N)

    # With reranking we retrieve a wide candidate set, then narrow to RERANK_TOP_N.
    candidate_k = RERANK_CANDIDATES if rerank else top_k

    results = []
    for item in dataset:
        # `empty` questions have no ground-truth chunk (the answer is absent by design); they are
        # withhold/negative tests, not recall probes, and RetrieverEvaluator needs expected_ids.
        if item.get("type") == "empty":
            continue
        # A contract filter only makes sense for single-contract items. Cross-contract broad/
        # complex questions (contract_id is null) span contracts, so never scope them.
        use_filter = filter_by_contract and item["contract_id"]
        if use_filter:
            filters = MetadataFilters(
                filters=[MetadataFilter(key="contract_id", operator=FilterOperator.EQ,
                                        value=item["contract_id"])]
            )
            retriever = index.as_retriever(similarity_top_k=candidate_k, filters=filters)
        else:
            retriever = index.as_retriever(similarity_top_k=candidate_k)
        if reranker is not None:
            retriever = RerankingRetriever(retriever, reranker)
        # with a contract filter, search on intent only: the item's pre-stripped search_query.
        # (the human-style `query` names the contract, which biases ranking toward header chunks.)
        query_text = item["search_query"] if use_filter else item["query"]
        evaluator = RetrieverEvaluator.from_metric_names(["hit_rate", "mrr"], retriever=retriever)
        result = evaluator.evaluate(query=query_text, expected_ids=item["expected_ids"])
        results.append((item, result))
    return results


if __name__ == "__main__":
    from collections import Counter

    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    n_empty = sum(1 for it in dataset if it.get("type") == "empty")

    results = evaluate()
    n = len(results)
    hit = sum(r.metric_vals_dict["hit_rate"] for _, r in results) / n
    mrr = sum(r.metric_vals_dict["mrr"] for _, r in results) / n

    mode = "contract-filtered" if FILTER_BY_CONTRACT else "pure-vector"
    if RERANK:
        mode += f" +rerank({RERANK_CANDIDATES}->{RERANK_TOP_N})"
    k_shown = RERANK_TOP_N if RERANK else TOP_K
    print(f"mode={mode} top_k={k_shown} | scored={n} | hit_rate={hit:.3f} | mrr={mrr:.3f}"
          f" | empty(withheld, not scored)={n_empty}\n")

    # Per-type recall: single is the headline recall gate; broad/complex are lenient (any
    # expected chunk in top-k counts), so their hit_rate reads high by construction.
    by_type: dict[str, list[int]] = {}
    for item, r in results:
        by_type.setdefault(item.get("type", "single"), []).append(int(r.metric_vals_dict["hit_rate"]))
    for t in ("single", "broad", "complex"):
        hits = by_type.get(t, [])
        if hits:
            print(f"  {t:<8} hit_rate={sum(hits) / len(hits):.3f}  ({sum(hits)}/{len(hits)})")
    print()

    print(f"  {'hit':>3} {'mrr':>5}  type     intent/contract                query")
    for item, r in results:
        h = int(r.metric_vals_dict["hit_rate"])
        m = r.metric_vals_dict["mrr"]
        mark = " " if h else "*"
        print(f" {mark}{h:>3} {m:>5.2f}  {item.get('type', 'single'):<8} {item['intent']:<20} "
              f"{str(item['contract_id'] or '-'):<9} {item['query'][:38]}")

    misses = [(item, r) for item, r in results if r.metric_vals_dict["hit_rate"] == 0]
    if misses:
        print(f"\n{len(misses)} MISS(es):")
        for item, r in misses:
            print(f"  [{item.get('type')}/{item['contract_id']}/{item['intent']}] expected {len(item['expected_ids'])} ids")
            print(f"    retrieved: {r.retrieved_ids}")
