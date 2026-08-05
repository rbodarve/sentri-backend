"""Step 6 of the RAG pipeline: measure retrieval recall (hit_rate) and MRR.

Runs the ground-truth set (eval/eval_set.json) through the persisted index and reports
hit_rate (= recall@k: did any expected chunk appear in the top-k?) and MRR. Only the
embedding model runs -- no generation LLM -- so this is zero-VRAM.

By default it uses pure vector retrieval (no metadata filter): the honest test of whether
the Step-2 contract enrichment alone disambiguates. Set RAG_FILTER_BY_CONTRACT=1 to scope
each query to its contract via MetadataFilters (the documented fix for cross-contract misses).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from rag.index import load_index

from rag.config import RERANK_CANDIDATES, RERANK_TOP_N

EVAL_PATH = Path("eval/eval_set.json")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
FILTER_BY_CONTRACT = os.getenv("RAG_FILTER_BY_CONTRACT", "0") == "1"
RERANK = os.getenv("RAG_RERANK", "0") == "1"


# When we scope retrieval with a contract filter, the contract id in the query text is
# redundant and biases ranking toward header chunks. Strip it so the semantic query is
# intent-only (mirrors a real pipeline: parse the id for the filter, search on the rest).
_CONTRACT_PHRASE_RE = re.compile(r"\s*(of\s+|for\s+)?contract\s+24[A-Za-z]{1,2}\d{4,5}", re.I)


def strip_contract_phrase(query: str) -> str:
    return _CONTRACT_PHRASE_RE.sub("", query).strip()


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
        if filter_by_contract:
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
        query_text = item["search_query"] if filter_by_contract else item["query"]
        evaluator = RetrieverEvaluator.from_metric_names(["hit_rate", "mrr"], retriever=retriever)
        result = evaluator.evaluate(query=query_text, expected_ids=item["expected_ids"])
        results.append((item, result))
    return results


if __name__ == "__main__":
    results = evaluate()
    n = len(results)
    hit = sum(r.metric_vals_dict["hit_rate"] for _, r in results) / n
    mrr = sum(r.metric_vals_dict["mrr"] for _, r in results) / n

    mode = "contract-filtered" if FILTER_BY_CONTRACT else "pure-vector"
    if RERANK:
        mode += f" +rerank({RERANK_CANDIDATES}->{RERANK_TOP_N})"
    k_shown = RERANK_TOP_N if RERANK else TOP_K
    print(f"mode={mode} top_k={k_shown} | questions={n} | hit_rate={hit:.3f} | mrr={mrr:.3f}\n")

    print(f"  {'hit':>3} {'mrr':>5}  intent/contract           query")
    for item, r in results:
        h = int(r.metric_vals_dict["hit_rate"])
        m = r.metric_vals_dict["mrr"]
        mark = " " if h else "*"
        print(f" {mark}{h:>3} {m:>5.2f}  {item['intent']:<16} {item['contract_id']:<9} {item['query'][:38]}")

    misses = [(item, r) for item, r in results if r.metric_vals_dict["hit_rate"] == 0]
    if misses:
        print(f"\n{len(misses)} MISS(es):")
        for item, r in misses:
            print(f"  [{item['contract_id']}/{item['intent']}] expected {len(item['expected_ids'])} ids")
            print(f"    retrieved: {r.retrieved_ids}")
