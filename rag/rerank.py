"""Step 7 of the RAG pipeline: CPU cross-encoder reranking.

Vector search (bge-small) has good recall only at large k -- correct chunks for abstract
queries like "which contractor was awarded X" land around rank 15-20. A cross-encoder
reranker scores each (query, chunk) pair directly and reorders, pulling the real answer
into the top few. Runs on CPU (zero VRAM).

``RerankingRetriever`` wraps a base retriever so it plugs straight into RetrieverEvaluator:
retrieve RERANK_CANDIDATES by vector similarity, then rerank down to RERANK_TOP_N.

The cross-encoder scores a LABELLED copy of each candidate ("Contract Agreement for contract
24CC0265, page 1: <text>"). It scores only the passage text, and many answers sit in chunks whose
text never says what they are -- a bare 'MAR 22 2024' stamp on a Notice of Award, a 63-char "made
this APR 01 2024" line, a notarial paragraph, a headerless BOQ table -- so a date/PCAB/CTC question
ranked them 10-37 of 40, past the top-10 cut. Labelled, 14 of 15 such misses rank 0-5. (rag.loader
now also labels the worst of these -- date stamps, "made this" lines, notarial register entries --
in the chunk text itself.) The rerank label is rerank-only: the embedding stays pure content (injecting metadata there homogenized vectors and
hurt recall) and the returned nodes are the originals, so the LLM context is unchanged.
"""

from __future__ import annotations

from functools import lru_cache

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from rag.config import RERANK_MODEL, RERANK_TOP_N

_DOC_LABEL = {
    "COA": "Contract Agreement", "CONTRACT_AGREEMENT": "Contract Agreement",
    "NOA": "Notice of Award", "NTP": "Notice to Proceed", "ROA": "Resolution of Award",
    "ADS": "Invitation to Bid", "SIGNATORIES": "Signatories",
}


def _labelled(candidate: NodeWithScore) -> NodeWithScore:
    """A copy of the candidate whose text leads with its document type, contract and page."""
    md = candidate.node.metadata
    doc = _DOC_LABEL.get(md.get("doc_type"), md.get("doc_type"))
    node = candidate.node.model_copy()
    node.text = f"{doc} for contract {md.get('contract_id')}, page {md.get('pdf_page')}: {node.text}"
    return NodeWithScore(node=node, score=candidate.score)


@lru_cache(maxsize=None)
def get_reranker(top_n: int = RERANK_TOP_N) -> SentenceTransformerRerank:
    """Cached per top_n: loading the cross-encoder is the expensive part, so any per-call site
    (e.g. a widen-k retry) reuses the loaded model instead of reloading it on CPU. The reranker is
    stateless across postprocess_nodes calls, so sharing one instance is safe."""
    return SentenceTransformerRerank(model=RERANK_MODEL, top_n=top_n)


class RerankingRetriever(BaseRetriever):
    """Base retriever + cross-encoder reranker, exposed as a single retriever.

    An optional ``trace`` (rag.trace.QueryTrace) records the vector candidates and the
    reranked survivors -- the seam where retrieval recall and rerank precision are visible.
    """

    def __init__(self, base_retriever: BaseRetriever, reranker: SentenceTransformerRerank,
                 trace=None):
        self._base_retriever = base_retriever
        self._reranker = reranker
        self._trace = trace
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        candidates = self._base_retriever.retrieve(query_bundle)
        originals = {c.node.node_id: c.node for c in candidates}
        ranked = self._reranker.postprocess_nodes([_labelled(c) for c in candidates],
                                                  query_bundle=query_bundle)
        reranked = [NodeWithScore(node=originals[n.node.node_id], score=n.score) for n in ranked]
        if self._trace is not None:
            self._trace.emit("retrieve", nodes=candidates, k=len(candidates),
                             query=query_bundle.query_str)
            self._trace.emit("rerank", nodes=reranked, top_n=len(reranked))
        return reranked
