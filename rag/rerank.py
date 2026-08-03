"""Step 7 of the RAG pipeline: CPU cross-encoder reranking.

Vector search (bge-small) has good recall only at large k -- correct chunks for abstract
queries like "which contractor was awarded X" land around rank 15-20. A cross-encoder
reranker scores each (query, chunk) pair directly and reorders, pulling the real answer
into the top few. Runs on CPU (zero VRAM).

``RerankingRetriever`` wraps a base retriever so it plugs straight into RetrieverEvaluator:
retrieve RERANK_CANDIDATES by vector similarity, then rerank down to RERANK_TOP_N.
"""

from __future__ import annotations

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from rag.config import RERANK_MODEL, RERANK_TOP_N


def get_reranker(top_n: int = RERANK_TOP_N) -> SentenceTransformerRerank:
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
        reranked = self._reranker.postprocess_nodes(candidates, query_bundle=query_bundle)
        if self._trace is not None:
            self._trace.emit("retrieve", nodes=candidates, k=len(candidates),
                             query=query_bundle.query_str)
            self._trace.emit("rerank", nodes=reranked, top_n=len(reranked))
        return reranked
