"""Step 8 of the RAG pipeline: Augmented Generation (the G in RAG).

Ties the whole pipeline together: retrieve (contract-filter + rerank -> top 10) supplies
the context, and a local LLM (Ollama, default granite4.1:3b, runs 100% on a 4GB GPU) writes
a grounded, cited answer. Retrieval searches on the intent-only query (contract id stripped,
as it biases ranking once we filter); the LLM sees the full question so it answers about the
right contract.

The contract filter engages whenever the question identifies a single contract: either by an
explicit id, or -- when none is written -- by resolving a distinctive location/name token from
the manifest back to its contract (so "signatories in Olongapo City" filters to 24CM0001
instead of searching all six contracts and drowning the right document in signature-heavy
pages from the others). Ambiguous questions (zero or several contracts) stay unfiltered.

Answers are constrained to the retrieved context to curb hallucination.
"""

from __future__ import annotations

import re

from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from rag.config import GEN_MODEL, RERANK_CANDIDATES, RERANK_TOP_N, get_llm
from rag.enrich import CONTRACT_ID_RE
from rag.evaluate import strip_contract_phrase
from rag.index import load_index
from rag.manifest import format_manifest, load_manifest
from rag.rerank import RerankingRetriever, get_reranker
from rag.trace import QueryTrace
from rag.verify import _GEO_STOP, Report, Verifier, format_report

_QA_TEMPLATE = PromptTemplate(
    "You answer questions about DPWH infrastructure-procurement documents.\n"
    "Use ONLY the context below. If the answer is not in the context, say you don't know.\n"
    "Be concise and name the source document(s).\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)


def _resolver_index(manifest: list[dict]) -> dict[str, set[str]]:
    """Distinctive location/name token (lowercased) -> set of contract_ids that use it.

    Shares the verifier's geo/boilerplate stoplist so "distinctive" means the same thing on
    both sides: a token that identifies a contract on the way in is exactly one the verifier
    would police on the way out. Tokens shorter than 4 chars or in the stoplist are dropped.
    """
    idx: dict[str, set[str]] = {}
    for r in manifest:
        text = f"{r.get('location', '')} {r.get('contract_name', '')}".lower()
        for tok in re.findall(r"[a-zñ]+", text):
            if len(tok) < 4 or tok in _GEO_STOP:
                continue
            idx.setdefault(tok, set()).add(r["contract_id"])
    return idx


class RagAnswerer:
    """Full RAG: retrieve+rerank the context, then generate a grounded answer."""

    def __init__(self):
        Settings.llm = get_llm()
        self._index = load_index()
        self._reranker = get_reranker(RERANK_TOP_N)
        self._synthesizer = get_response_synthesizer(
            response_mode="compact", text_qa_template=_QA_TEMPLATE
        )
        # The manifest (complete list of contracts) is injected into every query's context, so
        # global questions ("how many / list all / add the amounts") see the whole corpus, which
        # top-k retrieval alone never supplies. No intent classification -- it is always present.
        manifest = load_manifest()
        manifest_node = TextNode(text=format_manifest(manifest), metadata={"is_manifest": True})
        manifest_node.excluded_llm_metadata_keys = ["is_manifest"]
        self._manifest_node = NodeWithScore(node=manifest_node, score=1.0)
        # Maps a location/name token to its contract, so a question that names a project by place
        # or name (not id) can still engage the contract filter (see answer()).
        self._resolver = _resolver_index(manifest)
        # Final pre-send check: the same manifest is the oracle the answer is verified against.
        self._verifier = Verifier(manifest)

    def _retriever(self, contract_id: str | None, trace: QueryTrace | None = None):
        filters = None
        if contract_id:
            filters = MetadataFilters(
                filters=[MetadataFilter(key="contract_id", operator=FilterOperator.EQ,
                                        value=contract_id)]
            )
        base = self._index.as_retriever(similarity_top_k=RERANK_CANDIDATES, filters=filters)
        return RerankingRetriever(base, self._reranker, trace=trace)

    def _resolve_contract_id(self, question: str) -> str | None:
        """The single contract a question names by location/name, or None if zero or several.

        Requiring exactly one contract mirrors the verifier's rule: a location shared by two
        contracts, or two different locations in one question, is ambiguous -- leave retrieval
        unfiltered (the always-injected manifest still supplies the whole corpus)."""
        low = question.lower()
        ids: set[str] = set()
        for tok, owners in self._resolver.items():
            if re.search(rf"\b{re.escape(tok)}\b", low):
                ids |= owners
        return next(iter(ids)) if len(ids) == 1 else None

    def answer(self, question: str, eval_id: str | None = None,
               gold_ids: list[str] | None = None):
        # RAG_TRACE=1 records what enters/leaves each stage; disabled it is a no-op. gold_ids
        # (a question's eval expected_ids) enable per-stage 'gold survived?' -> stage-of-death.
        trace = QueryTrace(question, eval_id=eval_id, gold_ids=gold_ids)
        self._trace = trace
        match = CONTRACT_ID_RE.search(question)
        if match:
            contract_id = match.group(0).upper()
            search_query = strip_contract_phrase(question)
            route = "explicit_id"
        else:
            # No explicit id: try to identify the contract by the location/name it mentions.
            contract_id = self._resolve_contract_id(question)
            search_query = question
            route = "resolved_token" if contract_id else "none"
        trace.emit("route", route=route, contract_id=contract_id,
                   filtered=contract_id is not None, search_query=search_query)
        nodes = self._retriever(contract_id, trace).retrieve(search_query)
        context = [self._manifest_node, *nodes]
        trace.emit("context", nodes=context, manifest_included=True)
        response = self._synthesizer.synthesize(question, context)
        trace.emit("generate", answer=str(response).strip(),
                   source_node_ids=[n.node.node_id for n in response.source_nodes],
                   model=GEN_MODEL)
        return response

    def verify(self, response) -> Report:
        """Final grounding check on a synthesized answer, run before it is displayed."""
        report = self._verifier.check(str(response))
        self._trace.emit("verify", ok=report.ok, blocks=report.blocks, flags=report.flags,
                         sources=format_sources(response))
        return report


def format_sources(response) -> str:
    """Distinct, sorted 'contract/doc_type pN' citations for a synthesized answer."""
    return ", ".join(sorted({
        f"{n.metadata['contract_id']}/{n.metadata['doc_type']} p{n.metadata['pdf_page']}"
        for n in response.source_nodes if not n.metadata.get("is_manifest")
    }))


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    rag = RagAnswerer()
    # Map question -> expected_ids so RAG_TRACE=1 runs attach gold and stage-of-death works.
    eval_path = Path("eval/eval_set.json")
    gold_by_query = (
        {e["query"]: e["expected_ids"] for e in json.loads(eval_path.read_text())}
        if eval_path.exists() else {}
    )
    questions = sys.argv[1:] or [
        "Which contractor was awarded contract 24CC0265?",
        "What is the contract amount for 24AJ0052?",
        "Who is the District Engineer for contract 24BJ0005?",
    ]
    for q in questions:
        response = rag.answer(q, gold_ids=gold_by_query.get(q))
        report = rag.verify(response)
        print(f"\nQ: {q}")
        if not report.ok:
            print(format_report(report))
            continue
        print(f"A ({GEN_MODEL}): {str(response).strip()}")
        print(f"sources: {format_sources(response)}")
        print(format_report(report))
