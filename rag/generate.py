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
import unicodedata

from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.postprocessor import PrevNextNodePostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from rag.config import GEN_MODEL, RERANK_CANDIDATES, RERANK_TOP_N, get_llm
from rag.enrich import CONTRACT_ID_RE, strip_contract_phrase
from rag.index import load_index
from rag.manifest import format_manifest, load_manifest
from rag.rerank import RerankingRetriever, get_reranker
from rag.trace import QueryTrace
from rag.verify import _GEO_STOP, Report, Verifier, format_report, _extract_sig_names

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


def _build_person_names(docstore) -> dict[str, set[str]]:
    """Extract person names from signature nodes in the full corpus.

    Iterates every signature and signature_summary node in the docstore and applies
    the same all-caps name extraction used by the verifier, keyed by contract_id.
    Called once at startup; the result is passed to Verifier so the grounding check
    knows which persons are legitimate signatories for each contract."""
    result: dict[str, set[str]] = {}
    for node in docstore.docs.values():
        md = node.metadata
        if md.get("category") not in ("signature", "signature_summary"):
            continue
        cid = md.get("contract_id", "").upper()
        if not cid:
            continue
        result.setdefault(cid, set()).update(_extract_sig_names(node.text))
    return result


def _norm(text: str) -> str:
    """Lowercase and strip diacritics (ñ->n) so entity resolution does not turn on a dropped
    tilde or an accented vowel ("macanao" must match "macañao")."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _resolver_index(manifest: list[dict]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(token index, phrase index): a distinctive location/name key -> contract_ids that use it.

    Both sides of the match are normalized (lowercased, diacritics stripped) so a dropped tilde
    still resolves. The token index holds single words (word-boundary matched); the phrase index
    holds despaced adjacent word pairs from the location ("launion" for "la union") so a
    multi-word place written without its space resolves too, instead of failing open to the whole
    corpus. Shares the verifier's geo/boilerplate stoplist and >=4-char guard so "distinctive"
    means the same thing on both sides: a token that identifies a contract on the way in is
    exactly one the verifier would police on the way out.
    """
    tokens: dict[str, set[str]] = {}
    phrases: dict[str, set[str]] = {}
    for r in manifest:
        cid = r["contract_id"]
        combined = _norm(f"{r.get('location', '')} {r.get('contract_name', '')}")
        for tok in re.findall(r"[a-z]+", combined):
            if len(tok) < 4 or tok in _GEO_STOP:
                continue
            tokens.setdefault(tok, set()).add(cid)
        loc_words = re.findall(r"[a-z]+", _norm(r.get("location", "")))
        for a, b in zip(loc_words, loc_words[1:]):
            joined = a + b
            if len(joined) >= 6:  # distinctive enough to substring-match safely
                phrases.setdefault(joined, set()).add(cid)
    return tokens, phrases


def build_retriever(index, contract_id, candidate_k, reranker=None, trace=None):
    """Filtered vector retriever, optionally rerank-wrapped -- the SINGLE construction shared by
    generation (RagAnswerer._retriever) and the recall gate (rag.evaluate), so the two cannot
    drift. contract_id None => unfiltered; reranker None => bare vector retriever (the ablation
    modes). Keeping one builder means a change to the production retriever is measured by the gate
    automatically, instead of the gate scoring a hand-copied replica."""
    filters = None
    if contract_id:
        filters = MetadataFilters(
            filters=[MetadataFilter(key="contract_id", operator=FilterOperator.EQ,
                                    value=contract_id)]
        )
    base = index.as_retriever(similarity_top_k=candidate_k, filters=filters)
    if reranker is None:
        return base
    return RerankingRetriever(base, reranker, trace=trace)


def resolve_contract_ids(question, resolver, resolver_phrases) -> set[str]:
    """All contracts a question names by a distinctive location/name token. Module-level so the
    recall gate can test the NL->filter step with the SAME logic production uses (not a hand-copy);
    matching is diacritic/spacing-robust (see _resolver_index)."""
    norm = _norm(question)
    ids: set[str] = set()
    for tok, owners in resolver.items():
        if re.search(rf"\b{re.escape(tok)}\b", norm):
            ids |= owners
    despaced = re.sub(r"[^a-z]", "", norm)
    for phrase, owners in resolver_phrases.items():
        if phrase in despaced:
            ids |= owners
    return ids


def route_contract_id(question, resolver, resolver_phrases) -> str | None:
    """The single contract a question filters to in production: explicit id wins, else exactly one
    distinctive resolved token, else None. The id-resolution half of RagAnswerer.route, module-
    level so rag.evaluate can gate the NL->filter step without loading the model."""
    match = CONTRACT_ID_RE.search(question)
    if match:
        return match.group(0).upper()
    ids = resolve_contract_ids(question, resolver, resolver_phrases)
    return next(iter(ids)) if len(ids) == 1 else None


class RagAnswerer:
    """Full RAG: retrieve+rerank the context, then generate a grounded answer."""

    def __init__(self):
        Settings.llm = get_llm()
        self._index = load_index()
        self._reranker = get_reranker(RERANK_TOP_N)
        # Reading-order (PREV/NEXT) neighbour expansion (Step 3, rag.relationships): after
        # rerank, pull each hit's immediate neighbours from the docstore so a chunk that lands
        # mid-block (a table row without its header, an answer split across two chunks) reaches
        # the LLM with its adjacent context. mode="both"/num_nodes=1 is the minimal expansion;
        # the postprocessor dedups shared neighbours and re-sorts into reading order. Applied only
        # on the generation path (not the recall retriever), so `make eval` is unaffected.
        self._neighbors = PrevNextNodePostprocessor(
            docstore=self._index.storage_context.docstore, num_nodes=1, mode="both"
        )
        self._synthesizer = get_response_synthesizer(
            response_mode="compact", text_qa_template=_QA_TEMPLATE
        )
        # The manifest (complete list of contracts) is injected into every query's context, so
        # global questions ("how many / list all / add the amounts") see the whole corpus, which
        # top-k retrieval alone never supplies. No intent classification -- it is always present.
        manifest = load_manifest()
        self._manifest = manifest  # the full record list, for deterministic enumeration (agent)
        manifest_node = TextNode(text=format_manifest(manifest), metadata={"is_manifest": True})
        manifest_node.excluded_llm_metadata_keys = ["is_manifest"]
        self._manifest_node = NodeWithScore(node=manifest_node, score=1.0)
        # Maps a location/name token (and despaced multi-word place phrase) to its contract, so a
        # question that names a project by place or name (not id) can still engage the contract
        # filter (see answer()).
        self._resolver, self._resolver_phrases = _resolver_index(manifest)
        # Final pre-send check: the same manifest is the oracle; person_names supplies the
        # corpus-wide signatory map so the grounding check can detect person-contract
        # fabrications (e.g. a signatory from contract A falsely attributed to contract B).
        person_names = _build_person_names(self._index.storage_context.docstore)
        self._verifier = Verifier(manifest, person_names=person_names)
        # answer() stashes the live trace here; default it so verify() can be called standalone
        # (or after answer_once) and degrade to no-trace instead of raising AttributeError.
        self._trace: QueryTrace | None = None

    # -- public surface for the agentic layer (rag.agent) ---------------------------------
    # The agent reuses (never replaces) this pipeline, but it needs a few of these internals.
    # Exposing them as a public contract means a refactor of the privates can't silently break
    # the agent -- these accessors are the pinned interface.
    @property
    def verifier(self) -> Verifier:
        """The manifest grounding oracle (same instance the pipeline verifies against)."""
        return self._verifier

    @property
    def contract_ids(self) -> set[str]:
        """The real contract ids in the corpus -- guards against filtering on an invented one."""
        return {r["contract_id"] for r in self._manifest}

    @property
    def manifest(self) -> list[dict]:
        """The complete manifest: one structured record per contract."""
        return self._manifest

    @property
    def manifest_text(self) -> str:
        """The manifest rendered as the context-injection table (the analytical route's substrate)."""
        return self._manifest_node.node.text

    def resolve_contract_ids(self, question: str) -> set[str]:
        """All contracts a question names by a distinctive location/name token (see route())."""
        return self._resolve_contract_ids(question)

    def _retriever(self, contract_id: str | None, trace: QueryTrace | None = None,
                   reranker=None):
        return build_retriever(self._index, contract_id, RERANK_CANDIDATES,
                               reranker or self._reranker, trace)

    def _id_node(self, contract_id: str) -> NodeWithScore:
        """An authoritative context note carrying the resolved contract id, so the answer prints
        the real id (known from routing) instead of fabricating a 'Contract ID' from the project
        title. Flagged is_manifest so it is excluded from citations/evidence like the manifest."""
        node = TextNode(
            text=(f"AUTHORITATIVE: the contract in question is contract id {contract_id}. "
                  f'Use exactly "{contract_id}" for any Contract ID field; never use the '
                  "project name, location, or title as the contract id."),
            metadata={"is_manifest": True},
        )
        node.excluded_llm_metadata_keys = ["is_manifest"]
        return NodeWithScore(node=node, score=1.0)

    def _resolve_contract_ids(self, question: str) -> set[str]:
        """All contracts a question names by a distinctive location/name token.

        Matching is diacritic- and spacing-robust: single tokens are matched by word boundary
        against the normalized question, and despaced multi-word place phrases ("launion") as a
        substring of the despaced question, so a missing space or dropped tilde still resolves
        instead of leaving the filter to fail open to the whole corpus. Delegates to the
        module-level resolve_contract_ids so the recall gate shares this exact logic."""
        return resolve_contract_ids(question, self._resolver, self._resolver_phrases)

    def _resolve_contract_id(self, question: str) -> str | None:
        """The single contract a question names by location/name, or None if zero or several.

        Requiring exactly one contract mirrors the verifier's rule: a location shared by two
        contracts, or two different locations in one question, is ambiguous -- leave retrieval
        unfiltered (the always-injected manifest still supplies the whole corpus)."""
        ids = self._resolve_contract_ids(question)
        return next(iter(ids)) if len(ids) == 1 else None

    def route(self, question: str) -> tuple[str | None, str, str]:
        """Resolve (contract_id, intent-only search_query, route-name) for a question.

        Explicit id wins (and is stripped from the search text, as it biases ranking once we
        filter); else fall back to a distinctive location/name token; else stay unfiltered."""
        match = CONTRACT_ID_RE.search(question)
        if match:
            return match.group(0).upper(), strip_contract_phrase(question), "explicit_id"
        contract_id = self._resolve_contract_id(question)
        return contract_id, question, "resolved_token" if contract_id else "none"

    def answer_once(self, question: str, contract_id: str | None, search_query: str,
                    reranker=None, trace: QueryTrace | None = None):
        """Single filtered+reranked retrieval pass, manifest-injected, then synthesized.

        The reusable core of answer(): the agent (rag.agent) calls it per sub-question and can
        pass a wider ``reranker`` to widen k during self-correction."""
        nodes = self._retriever(contract_id, trace, reranker).retrieve(search_query)
        nodes = self._neighbors.postprocess_nodes(nodes)
        context = [self._manifest_node, *nodes]
        if contract_id:
            # Surface the resolved id to generation so the answer cites it, not the project title.
            context = [self._id_node(contract_id), *context]
        if trace is not None:
            trace.emit("context", nodes=context, manifest_included=True)
        response = self._synthesizer.synthesize(question, context)
        if trace is not None:
            trace.emit("generate", answer=str(response).strip(),
                       source_node_ids=[n.node.node_id for n in response.source_nodes],
                       model=GEN_MODEL)
        return response

    def answer(self, question: str, eval_id: str | None = None,
               gold_ids: list[str] | None = None):
        # RAG_TRACE=1 records what enters/leaves each stage; disabled it is a no-op. gold_ids
        # (a question's eval expected_ids) enable per-stage 'gold survived?' -> stage-of-death.
        trace = QueryTrace(question, eval_id=eval_id, gold_ids=gold_ids)
        self._trace = trace
        contract_id, search_query, route = self.route(question)
        trace.emit("route", route=route, contract_id=contract_id,
                   filtered=contract_id is not None, search_query=search_query)
        return self.answer_once(question, contract_id, search_query, trace=trace)

    def verify(self, response) -> Report:
        """Final grounding check on a synthesized answer, run before it is displayed."""
        report = self._verifier.check(str(response))
        if self._trace is not None:  # answer() sets the trace; standalone verify() has none
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
    eval_path = Path("eval/eval_retrieval.json")
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
