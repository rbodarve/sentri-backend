"""Central, swappable configuration for the RAG pipeline.

The whole point of this module is the handoff: a team on different hardware changes the
embedding model here (or via environment variables) without touching pipeline code. The
model factory uses lazy imports so you only need the packages for the provider you pick.

Defaults: HuggingFace ``bge-small`` on CPU -- no VRAM, runs anywhere.
"""

from __future__ import annotations

import os

# --- swap points (override via env vars, or edit the defaults) -----------------------
EMBED_PROVIDER = os.getenv("RAG_EMBED_PROVIDER", "huggingface")  # "huggingface" | "ollama"
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
OLLAMA_BASE_URL = os.getenv("RAG_OLLAMA_BASE_URL", "http://localhost:11434")
PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "index_store")

# Reuse the shared HuggingFace hub cache instead of LlamaIndex's separate
# ~/.cache/llama_index, so bge-small isn't duplicated across projects. Honors HF_HOME.
HF_HUB_CACHE = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")

# Reranker (Step 7): CPU cross-encoder, zero VRAM. Retrieve RERANK_CANDIDATES, keep RERANK_TOP_N.
# 40 -> 10 measured at 100% recall on eval/eval_set.json (candidates and top_n must rise together).
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_CANDIDATES = int(os.getenv("RAG_RERANK_CANDIDATES", "40"))
RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "10"))

# Agentic controller (rag.agent): self-correction retry budget and the widened rerank top_n
# it falls back to. Deterministic router + verifier drive the loop; the LLM is used only for
# semantic decomposition. AGENT_WIDE_TOP_N keeps more reranked chunks per retry (candidates
# are already RERANK_CANDIDATES, so this only widens what reaches the LLM -- recall is unchanged).
AGENT_MAX_ATTEMPTS = int(os.getenv("RAG_AGENT_MAX_ATTEMPTS", "2"))
AGENT_WIDE_TOP_N = int(os.getenv("RAG_AGENT_WIDE_TOP_N", "20"))

# Generation (Step 8): local LLM. Default granite4.1:3b -- measured 100% GPU on a 4GB card
# (no RAM spill). Swap to a bigger model on better hardware without touching pipeline code.
GEN_PROVIDER = os.getenv("RAG_GEN_PROVIDER", "ollama")  # "ollama"
GEN_MODEL = os.getenv("RAG_GEN_MODEL", "granite4.1:3b")
GEN_TIMEOUT = float(os.getenv("RAG_GEN_TIMEOUT", "120"))
# Deterministic, factual output for extractive QA (curbs the small model rambling off-topic).
GEN_TEMPERATURE = float(os.getenv("RAG_GEN_TEMPERATURE", "0.0"))
# Cap the context window: LlamaIndex otherwise defaults Ollama to the model's full window
# (131072 for granite4.1), which balloons the KV cache to ~21GB and spills to RAM. 4096 is
# measured at 2.7GB / 100% GPU and easily holds the top-10 retrieved chunks.
GEN_NUM_CTX = int(os.getenv("RAG_GEN_NUM_CTX", "4096"))


def get_embed_model():
    """Instantiate the configured embedding model. Imported lazily per provider."""
    if EMBED_PROVIDER == "huggingface":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=EMBED_MODEL, cache_folder=HF_HUB_CACHE)
    if EMBED_PROVIDER == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    raise ValueError(
        f"Unknown RAG_EMBED_PROVIDER={EMBED_PROVIDER!r}; expected 'huggingface' or 'ollama'"
    )


def get_llm():
    """Instantiate the configured generation LLM. Imported lazily per provider."""
    if GEN_PROVIDER == "ollama":
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=GEN_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=GEN_TIMEOUT,
            context_window=GEN_NUM_CTX,
            temperature=GEN_TEMPERATURE,
            # granite emits fill-in-middle sentinels then fails to stop; halt on them so the
            # output is just the concise answer. Harmless for other models (never emitted).
            additional_kwargs={"stop": ["<|fim_middle|>", "<|fim_prefix|>",
                                        "<|fim_suffix|>", "<|endoftext|>"]},
        )
    raise ValueError(f"Unknown RAG_GEN_PROVIDER={GEN_PROVIDER!r}; expected 'ollama'")
