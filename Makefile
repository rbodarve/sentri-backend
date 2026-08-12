# Convenience targets for the RAG pipeline. Each delegates to scripts/.
# Usage: make setup | make check | make build | make eval [MODE=baseline|filtered|rerank] | make ask Q="..." | make chat | make agent Q="..." | make chat-agentic | make eval-agentic | make serve | make all | make clean
.PHONY: help setup check coverage build eval ask chat agent chat-agentic eval-agentic serve serve-ngrok all clean

help:
	@echo "RAG pipeline commands (run via 'make <target>'):"
	@echo "  make setup                          Create conda env 'ragtest' + install pinned deps (one-time)"
	@echo "  make check                          Stage 1-3 self-checks (loader, enrich, relationships); no model"
	@echo "  make coverage                       Model-free field-coverage report over the manifest (analytical route's data ceiling)"
	@echo "  make build                          Build retrieval eval set + embed index + extract manifest (manifest step needs Ollama)"
	@echo "  make eval [MODE=baseline|filtered|rerank]  Measure retrieval recall (default rerank -> 1.000)"
	@echo "  make ask Q=\"...\"                     One-shot retrieve + rerank + generate a grounded answer (needs Ollama)"
	@echo "  make chat                           Interactive query loop (index + LLM loaded once; needs Ollama)"
	@echo "  make agent Q=\"...\"                   Agentic controller: router + fan-out + decomposition + self-correction (needs Ollama)"
	@echo "  make chat-agentic                   Interactive query loop routed through the agentic controller (needs Ollama)"
	@echo "  make eval-agentic                   Answer-level eval of the agent (needs Ollama)"
	@echo "  make serve [HOST=.. PORT=..]        Serve the agent as a streaming (SSE) HTTP API (needs Ollama)"
	@echo "  make serve-ngrok                    Serve the SSE API behind an ngrok public tunnel (needs Ollama + ngrok)"
	@echo "  make all                            check + build + eval"
	@echo "  make clean                          Remove the regenerable index_store/ + caches"
	@echo "  make help                           Show this list"

setup:
	bash scripts/setup.sh

check:
	bash scripts/check.sh

# Model-free field-coverage report over the built manifest (the analytical route's data ceiling).
coverage:
	bash scripts/coverage.sh

build:
	bash scripts/build.sh

eval:
	bash scripts/evaluate.sh $(MODE)

ask:
	bash scripts/ask.sh "$(Q)"

chat:
	bash scripts/chat.sh

# Agentic controller: router + fan-out + semantic decomposition + verifier-driven self-correction.
agent:
	bash scripts/agent.sh "$(Q)"

# Interactive query loop routed through the agentic controller (needs Ollama).
chat-agentic:
	bash scripts/chat_agentic.sh

# Answer-level eval for the agentic controller (needs Ollama; make eval stays the recall gate).
eval-agentic:
	bash scripts/evaluate_agentic.sh

# Serve the agentic controller as a streaming SSE HTTP API (needs Ollama).
serve:
	bash fastapi/serve.sh

# Same SSE API, exposed through an ngrok public tunnel (needs Ollama + ngrok).
serve-ngrok:
	bash fastapi/serve_ngrok.sh

all: check build eval

# Remove the regenerable vector index (rebuild with `make build`) and caches.
clean:
	rm -rf index_store __pycache__ rag/__pycache__
