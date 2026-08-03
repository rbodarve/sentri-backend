# Convenience targets for the RAG pipeline. Each delegates to scripts/.
# Usage: make setup | make check | make build | make eval [MODE=baseline|filtered|rerank] | make ask Q="..." | make chat | make all | make clean
.PHONY: setup check build eval ask chat all clean

setup:
	bash scripts/setup.sh

check:
	bash scripts/check.sh

build:
	bash scripts/build.sh

eval:
	bash scripts/evaluate.sh $(MODE)

ask:
	bash scripts/ask.sh "$(Q)"

chat:
	bash scripts/chat.sh

all: check build eval

# Remove the regenerable vector index (rebuild with `make build`) and caches.
clean:
	rm -rf index_store __pycache__ rag/__pycache__
