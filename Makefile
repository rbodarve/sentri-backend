# Convenience targets for the RAG pipeline. Each delegates to scripts/.
# Usage: make setup | make check | make build | make eval [MODE=baseline|filtered|rerank] | make ask Q="..." | make chat | make all
.PHONY: setup check build eval ask chat all

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
