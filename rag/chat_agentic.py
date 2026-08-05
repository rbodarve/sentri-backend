"""Interactive query loop over the agentic controller (rag.agent).

Same loop as rag.chat, but every question is routed through AgenticRag -- router + fan-out +
semantic decomposition + verifier-driven self-correction -- instead of the single-pass RAG.
Loads the index + reranker + LLM once. Exit with an empty line, 'exit'/'quit', or Ctrl-D.
"""

from __future__ import annotations

from rag.agent import AgenticRag, format_result
from rag.config import GEN_MODEL


def main() -> None:
    agent = AgenticRag()
    print(f"Agentic RAG chat ready ({GEN_MODEL}). Ask a question; empty line or Ctrl-D to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        print(f"\n{format_result(agent.answer(question))}")


if __name__ == "__main__":
    main()
