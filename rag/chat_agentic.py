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
        try:
            print(f"\n{format_result(agent.answer(question))}")
        except Exception as exc:  # a generation/Ollama error must not end the session
            print(f"\n[error answering that question: {exc}] -- try again or ask another.")


if __name__ == "__main__":
    main()
