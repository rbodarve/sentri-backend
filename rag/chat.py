"""Interactive query loop over the RAG pipeline.

Loads the index + reranker + LLM once, then reads questions from a prompt and prints a
grounded, cited answer for each. Cheaper than repeated one-shot `rag.generate` calls,
which pay the model-load cost every invocation. Exit with an empty line, 'exit'/'quit',
or Ctrl-D.
"""

from __future__ import annotations

from rag.config import GEN_MODEL
from rag.generate import RagAnswerer, format_sources
from rag.verify import format_report


def main() -> None:
    rag = RagAnswerer()
    print(f"RAG chat ready ({GEN_MODEL}). Ask a question; empty line or Ctrl-D to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        response = rag.answer(question)
        report = rag.verify(response)
        if not report.ok:
            print(f"\n{format_report(report)}")
            continue
        print(f"\n{str(response).strip()}")
        print(f"sources: {format_sources(response)}")
        print(format_report(report))


if __name__ == "__main__":
    main()
