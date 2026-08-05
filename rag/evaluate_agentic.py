"""Answer-level eval for the agentic controller (rag.agent).

`rag.evaluate` scores *retrieval recall* on single-contract questions (zero-VRAM, no LLM) and
is the headline 1.000 gate. This companion scores what the agent adds -- routing, fan-out,
semantic decomposition, and verifier-driven withholding -- at the *answer* level, so it needs a
running Ollama (generation).

Each question in eval/eval_agentic.json declares its expected outcome:

  expect="answer"   -> pass = the answer cleared the manifest Verifier AND contains every anchor
                       substring (case-insensitive). Anchors are short answer fragments, e.g. a
                       contractor or district-engineer surname.
  expect="withhold" -> pass = the agent correctly withheld (invented contract, or a mis-bound
                       location/contractor the Verifier caught). No anchors.

Pass is deliberately strict on grounding: an answer that is right but ungrounded, or grounded
but missing an expected fact, both fail.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.agent import AgenticRag

EVAL_PATH = Path("eval/eval_agentic.json")


def score(result, item) -> bool:
    if item["expect"] == "withhold":
        return not result.ok
    text = result.text.lower()
    return result.ok and all(a.lower() in text for a in item["anchors"])


def main() -> None:
    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    agent = AgenticRag()

    rows = []
    for item in dataset:
        result = agent.answer(item["query"])
        rows.append((item, result, score(result, item)))

    n = len(rows)
    passed = sum(1 for _, _, ok in rows if ok)
    print(f"\nmode=agentic (answer-level) | questions={n} | pass_rate={passed / n:.3f}\n")
    print(f"  {'ok':>2}  {'kind':<11} {'expect':<9} query")
    for item, result, ok in rows:
        print(f"  {'✓' if ok else '✗':>2}  {item['kind']:<11} {item['expect']:<9} {item['query'][:52]}")

    fails = [(item, result) for item, result, ok in rows if not ok]
    if fails:
        print(f"\n{len(fails)} FAIL(s):")
        for item, result in fails:
            print(f"  [{item['kind']}/{item['expect']}] {item['query']}")
            print(f"    got (ok={result.ok}): {result.text[:160]!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
