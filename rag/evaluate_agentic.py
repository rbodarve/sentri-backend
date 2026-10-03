"""Answer-level eval for the agentic controller (rag.agent).

`rag.evaluate` scores *retrieval recall* on single-contract questions (zero-VRAM, no LLM) and
is the headline 1.000 gate. This companion scores what the agent adds -- routing, fan-out,
semantic decomposition, and verifier-driven withholding -- at the *answer* level, so it needs a
running Ollama (generation).

Each question in eval/eval_agentic.json declares its expected router "kind" and outcome:

  kind              -> the router decision this question must produce (simple|fanout|semantic|
                       analytical|enumerate|rank|aggregate). Checked FIRST, so a misroute fails even if the
                       answer happens to pass -- the gate covers routing, not just answer text.
                       `--routes` checks only this, model-free (no Ollama, no index).
  ordered=true      -> the anchors must also appear in the listed order (a ranking).
  expect="route"    -> pass = the route alone. For a question with no single right answer (a
                       per-contract follow-up asked without its contract) that must not misroute.
  expect="answer"   -> pass = the answer cleared the manifest Verifier AND contains every anchor
                       substring (case-insensitive). Anchors are short answer fragments, e.g. a
                       contractor or district-engineer surname.
  expect="withhold" -> pass = the agent withheld AND the report names the SPECIFIC trap declared in
                       "withhold_reason" (e.g. the invented id, or the mis-bound location). A
                       coincidental grounding failure on an unrelated token does not pass.

Pass is deliberately strict on grounding: an answer that is right but ungrounded, or grounded
but missing an expected fact, both fail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag.agent import AgenticRag, manifest_router

EVAL_PATH = Path("eval/eval_agentic.json")


def score(result, item) -> bool:
    if result.kind != item["kind"]:  # the router must land on the declared route (misroute = fail)
        return False
    if item["expect"] == "route":  # an ambiguous question with no ground-truth answer: routing only
        return True
    if item["expect"] == "withhold":
        # Not just "failed grounding" -- confirm the SPECIFIC adversarial trap was what got caught
        # (the invented id, or the mis-bound location/contractor), so a coincidental block/flag on
        # an unrelated token can't score as a correct withhold.
        if result.ok:
            return False
        reasons = " ".join(result.report.blocks + result.report.flags).lower()
        return all(r.lower() in reasons for r in item.get("withhold_reason", []))
    text = result.text.lower()
    if item.get("ordered"):  # a ranking: the anchors must also appear in the declared order
        pos = [text.find(a.lower()) for a in item["anchors"]]
        return result.ok and -1 not in pos and pos == sorted(pos)
    return result.ok and all(a.lower() in text for a in item["anchors"])


def check_routes(dataset) -> None:
    """Model-free routing gate (--routes): classify every question with the router on a manifest
    stub, as rag.evaluate does, so a misroute is caught without Ollama or the index."""
    from rag.manifest import load_manifest
    route = manifest_router(load_manifest())
    misroutes = [(item, got) for item in dataset
                 if (got := route(item["query"])[0]) != item["kind"]]
    print(f"routes: {len(dataset) - len(misroutes)}/{len(dataset)} questions -> declared kind")
    for item, got in misroutes:
        print(f"  MISROUTE expected {item['kind']} got {got}  {item['query']}")
    if misroutes:
        raise SystemExit(1)


def main() -> None:
    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    if "--routes" in sys.argv:
        return check_routes(dataset)
    agent = AgenticRag()

    rows = []
    for item in dataset:
        result = agent.answer(item["query"])
        rows.append((item, result, score(result, item)))

    n = len(rows)
    passed = sum(1 for _, _, ok in rows if ok)
    print(f"\nmode=agentic (answer-level) | questions={n} | pass_rate={passed / n:.3f}\n")
    print(f"  {'ok':>2}  {'route(exp/got)':<20} {'expect':<9} query")
    for item, result, ok in rows:
        route = item['kind'] if result.kind == item['kind'] else f"{item['kind']}->{result.kind}"
        print(f"  {'✓' if ok else '✗':>2}  {route:<20} {item['expect']:<9} {item['query'][:52]}")

    fails = [(item, result) for item, result, ok in rows if not ok]
    if fails:
        print(f"\n{len(fails)} FAIL(s):")
        for item, result in fails:
            misroute = "" if result.kind == item["kind"] else f" [MISROUTE: got {result.kind}]"
            print(f"  [{item['kind']}/{item['expect']}]{misroute} {item['query']}")
            print(f"    got (ok={result.ok}): {result.text[:160]!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
