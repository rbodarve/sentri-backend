"""Query-path tracer: records what goes in and out of each retrieval stage.

One question produces an ordered stream of JSONL records sharing a ``query_id`` -- one
per stage (intake, route, retrieve, rerank, context, generate, verify). The trace records
node *ids, scores and metadata* (never chunk text or vectors) so a run is machine-diffable
and quiet. Gated by ``RAG_TRACE=1``: off means zero records and near-zero overhead.

The payoff is *stage-of-death*: given a question's ground-truth chunk ids (``eval_set.json``
``expected_ids``, the same UUID space as ``node.node_id``), :func:`stage_of_death` reports the
first stage where the gold chunk disappears -- retrieval miss vs rerank drop vs generation --
so a failing eval question points straight at the component to fix.

Set ``RAG_TRACE_CONTENT=1`` to also record a truncated text snippet per node.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

_TRACE_DIR = Path(os.getenv("RAG_PERSIST_DIR", "index_store")) / "trace"
_STAGES = ("intake", "route", "retrieve", "rerank", "context", "generate", "verify")


def _enabled() -> bool:
    return os.getenv("RAG_TRACE") == "1"


def _node_summary(node_with_score) -> dict:
    """Decision-relevant fields for one NodeWithScore: id, score, source metadata."""
    node = node_with_score.node
    meta = node.metadata
    score = node_with_score.score
    summary = {
        "node_id": node.node_id,
        "score": None if score is None else float(score),  # numpy float32 -> JSON-safe
        "contract_id": meta.get("contract_id"),
        "doc_type": meta.get("doc_type"),
        "pdf_page": meta.get("pdf_page"),
        "is_manifest": bool(meta.get("is_manifest")),
    }
    if os.getenv("RAG_TRACE_CONTENT") == "1":
        summary["snippet"] = node.get_content()[:200]
    return summary


class QueryTrace:
    """Per-query trace context. Emits one JSONL record per stage to the trace dir.

    A disabled trace (``RAG_TRACE`` unset) is a no-op: :meth:`emit` returns immediately and
    no file is opened, so instrumentation left in the hot path costs nothing in production.
    """

    def __init__(self, question: str, eval_id: str | None = None,
                 gold_ids: list[str] | None = None):
        self.enabled = _enabled()
        self.gold = set(gold_ids or [])
        self._seq = 0
        stamp = datetime.now(timezone.utc)
        digest = hashlib.sha1(f"{question}{stamp.isoformat()}".encode()).hexdigest()[:12]
        self.query_id = digest
        if self.enabled:
            _TRACE_DIR.mkdir(parents=True, exist_ok=True)
            self._path = _TRACE_DIR / f"{self.query_id}.jsonl"
            self._path.write_text("", encoding="utf-8")  # fresh file per query
        self.emit("intake", question=question, eval_id=eval_id,
                  gold_node_ids=sorted(self.gold))

    def _gold_view(self, nodes: list[dict]) -> dict:
        """Whether the gold chunk survived to this stage, and at what rank (1-based)."""
        if not self.gold:
            return {"gold_present": None, "gold_rank": None}
        for rank, n in enumerate(nodes, start=1):
            if n["node_id"] in self.gold:
                return {"gold_present": True, "gold_rank": rank}
        return {"gold_present": False, "gold_rank": None}

    def emit(self, stage: str, *, nodes: list | None = None, **fields) -> None:
        """Append one record for ``stage``. ``nodes`` (NodeWithScore list) is summarised."""
        if not self.enabled:
            return
        assert stage in _STAGES, f"unknown trace stage {stage!r}"
        record = {
            "query_id": self.query_id,
            "seq": self._seq,
            "stage": stage,
            "ts": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        if nodes is not None:
            summaries = [_node_summary(n) for n in nodes]
            record["count"] = len(summaries)
            record["nodes"] = summaries
            record.update(self._gold_view(summaries))
        self._seq += 1
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def stage_of_death(records: list[dict]) -> dict:
    """Diagnose the first stage where the gold chunk disappears from a query's records.

    Returns ``{"query_id", "gold": bool, "verdict", "last_gold_stage"}``. ``verdict`` is one of:
    retrieval-miss, rerank-drop, generation, grounding, or ok.
    """
    by_stage = {r["stage"]: r for r in records}
    query_id = records[0]["query_id"] if records else None
    intake = by_stage.get("intake", {})
    if not intake.get("gold_node_ids"):
        return {"query_id": query_id, "gold": False, "verdict": "no-gold",
                "last_gold_stage": None}

    last_gold_stage = None
    for stage in ("retrieve", "rerank"):
        rec = by_stage.get(stage)
        if rec is None:
            continue
        if rec.get("gold_present"):
            last_gold_stage = stage
        elif last_gold_stage is None:
            return {"query_id": query_id, "gold": True, "verdict": "retrieval-miss",
                    "last_gold_stage": None}
        else:
            return {"query_id": query_id, "gold": True, "verdict": "rerank-drop",
                    "last_gold_stage": last_gold_stage}

    verify = by_stage.get("verify", {})
    if verify and not verify.get("ok", True):
        return {"query_id": query_id, "gold": True, "verdict": "grounding",
                "last_gold_stage": last_gold_stage}
    return {"query_id": query_id, "gold": True, "verdict": "ok",
            "last_gold_stage": last_gold_stage}


def read_trace(query_id: str) -> list[dict]:
    """Load one query's JSONL records in order."""
    path = _TRACE_DIR / f"{query_id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    # Summarise every trace file: query, verdict, and where gold was last seen.
    files = sorted(_TRACE_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no traces in {_TRACE_DIR} (run with RAG_TRACE=1 first)")
    print(f"{'query_id':<14}{'verdict':<16}{'last_gold':<10}question")
    for path in files:
        records = read_trace(path.stem)
        verdict = stage_of_death(records)
        question = next((r.get("question") for r in records if r["stage"] == "intake"), "")
        print(f"  {verdict['query_id']:<12}  {verdict['verdict']:<16}"
              f"{str(verdict['last_gold_stage'] or '-'):<10}{question}")
