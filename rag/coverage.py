"""Preflight: a model-free field-coverage report over the corpus manifest.

Field coverage is a property of the DATA, not the model: which manifest fields actually carry a
value versus the "not stated" sentinel, and which doc_types each contract holds. It is computed
with zero model calls (pure dict/count), the same way rag.relationships reasons about the
~4%-populated reference_above/below fields -- it never re-OCRs and never calls the LLM.

Why it exists: an analytical/pattern question (rag.agent's "analytical" route) can only surface a
pattern over a field that is actually populated across the corpus. This report states the CEILING
-- what is even findable -- so a failed analytical answer is diagnosable as a DATA ceiling (widen
the extracted fields in rag.manifest._EXTRACT_FIELDS) rather than a MODEL ceiling (swap the model
/ check the eval). The two are separate assessments: coverage here is deterministic and offline;
whether a given model reaches this ceiling is measured separately by eval/eval_agentic.json.
"""

from __future__ import annotations

from rag.manifest import _EXTRACT_FIELDS, load_manifest

# The sentinel rag.manifest writes when a field is absent from the OCR (see its extraction prompt).
_ABSENT = "not stated"


def _is_present(value) -> bool:
    """A field is populated when it holds a real value: a non-empty list, or a non-empty string
    that is not the 'not stated' sentinel rag.manifest uses for an absent field."""
    if isinstance(value, list):
        return bool(value)
    return bool(value) and str(value).strip().lower() != _ABSENT


def field_coverage(manifest: list[dict]) -> dict[str, dict]:
    """Per extracted field: how many contracts carry a real value, and which lack it."""
    total = len(manifest)
    out: dict[str, dict] = {}
    for name in _EXTRACT_FIELDS:
        missing = [r["contract_id"] for r in manifest if not _is_present(r.get(name))]
        out[name] = {"present": total - len(missing), "total": total, "missing": missing}
    return out


def doc_type_coverage(manifest: list[dict]) -> dict[str, list[str]]:
    """Per doc_type: which contracts hold at least one document of that type -- i.e. which source
    signals even exist for extraction to draw a field from."""
    out: dict[str, list[str]] = {}
    for r in manifest:
        for dt in r.get("doc_types", []):
            out.setdefault(dt, []).append(r["contract_id"])
    return out


def _verdict(present: int, total: int) -> str:
    """A field's analytical ceiling. FINDABLE: populated for every contract, patterns are fully
    supported. PARTIAL: populated for some -- it both supports cross-contract patterns AND its
    absentees are themselves anomalies (e.g. a contract with no amount is an outlier). NONE: a
    DATA ceiling -- no model can surface a pattern over a field that was never extracted."""
    if present == 0:
        return "not extractable (DATA ceiling: no model can find a pattern here)"
    if present < total:
        return "partial (patterns findable; absentees are themselves anomalies)"
    return "findable"


def format_report(manifest: list[dict]) -> str:
    total = len(manifest)
    lines = [
        f"CORPUS FIELD COVERAGE -- {total} contracts (model-free; the ceiling of what the "
        "analytical route can surface).",
        "",
        "Extracted fields (LLM-extracted in rag.manifest):",
    ]
    fields = field_coverage(manifest)
    width = max(len(name) for name in fields)
    for name, cov in fields.items():
        line = f"  {name:<{width}}  {cov['present']}/{cov['total']}  -- {_verdict(cov['present'], cov['total'])}"
        if cov["missing"]:
            line += f"  [missing: {', '.join(sorted(cov['missing']))}]"
        lines.append(line)

    lines += ["", "Document types present (source signals available to extract from):"]
    docs = doc_type_coverage(manifest)
    dwidth = max((len(dt) for dt in docs), default=0)
    for dt, ids in sorted(docs.items()):
        lines.append(f"  {dt:<{dwidth}}  {len(ids)}  ({', '.join(sorted(ids))})")

    lines += [
        "",
        "How to read this:",
        "  - Patterns/anomalies are findable ONLY over the fields above. A field at 0/N is a DATA",
        "    ceiling -- add it to rag.manifest._EXTRACT_FIELDS to raise the ceiling; a stronger",
        "    model cannot surface a signal that was never placed in its context.",
        "  - Whether the model actually reaches this ceiling is a separate, model-specific check:",
        "    see eval/eval_agentic.json (the analytical rows), not this report.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_report(load_manifest()))
