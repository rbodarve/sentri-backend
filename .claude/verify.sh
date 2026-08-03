#!/bin/sh
# Corpus integrity check for the ragtest DPWH RAG data.
# The database/ JSONs are the authoritative OCR transcription; this only
# validates their structure (it never re-OCRs the source/ PDFs).
# Passes silently (exit 0); on any problem prints offenders and exits 1.
exec python3 - "$(dirname "$0")/.." <<'PY'
import json, sys, glob, os

# The four chunk categories in every task_*.json (all share one schema). Only 'text'
# is required non-empty -- every source has OCR text; table/image/signature are
# legitimately empty for many documents (a plan PDF has no signatures), so they are
# checked for content only when present.
SECTIONS = ("text", "table", "image", "signature")

root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "database", "*.json")))
if not files:
    print("FAIL: no database/*.json files found")
    sys.exit(1)

bad = []
for f in files:
    name = os.path.relpath(f, root)
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        bad.append(f"{name}: invalid JSON ({e})")
        continue
    if not isinstance(d, dict):
        bad.append(f"{name}: top level is {type(d).__name__}, expected object")
        continue
    text = d.get("text")
    if not isinstance(text, dict) or not text:
        bad.append(f"{name}: 'text' missing or empty (expected non-empty object of OCR chunks)")
        continue
    malformed = [s for s in SECTIONS if not isinstance(d.get(s, {}), dict)]
    if malformed:
        bad.append(f"{name}: section(s) not an object: {', '.join(malformed)}")
        continue
    empty, total = [], 0
    for s in SECTIONS:
        for k, v in d.get(s, {}).items():
            total += 1
            if not isinstance(v, dict) or not str(v.get("content", "")).strip():
                empty.append(f"{s}/{k}")
    if empty:
        bad.append(f"{name}: {len(empty)}/{total} chunks have empty 'content' "
                   f"({', '.join(empty[:3])}{', ...' if len(empty) > 3 else ''})")

if bad:
    print(f"FAIL: {len(bad)}/{len(files)} database files failed integrity check:")
    for b in bad:
        print("  -", b)
    sys.exit(1)

sys.exit(0)
PY
