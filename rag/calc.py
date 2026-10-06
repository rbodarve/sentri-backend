"""Cited calculation (PLAN.md Phase E, E4). The LLM only picks operands and one op under an Ollama
`format` JSON schema; deterministic guards check each operand against its cited chunk; Python
computes in Decimal. Any failed check withholds: no LLM computes, and nothing is guessed."""

import html
import operator
import re
from dataclasses import dataclass
from decimal import Decimal
from functools import reduce

# The 4-op dispatch table over Decimal: no exec/eval, and no float anywhere on the path.
OPS = {"add": operator.add, "subtract": operator.sub, "multiply": operator.mul,
       "divide": operator.truediv}
_SYMBOL = {"add": "+", "subtract": "-", "multiply": "x", "divide": "/"}

_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PLAIN_RE = re.compile(r"\d+(?:\.\d+)?")
_BRACKET_RE = re.compile(r"\([^)]*\)")           # unit marks and abbreviations: "(P)", "(TCB)"
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and as for in is its of s the to".split())
_N_WORDS = {"both": 2, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10}
# A stated count: "the two ... items", "all 5 Part G items", "both bidders" (<= 5 words between).
_STATED_N_RE = re.compile(r"\b(\d+|" + "|".join(_N_WORDS) + r")\b(?:\s+\S+){0,5}?\s+(?:items|bidders)\b",
                          re.I)


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower())) - _STOP


def _plain(value: str) -> str:
    return value.replace(",", "").strip()


@dataclass
class Slot:
    """One value position of a chunk: its text, its column header ("" in a text chunk) and its row
    (the table row, or the sentence of a text chunk)."""

    text: str
    header: str
    row: str
    pos: int = 0    # the slot's index in its chunk: (chunk, pos) is one cell (row + column)


def doc_chunks(nodes: list) -> dict:
    """The retrieved DOCUMENT chunks by id -- the request's chunk enum. Leaves out the id note, the
    contract record and every other is_manifest node, and the derived signatory summaries."""
    return {n.node.node_id: n for n in nodes if not n.node.metadata.get("is_manifest")
            and n.node.metadata.get("category") != "signature_summary"}


def handles(chunks: dict) -> dict:
    """The request's chunk handles C1..Cn, one per document chunk in context order. A 3b model binds
    a short handle to its chunk; it mis-cited the 36-char UUIDs (E4 try 1)."""
    return {f"C{i}": n for i, n in enumerate(chunks.values(), 1)}


def schema(chunk_ids: list[str]) -> dict:
    """The Ollama `format` JSON schema: operands first, then op. A value is a string: no float."""
    operand = {"type": "object",
               "properties": {"label": {"type": "string"},
                              # Digits, commas and a point only (Ollama honours it: probe E4 try 2).
                              "value": {"type": "string", "pattern": r"^[0-9][0-9,]*(\.[0-9]+)?$"},
                              "chunk": {"type": "string", "enum": chunk_ids}},
               "required": ["label", "value", "chunk"]}
    return {"type": "object",
            "properties": {"operands": {"type": "array", "items": operand, "minItems": 2},
                           "op": {"type": "string", "enum": list(OPS)}},
            "required": ["operands", "op"]}


def slots(text: str) -> list[Slot]:
    """Every value position of a chunk: each table cell under its column header (the first row),
    or each numeric token of a text chunk in its sentence."""
    rows = [[html.unescape(c).strip() for c in _CELL_RE.findall(r)] for r in _ROW_RE.findall(text)]
    if rows:
        head = rows[0]
        found = [Slot(c, head[i] if i < len(head) else "", " ".join(r))
                 for r in rows[1:] for i, c in enumerate(r)]
    else:
        found = [Slot(m.group(0), "", s)
                 for s in _SENTENCE_RE.split(text) for m in _NUM_RE.finditer(s)]
    for pos, s in enumerate(found):
        s.pos = pos
    return found


def guard_binding(label: str, question: str) -> bool:
    """Every content word of the label appears in the calc clause: the operand is an item the
    question names. No exemption (the try-2 exemption let E20's item numbers through): an "all N
    items" clause binds only the operands whose labels it names (E5 CR4)."""
    return _words(label) <= _words(question)


def guard_value(value: str, found: list[Slot]) -> list[Slot]:
    """The slots whose WHOLE text equals the value, commas ignored: no forged or partial value."""
    plain = _plain(value)
    if not _PLAIN_RE.fullmatch(plain):
        return []
    return [s for s in found if _plain(s.text) == plain]


def guard_column(found: list[Slot], question: str) -> list[Slot]:
    """The table cells whose column header words (bracketed unit marks dropped) all appear in the
    question. A text-chunk slot has no column."""
    asked = _words(question)
    return [s for s in found if not s.header or _names_field(s.header, asked)]


def _names_field(header: str, asked: set[str]) -> bool:
    """A column header whose words (bracketed unit marks dropped) all appear in the asked words."""
    return _words(_BRACKET_RE.sub(" ", header)) <= asked


def guard_row(found: list[Slot], label: str) -> list[Slot]:
    """The slots whose row (table row, or text sentence) holds every label word that is not a word
    of the slot's column header."""
    def holds(s: Slot) -> bool:
        need = _words(label) - _words(s.header)
        return bool(need) and need <= _words(s.row)
    return [s for s in found if holds(s)]


def stated_count(question: str) -> int | None:
    """The stated N of "the two/three/both/<digit> ... items|bidders" (incl. "all N items"), else None."""
    m = _STATED_N_RE.search(question)
    if not m:
        return None
    return int(m.group(1)) if m.group(1).isdigit() else _N_WORDS[m.group(1).lower()]


def guard_count(operands: list, question: str) -> str:
    """A stated count: the request must return exactly N operands. A reason, or "" when it holds."""
    n = stated_count(question)
    if n is not None and len(operands) != n:
        return f"the question states {n} items, the request found {len(operands)}"
    return ""


def guard_duplicate(picked: list) -> str:
    """Two operands on the same cell (chunk, row, column) -> a reason (the #256 shape: one operand
    copied twice). The same row in two columns (E10) is two cells. "" when all differ."""
    cells = [(n.node.node_id, s.pos) for _, s, n in picked]
    return "two operands cite the same cell" if len(set(cells)) < len(cells) else ""


def order_operands(picked: list, order: tuple[str, ...]) -> list | None:
    """Subtract: put the operands in the order the strict pattern fixed (minuend, subtrahend). Each
    operand scores the words it shares with each order fragment; None when the two orders tie."""
    if len(picked) != 2 or len(order) != 2:
        return None
    seen = [_words(f"{label} {s.header} {s.row}") for label, s, _ in picked]
    frags = [_words(f) for f in order]
    straight = len(seen[0] & frags[0]) + len(seen[1] & frags[1])
    swapped = len(seen[0] & frags[1]) + len(seen[1] & frags[0])
    if straight == swapped:
        return None
    return picked if straight > swapped else picked[::-1]


def evaluate(op: str, values: list[Decimal]) -> Decimal | None:
    """The op over Decimal operands; None (withhold) on a wrong operand count or divide by zero."""
    if op not in OPS or len(values) < 2 or (op != "add" and len(values) != 2):
        return None
    if op == "divide" and values[1] == 0:
        return None
    return reduce(OPS[op], values)



def guard_text_field(picked: list, chunks: dict, question: str) -> str:
    """The text-field rule (the E05 hole): when the calc clause names a field that is a column header
    of a table chunk in this pass, an operand cited from a text chunk (no column) withholds -- a
    sentence cannot show which field its number is. A reason, or "" when it holds."""
    asked = _words(question)
    fields = {s.header for n in chunks.values() for s in slots(n.node.text)
              if s.header and _names_field(s.header, asked)}
    texts = [label for label, s, _ in picked if not s.header]
    if fields and texts:
        return (f"operand {texts[0]!r} is cited from text, but the question names the table field "
                f"{sorted(fields)[0]!r}")
    return ""


def check(request: dict, chunks: dict, question: str, op: str,
          order: tuple[str, ...] = ()) -> tuple[list, str]:
    """Run every guard on a parsed request. Returns (picked, "") -- picked = [(label, slot, node)]
    -- or ([], reason) to withhold. ``chunks`` is handles(doc_chunks()) of this pass; ``question``
    is the calc part."""
    operands = request.get("operands") or []
    if request.get("op") != op:
        return [], f"the request's op {request.get('op')!r} is not the question's {op!r}"
    if reason := guard_count(operands, question):
        return [], reason
    picked = []
    for o in operands:
        label, value, cid = (str(o.get(k, "")) for k in ("label", "value", "chunk"))
        node = chunks.get(cid)  # an unknown handle never falls back: withhold
        if node is None:
            return [], f"operand {label!r} cites {cid!r}, not a retrieved document chunk"
        if not guard_binding(label, question):
            return [], f"operand {label!r} is not an item the question names"
        found = guard_value(value, slots(node.node.text))
        if not found:
            return [], f"operand {label!r}: {value!r} is not a whole cell or number of its chunk"
        found = guard_column(found, question)
        if not found:
            return [], f"operand {label!r}: {value!r} is not in a column the question names"
        found = guard_row(found, label)
        if not found:
            return [], f"operand {label!r}: {value!r} is not in a row that names {label!r}"
        picked.append((label, found[0], node))
    if reason := guard_duplicate(picked) or guard_text_field(picked, chunks, question):
        return [], reason
    if op == "subtract":
        picked = order_operands(picked, order)
        if picked is None:
            return [], "the subtract order is ambiguous"
    return picked, ""


def _cited(label: str, s: Slot, n) -> str:
    return f"{s.text} ({label}; {n.node.metadata.get('pdf_source')} p.{n.node.metadata.get('pdf_page')})"


def cite(picked: list) -> str:
    """Option C: the guarded operands, one per line with citations; the total is not computed."""
    return "\n".join(_cited(*p) for p in picked) + "\nThe total is not computed."


def run_tape(picked: list, op: str) -> str:
    result = evaluate(op, [Decimal(_plain(s.text)) for _, s, _ in picked])
    if result is None:
        return ""
    return f" {_SYMBOL[op]} ".join(_cited(*p) for p in picked) + f" = {result:,.2f}"


def run(request: dict, chunks: dict, question: str, op: str,
        order: tuple[str, ...] = ()) -> tuple[str, str, list]:
    """DEAD under option C (E4-C, user 2026-10-06): kept with its pins, not called by the agent.
    check() + the Decimal evaluator. Returns (tape, "", cited nodes), or ("", reason, [])."""
    picked, reason = check(request, chunks, question, op, order)
    if reason:
        return "", reason, []
    tape = run_tape(picked, op)
    if not tape:
        return "", f"cannot {op} these {len(picked)} operands (divide by zero or operand count)", []
    return tape, "", [n for _, _, n in picked]
