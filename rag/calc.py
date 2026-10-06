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
_TOTAL_COST_RE = re.compile(r"\btotal\s+costs?\b", re.I)


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower())) - _STOP


def _field_words(question: str) -> set[str]:
    """The words a question names fields with. G-synonym (C2): the adjacent phrase "total cost(s)"
    also names {total, amount} (the BoQ "Total Amount (P)"). One phrase, not a word map: "the
    total of the unit costs" names no Total Amount."""
    asked = _words(question)
    return asked | {"total", "amount"} if _TOTAL_COST_RE.search(question) else asked


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
    cells: tuple = ()  # the table row's cells (() in a text chunk): the row's item, for G7


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
    rows = _rows(text)
    if rows:
        head = rows[0]
        found = [Slot(c, head[i] if i < len(head) else "", " ".join(r), cells=tuple(r))
                 for r in rows[1:] for i, c in enumerate(r)]
    else:
        found = [Slot(m.group(0), "", s)
                 for s in _SENTENCE_RE.split(text) for m in _NUM_RE.finditer(s)]
    for pos, s in enumerate(found):
        s.pos = pos
    return found


def _seq(text: str) -> list[str]:
    """The content words of a text, in order."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP]


def _held(text: str, have: str, skip: frozenset | set = frozenset()) -> bool:
    """THE word matcher (binding, row guard, G7's named-row count): every content word of ``text``
    not in ``skip`` is a word of ``have``. G-glue (C7, OCR glue): or the word joined to its ADJACENT
    word of ``text`` is a word of ``have`` ("project" + "billboard" = "projectbillboard"), or it is
    two ADJACENT words of ``have`` joined. Adjacent words only, no other merges."""
    need, seen = _WORD_RE.findall(text.lower()), _WORD_RE.findall(have.lower())
    words, glued = set(seen), {a + b for a, b in zip(seen, seen[1:])}
    for i, w in enumerate(need):
        if w in _STOP or w in skip or w in words or w in glued:
            continue
        if (i and need[i - 1] + w in words) or (i + 1 < len(need) and w + need[i + 1] in words):
            continue
        return False
    return True


def _rows(text: str) -> list[list[str]]:
    """The rows of a table chunk as cell lists, header first; [] for text."""
    return [[html.unescape(c).strip() for c in _CELL_RE.findall(r)] for r in _ROW_RE.findall(text)]


def row_cells(text: str) -> list[list[str]]:
    """The body rows of a table chunk as cell lists (the header row dropped); [] for text."""
    return _rows(text)[1:]


def _is_number(cell: str) -> bool:
    return bool(_PLAIN_RE.fullmatch(_plain(cell)))


def _item(cells) -> tuple:
    """A table row's item: the content words of its first non-numeric cell with any (the BoQ item
    number, the bidder name). The same item in two tables (ROA p1 and p2) has one key."""
    return next((tuple(_seq(c)) for c in cells if not _is_number(c) and _seq(c)), ())


def _column(header: str) -> frozenset:
    """A column header's words, bracketed unit marks dropped: "Total Bid as Read (TBR)" on ROA p2
    and "Total Bid as Read" on p1 are one column."""
    return frozenset(_words(_BRACKET_RE.sub(" ", header)))


def guard_binding(label: str, question: str, rows: list[list[str]] = ()) -> bool:
    """THE matcher for "an item the question names" (binding now; G7's named-row count too).
    Every content word of the label appears in the calc clause. No exemption (the try-2 exemption
    let E20's item numbers through): an "all N items" clause binds only the operands whose labels it
    names (E5 CR4). G-short (C3): or the question holds the label's first k >= 2 content words,
    contiguous, and exactly one of ``rows`` (the table's body rows) has a non-numeric cell that
    starts with them ("Amethyst Horizon Builders" -> the full legal name)."""
    if _held(label, question):
        return True
    lab, asked = _seq(label), _seq(question)
    held = [k for k in range(2, len(lab) + 1)
            if any(asked[i:i + k] == lab[:k] for i in range(len(asked)))]
    if not held:
        return False
    k = max(held)
    return sum(any(_seq(c)[:k] == lab[:k] for c in r if not _PLAIN_RE.fullmatch(_plain(c)))
               for r in rows) == 1


def guard_value(value: str, found: list[Slot]) -> list[Slot]:
    """The slots whose WHOLE text equals the value, commas ignored: no forged or partial value."""
    plain = _plain(value)
    if not _PLAIN_RE.fullmatch(plain):
        return []
    return [s for s in found if _plain(s.text) == plain]


def guard_column(found: list[Slot], question: str) -> list[Slot]:
    """The table cells whose column header words (bracketed unit marks dropped) all appear in the
    question. A text-chunk slot has no column."""
    asked = _field_words(question)
    return [s for s in found if not s.header or _names_field(s.header, asked)]


def _names_field(header: str, asked: set[str]) -> bool:
    """A column header whose words (bracketed unit marks dropped) all appear in the asked words."""
    return _words(_BRACKET_RE.sub(" ", header)) <= asked


def guard_row(found: list[Slot], label: str) -> list[Slot]:
    """The slots whose row (table row, or text sentence) holds every label word that is not a word
    of the slot's column header."""
    def holds(s: Slot) -> bool:
        head = _words(s.header)
        return bool(_words(label) - head) and _held(label, s.row, skip=head)
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
    """Two operands on the same cell -> a reason (the #256 shape: one operand copied twice). A table
    cell is keyed by (item, column), across chunks too: the same item and column from ROA p1 and p2
    is one figure twice (reviewer probe). The same row in two columns (E10) is two cells. A text
    number is keyed by (chunk, position). "" when all differ."""
    cells = [(_item(s.cells), _column(s.header)) if s.cells else (n.node.node_id, s.pos)
             for _, s, n in picked]
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



def asked_fields(chunks: dict, question: str) -> set[str]:
    """The column headers of this pass's table chunks that the calc clause names."""
    asked = _field_words(question)
    return {s.header for n in chunks.values() for s in slots(n.node.text)
            if s.header and _names_field(s.header, asked)}


def guard_text_field(found: list, fields: set[str]) -> list:
    """The text-field rule (the E05 hole), over the G-chunk search: when the calc clause names a
    table field of this pass, a slot with no column -- a text sentence or an empty-header cell --
    is dropped: it cannot show which field its number is. ``found`` is [(slot, node)]."""
    return [(s, n) for s, n in found if s.header] if fields else found


def check(request: dict, chunks: dict, question: str, op: str,
          order: tuple[str, ...] = ()) -> tuple[list, str]:
    """Run every guard on a parsed request. Returns (picked, "") -- picked = [(label, slot, node)]
    -- or ([], reason) to withhold. ``chunks`` is handles(doc_chunks()) of this pass; ``question``
    is the calc part. G-chunk (C5): the value is searched in EVERY document chunk of the pass, the
    cited one first; the cited handle must still be one of them. A table cell is cited first."""
    operands = request.get("operands") or []
    if request.get("op") != op:
        return [], f"the request's op {request.get('op')!r} is not the question's {op!r}"
    if reason := guard_count(operands, question):
        return [], reason
    fields = asked_fields(chunks, question)
    picked = []
    for o in operands:
        label, value, cid = (str(o.get(k, "")) for k in ("label", "value", "chunk"))
        cited = chunks.get(cid)  # an unknown handle never falls back: withhold
        if cited is None:
            return [], f"operand {label!r} cites {cid!r}, not a retrieved document chunk"
        # Binding per table: a G-short name binds only where its prefix starts exactly one row.
        binds = [n for n in chunks.values() if guard_binding(label, question, row_cells(n.node.text))]
        if not binds:
            return [], f"operand {label!r} is not an item the question names"
        search = [cited] + [n for n in chunks.values() if n is not cited]
        found = [(s, n) for n in search for s in guard_value(value, slots(n.node.text))]
        if not found:
            return [], f"operand {label!r}: {value!r} is not a whole cell or number of a retrieved chunk"
        found = [(s, n) for s, n in found if any(n is b for b in binds)]
        if not found:
            return [], f"operand {label!r} is not an item the question names"
        found = guard_text_field(found, fields)
        if not found:
            return [], (f"operand {label!r} is cited from text, but the question names the table "
                        f"field {sorted(fields)[0]!r}")
        found = [(s, n) for s, n in found if guard_column([s], question)]
        if not found:
            return [], f"operand {label!r}: {value!r} is not in a column the question names"
        found = [(s, n) for s, n in found if guard_row([s], label)]
        if not found:
            return [], f"operand {label!r}: {value!r} is not in a row that names {label!r}"
        s, n = sorted(found, key=lambda f: not f[0].header)[0]  # stable: cited chunk first
        picked.append((label, s, n))
    if reason := guard_duplicate(picked):
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


def pinned(picked: list, question: str) -> bool:
    """G-total (C1): is the add's operand set pinned? When every operand is a table cell, the
    DISTINCT items the question names in the operands' tables -- rows with a number in an operand's
    column, named by guard_binding (THE matcher: glue and short names count) -- must be a subset of
    the operands' items, also under a stated count ("the two items A, B and C" with 2 operands is
    not pinned: reviewer probe). Then (a) a stated count equals the operand count, or (b) no stated
    count: the named items equal the operands' items, and there are >= 2. A text operand cannot be
    counted: only (a). A missed operand leaves a named item over; an extra named row only blocks a
    total; a clause that names no rows ("all 5 Part G items") still totals under (a)."""
    n = stated_count(question)
    if any(not s.cells for _, s, _ in picked):
        return n is not None and n == len(picked)
    columns = {_column(s.header) for _, s, _ in picked}
    named = set()
    for node in {id(n): n for _, _, n in picked}.values():
        head, *body = _rows(node.node.text)
        for r in body:
            in_column = any(i < len(head) and _column(head[i]) in columns and _is_number(c)
                            for i, c in enumerate(r))
            if in_column and any(_seq(c) and guard_binding(c, question, body)
                                 for c in r if not _is_number(c)):
                named.add(_item(r))
    items = {_item(s.cells) for _, s, _ in picked}
    if not named <= items:
        return False
    return n == len(picked) if n is not None else len(picked) >= 2 and named == items


def answer_text(picked: list, question: str, op: str) -> str:
    """THE place that decides and writes the calc answer (the agent and the --check pins both call
    it): the Decimal tape with its total when the operands are pinned -- a subtract's 2 operands in
    the strict order (check() fixed it), or an add that pinned() accepts -- else option C (the cited
    operands, no total). No LLM computes."""
    if op == "subtract" or pinned(picked, question):
        tape = run_tape(picked, op)
        if tape:
            return tape
    return cite(picked)


# DEAD: no caller (G7: answer_text is the only total path). Restored unchanged on review.
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
