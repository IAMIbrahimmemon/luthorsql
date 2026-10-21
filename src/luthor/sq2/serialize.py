"""Architecture v2 inputs.

One model input per question:

    [schema text of the database]  [question + literal candidates + slot markers]
     cached once per database        a few hundred tokens per question

The schema text lists every table (its line is the table's "rows" option) and every column
(its line is the column's option) with type, key / reference, description and example values.
A column's option vector is the mean of the model's states over its line, so question words
and column words attend to each other in the same pass. Slot positions after the question
(one learned embedding each) are read out and point at options and literal candidates.

Literal candidates need no extra tokens: a mentioned one is the question's states over its
mention, an enumerated low-cardinality value is the schema's states over where it is listed.
With value-side input (--value-side) the selected candidates are also written after the question
("Values: ..."), and each one's states there (which have read the question) are added to its vector.
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass

from ..sq.candidates import LOWCARD_ENUM
from ..sq.plan import GRAMMAR_V4, ROWS
from ..sq.schema_card import SchemaCard

_LOG = logging.getLogger("luthorsql")

SCHEMA_HEAD = "Database schema. Each table, then its columns.\n"
DESC_MAX = 120
MAX_CANDS = 64

# slot markers per plan block: m = main plan, s = inner plan of a subquery filter, o = the
# right-hand plan of a set operation. "plan" carries the block's choice heads.
BLOCK_SLOTS = {
    "m": ("plan", "p0", "p1", "p2", "p3", "filter", "group", "order", "require", "limit",
          "time", "c0", "c1", "via", "join", "having", "x0", "x1"),
    "s": ("plan", "p0", "filter", "group", "order", "require", "limit", "x0"),
    "o": ("plan", "p0", "p1", "p2", "p3", "filter", "group", "order", "require", "limit", "x0"),
}
# grammar v4 slot markers: projections 5-6 of the main plan and a second ORDER BY key (main and set-op
# blocks). Only with LUTHOR_GRAMMAR=v4 (luthor.sq.plan.GRAMMAR_V4); the default keeps the v3 layout, so v3
# checkpoints (R1, R2) load and score exactly as before.
if GRAMMAR_V4:
    # j0..j7: the join-hint columns as ordered (left, right) pairs (v3 reads them as one unordered set)
    BLOCK_SLOTS["m"] = BLOCK_SLOTS["m"] + ("p4", "p5", "order2", "j0", "j1", "j2", "j3", "j4", "j5", "j6", "j7")
    BLOCK_SLOTS["o"] = BLOCK_SLOTS["o"] + ("order2",)
    # grammar v4.1: d0/d1 = the second condition of a conditional aggregate in projection 1 / 2
    BLOCK_SLOTS["m"] = BLOCK_SLOTS["m"] + ("d0", "d1")
SLOTS = [f"{b}.{s}" for b, ss in BLOCK_SLOTS.items() for s in ss]


@dataclass
class SchemaText:
    text: str
    option_ids: list[str]          # "table.*" (rows) and "table.column"
    spans: list[tuple[int, int]]   # char span of each option's line
    value_spans: dict[tuple[str, str], tuple[int, int]]  # (column id, value) -> char span where listed


def schema_text(card: SchemaCard, tables: list[str] | None = None, aug: "SchemaAug | None" = None) -> SchemaText:
    """Serialise the schema; `tables` (training) keeps only these tables, in this order. `aug` (training)
    shows renamed identifiers / hides descriptions; option ids stay the real names, so SQL is unchanged."""
    nm = (lambda x: aug.names.get(x, x.split(".")[-1])) if aug else (lambda x: x.split(".")[-1])
    keep_desc = (lambda: not aug or not aug.drop_desc)
    fk = {(a, ca): f"{nm(b)}.{nm(b + '.' + cb)}" for a, ca, b, cb in card.fks}
    by_name = {t.name: t for t in card.tables}
    order = [by_name[t] for t in tables if t in by_name] if tables is not None else card.tables
    parts, ids, spans, pos = [SCHEMA_HEAD], [], [], len(SCHEMA_HEAD)
    vspans: dict[tuple[str, str], tuple[int, int]] = {}
    for t in order:
        lines = []
        d = f": {t.description[:DESC_MAX]}" if t.description and keep_desc() else ""
        lines.append((f"Table {nm(t.name)}{d}\n", f"{t.name}.{ROWS}", []))
        for c in t.columns:
            bits = [c.type.lower() or "any"]
            if c.pk:
                bits.append("key")
            if (t.name, c.name) in fk:
                bits.append(f"references {fk[(t.name, c.name)]}")
            d = f": {c.description[:DESC_MAX]}" if c.description and keep_desc() else ""
            vals = card.values.get(f"{t.name}.{c.name}")
            head = f"  {nm(t.name + '.' + c.name)} ({'; '.join(bits)}){d}"
            rel = []
            if vals and len(vals) <= LOWCARD_ENUM:
                ex = " values:"
                for k, v in enumerate(vals):
                    ex += "," if k else ""
                    rel.append((v, len(head) + len(ex) + 1, len(head) + len(ex) + 1 + len(v)))
                    ex += " " + v
            elif c.samples:
                ex = f" e.g. {', '.join(s[:30] for s in c.samples[:3])}"
            else:
                ex = ""
            lines.append((head + ex + "\n", f"{t.name}.{c.name}", rel))
        for s, oid, rel in lines:
            ids.append(oid)
            spans.append((pos, pos + len(s) - 1))
            for v, a, b in rel:
                vspans[(oid, v)] = (pos + a, pos + b)
            parts.append(s)
            pos += len(s)
    return SchemaText("".join(parts), ids, spans, vspans)


_ABBR = {"number": "num", "customer": "cust", "identifier": "id", "address": "addr", "amount": "amt", "quantity": "qty",
         "description": "desc", "department": "dept", "employee": "emp", "account": "acct", "average": "avg",
         "total": "tot", "count": "cnt", "date": "dt", "first": "fst", "last": "lst", "name": "nm", "code": "cd",
         "status": "stat", "category": "cat", "product": "prod", "transaction": "txn", "order": "ord", "price": "prc",
         "value": "val", "year": "yr", "month": "mo", "type": "typ", "location": "loc", "country": "ctry",
         "information": "info", "manager": "mgr", "student": "stu", "teacher": "tchr", "phone": "ph", "street": "st"}


@dataclass
class SchemaAug:
    names: dict[str, str]   # "table" / "table.column" -> displayed name
    drop_desc: bool


def _words(ident: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", ident)
    return [w for w in re.split(r"[_\s\-\.]+", s) if w]


_STYLES = {"snake": lambda ws: "_".join(ws), "camel": lambda ws: ws[0] + "".join(w.title() for w in ws[1:]),
           "pascal": lambda ws: "".join(w.title() for w in ws), "upper": lambda ws: "_".join(w.upper() for w in ws),
           "lower": lambda ws: "".join(ws)}


def random_aug(card: SchemaCard, rnd: random.Random, p: float = 0.35) -> SchemaAug | None:
    """Training augmentation, with probability p per database-step: every identifier is split into words,
    often abbreviated (customer_number -> cust_num), and re-joined in ONE case style for the whole schema
    (real databases are consistent); descriptions are hidden half the time. Teaches the model to link
    questions to names it has never seen; the real names still go into the SQL."""
    if rnd.random() >= p:
        return None
    abbreviate = rnd.random() < 0.6
    style = _STYLES[rnd.choice(list(_STYLES))]
    seed = rnd.random()

    def show(ident: str) -> str:
        ws = [w.lower() for w in _words(ident)] or [ident.lower()]
        if abbreviate:
            r = random.Random(f"{seed}:{ident}")
            ws = [_ABBR.get(w, w[:4] if len(w) > 6 and r.random() < 0.5 else w) for w in ws]
        return style(ws)

    names = {}
    for t in card.tables:
        names[t.name] = show(t.name)
        for c in t.columns:
            names[f"{t.name}.{c.name}"] = show(c.name)
    return SchemaAug(names, drop_desc=rnd.random() < 0.5)


def train_tables(card: SchemaCard, gold_tables: set[str], budget_chars: int, rnd: random.Random,
                 shuffle_p: float = 0.3) -> list[str] | None:
    """Training: the full schema when it fits; otherwise the gold tables plus random others up
    to the budget (so the model still has to pick among look-alikes). Occasionally shuffles
    the table order so position carries no signal."""
    full = schema_text(card)
    names = [t.name for t in card.tables]
    if len(full.text) <= budget_chars:
        if rnd.random() < shuffle_p:
            rnd.shuffle(names)
            return names
        return None
    size = {t.name: len(schema_text(card, [t.name]).text) - len(SCHEMA_HEAD) for t in card.tables}
    keep = [n for n in names if n in gold_tables]
    used = len(SCHEMA_HEAD) + sum(size[n] for n in keep)
    rest = [n for n in names if n not in gold_tables]
    rnd.shuffle(rest)
    for n in rest:
        if used + size[n] <= budget_chars:
            keep.append(n)
            used += size[n]
    if rnd.random() < shuffle_p:
        rnd.shuffle(keep)
    else:
        keep.sort(key=names.index)
    return keep or None  # one table alone over the budget: show it whole and let the model try


def _col_link(col: str | None, qw: set[str]) -> int:
    """How strongly the question names a value's column: 2 = every word of the column name, 1 = some word of the
    column or its table, 0 = none."""
    if not col:
        return 0
    t, _, c = col.partition(".")
    cw, tw = _words_of(c) - {"id"}, _words_of(t)
    return 2 if cw and cw <= qw else int(bool((cw | tw) & qw))


def cand_rank(cands: list, body: str = "") -> list[int]:
    """v2 relevance order (deterministic, the same at training and serving): everything with a place in the question
    (spans, card-value hits, database-lookup hits) in find order; then enumerated values, highest first by
    2 x (a word of the value is in the question: "sales representative" -> 'Sales Rep') + _col_link, ties in find
    order."""
    qw = _words_of(body)

    def key(i):
        c = cands[i]
        if c.span is not None:
            return (0, 0, i)
        vw = {w for w in _words_of(str(c.value)) if len(w) > 2 and not w.isdigit()}
        return (1, -(2 * bool(vw & qw) + _col_link(c.col, qw)), i)
    return sorted(range(len(cands)), key=key)


@dataclass
class QuestionText:
    text: str
    q_offset: int   # char offset of the question body in `text`; the slot positions follow the text
    v_spans: list | None = None    # value-side: char span of each candidate's value in "Values:" (None: not shown)
    q_end: int | None = None                      # value-side: where the question part (body) ends


VALUE_MAX = 40      # chars of one value shown in the Values section
VALUE_ENTRIES = 24  # default number of candidates written there (train_sq2.py --value-entries): +130 tokens a
# question on the sys1 mix (+49% tokens per database-step); 98.8% of BIRD-train gold literals rank in the top 24


def value_section(cands: list, names: dict[str, str] | None = None, body: str = "",
                  max_entries: int = VALUE_ENTRIES) -> tuple[str, list[tuple[int, int] | None]]:
    """Value-side input (train_sq2.py --value-side): the `max_entries` most relevant selected candidates (cand_rank,
    the order that also cuts the list) written after the question, grouped by column in first-appearance order,
    e.g. "\nValues: 1995 | 12; customers.country = Egypt | Kenya". Spans (no column) come first as one group.
    Returns the text and each candidate's value span in it (relative; None = not written). `names`: identifiers as
    a renamed schema shows them (SchemaAug.names), so the section matches the schema."""
    shown = sorted(cand_rank(cands, body)[:max_entries])
    groups: dict = {}
    for i in shown:
        c = cands[i]
        groups.setdefault(c.col if c.kind == "value" and c.col else None, []).append(i)
    text, spans = "\nValues: ", [None] * len(cands)
    for g, (col, idx) in enumerate(sorted(groups.items(), key=lambda kv: (kv[0] is not None, kv[1][0]))):
        text += "; " if g else ""
        if col is not None:
            t, _, c = col.partition(".")
            text += f"{names.get(t, t)}.{names.get(col, c)} = " if names else f"{col} = "
        for k, i in enumerate(idx):
            text += " | " if k else ""
            v = " ".join(str(cands[i].value).split())[:VALUE_MAX] or "''"
            spans[i] = (len(text), len(text) + len(v))
            text += v
    return text, spans


def question_text(question: str, context: str = "", cands: list | None = None,
                  names: dict[str, str] | None = None, max_entries: int = VALUE_ENTRIES) -> QuestionText:
    """`cands` (value-side only): append the Values section after the question + context, before the slots."""
    head = "\nQuestion: "
    body = question + (f"\nContext: {context}" if context else "")
    if cands is None:
        return QuestionText(head + body + "\nPlan:", len(head))
    vt, vs = value_section(cands, names, body, max_entries) if cands else ("", [])
    q_end = len(head) + len(body)
    return QuestionText(head + body + vt + "\nPlan:", len(head),
                        [(sp[0] + q_end, sp[1] + q_end) if sp else None for sp in vs], q_end)


def char_to_token_ranges(offsets: list[tuple[int, int]], spans: list[tuple[int, int] | None]) -> list[tuple[int, int]]:
    """Token range [s, e) covering each char span (falls back to the nearest token)."""
    out = []
    starts = [a for a, _ in offsets]
    for sp in spans:
        if sp is None:
            out.append((0, 0))
            continue
        a, b = sp
        toks = [i for i, (s, e) in enumerate(offsets) if e > s and s < b and e > a]
        if not toks:
            j = max(0, min(range(len(starts)), key=lambda i: abs(starts[i] - a)))
            toks = [j]
        out.append((toks[0], toks[-1] + 1))
    return out


def char_to_token(offsets: list[tuple[int, int]], pos: int) -> int:
    for i, (s, e) in enumerate(offsets):
        if s <= pos < e:
            return i
    return max(i for i, (s, e) in enumerate(offsets) if s <= pos)


# ---- overflow-only schema trimming (serve time) ------------------------------------------
# OmniSQL-7B reads 32,768 tokens. A schema text longer than SERVE_MAX_CHARS (about 18k tokens, leaving room
# for the question, slots and a safety margin) cannot be read whole, so only then is it cut down to the tables
# a question most likely needs. Every benchmark database fits (the longest, european_football_2, is 17.7k
# characters), so on them this never runs and their outputs are unchanged.
SERVE_MAX_CHARS = 60_000

_WORD = re.compile(r"[a-z0-9]+")


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _words_of(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).replace("_", " ").lower())}


def table_relevance(card: SchemaCard, question: str, cand_cols: set[str] = frozenset()) -> dict[str, float]:
    """Score each table for one question: its name and column names written in the question (strong), words of
    its descriptions (weak), and literal values the question mentions that live in it (strong)."""
    q = _words_of(question)
    out = {}
    for t in card.tables:
        s = 3.0 * len(_words_of(t.name) & q)
        for c in t.columns:
            nw = _words_of(c.name) - {"id"}
            if nw and nw <= q:
                s += 2.0
            elif nw & q:
                s += 0.5
            if c.description:
                s += 0.2 * len((_words_of(c.description) - {"the", "of", "a", "an", "and", "or", "in", "is", "to"}) & q)
            if f"{t.name}.{c.name}" in cand_cols:
                s += 2.0
        out[t.name] = s
    return out


def serve_tables(card: SchemaCard, question: str, cand_cols: set[str] = frozenset(),
                 max_chars: int = SERVE_MAX_CHARS) -> list[str] | None:
    """None when the whole schema fits (the normal case). Otherwise the most relevant tables, each followed by
    the tables on its FK path to the ones already kept (so the joins still exist), up to the budget."""
    if len(schema_text(card).text) <= max_chars:
        return None
    from ..sq.plan import join_tree
    size = {t.name: len(schema_text(card, [t.name]).text) - len(SCHEMA_HEAD) for t in card.tables}
    rel = table_relevance(card, question, cand_cols)
    names = [t.name for t in card.tables]
    order = sorted(names, key=lambda n: (-rel[n], names.index(n)))
    g = card.graph()
    keep: list[str] = []
    used = len(SCHEMA_HEAD)
    for n in order:
        if n in keep:
            continue
        add = [n]
        if keep:
            try:
                tree, _ = join_tree(g, keep + [n])
                add = [x for x in tree if x not in keep]
            except Exception:
                _LOG.debug("no join path to table %s; added alone", n, exc_info=True)
        cost = sum(size[x] for x in add)
        if used + cost > max_chars:
            if not keep and size[n] <= max_chars - used:  # never return nothing
                keep, used = [n], used + size[n]
            continue
        keep += add
        used += cost
    keep.sort(key=names.index)
    return keep or None  # one table alone over the budget: show it whole and let the model try
