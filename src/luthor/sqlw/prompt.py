"""Prompt for the SQL writer: compact DDL (types, keys, descriptions, sample values) + question + evidence.

Long schemas are cut to a character budget. Tables are ranked by word overlap with the question and the
evidence; in training the tables the gold SQL uses are always kept (at serve time the ranking alone decides,
and the slot model's candidates can add their tables via `must`).
"""
from __future__ import annotations

import re

from ..sq.schema_card import SchemaCard

SYSTEM = ("You write one SQLite query that answers the question exactly. Use only the tables and columns "
          "given. Reply with the SQL only.")
BUDGET = 9000  # characters of DDL


def _q(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else f'"{name}"'


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower().replace("_", " ")) if len(w) > 2}


def _one_line(s: str) -> str:
    """Descriptions and sample values may contain line breaks, which would break the DDL line (round 5)."""
    return " ".join(str(s).split())


def table_ddl(card: SchemaCard, t) -> str:
    lines = []
    for c in t.columns:
        bits = [f"  {_q(c.name)} {c.type.lower() or 'text'}"]
        if c.pk:
            bits.append("PRIMARY KEY")
        note = []
        if c.description and c.description.lower() != c.name.lower():
            note.append(_one_line(c.description)[:160])
        if c.samples:
            note.append("e.g. " + ", ".join(_one_line(s)[:40] for s in c.samples[:3]))
        lines.append(" ".join(bits) + (f",  -- {'; '.join(note)}" if note else ","))
    for t1, c1, t2, c2 in card.fks:
        if t1 == t.name:
            lines.append(f"  FOREIGN KEY ({_q(c1)}) REFERENCES {_q(t2)}({_q(c2)}),")
    if lines:
        lines[-1] = lines[-1].replace(",  --", "  --", 1) if ",  --" in lines[-1] else lines[-1].rstrip(",")
    head = f"CREATE TABLE {_q(t.name)} (" + (f"  -- {t.description.strip()[:200]}" if t.description else "")
    return "\n".join([head, *lines, ");"])


def used_tables(card: SchemaCard, sql: str) -> list[str]:
    low = sql.lower()
    return [t.name for t in card.tables if re.search(r'(?<![\w"])"?' + re.escape(t.name.lower()) + r'"?(?![\w"])', low)]


def schema_ddl(card: SchemaCard, question: str, context: str = "", must: list[str] | None = None,
               budget: int = BUDGET) -> str:
    ddl = {t.name: table_ddl(card, t) for t in card.tables}
    if sum(len(v) + 2 for v in ddl.values()) <= budget:
        keep = [t.name for t in card.tables]
    else:
        qw = _words(question + " " + context)
        score = {t.name: len(qw & _words(t.name + " " + " ".join(c.name + " " + c.description for c in t.columns)))
                 for t in card.tables}
        keep, total = [], 0
        for name in list(dict.fromkeys(must or [])) + sorted(ddl, key=lambda n: -score[n]):
            if name in ddl and name not in keep and (total + len(ddl[name]) <= budget or name in (must or [])):
                keep.append(name)
                total += len(ddl[name]) + 2
        order = [t.name for t in card.tables]
        keep.sort(key=order.index)
    return "\n\n".join(ddl[n] for n in keep)


def value_hints(vindex, question: str, context: str = "", max_hints: int = 12) -> list[str]:
    """Database values that match words in the question or evidence, as `table.column = 'value'` lines
    (round 5). `vindex` is a luthor.sq.candidates.ValueIndex built with skip_card_values=False."""
    out = []
    for col, v, _, _ in vindex.match(question + " \n " + context, max_hits=4 * max_hints):
        t, c = col.split(".", 1)
        h = f"{_q(t)}.{_q(c)} = '{v.replace(chr(39), chr(39) * 2)}'"
        if h not in out:
            out.append(h)
        if len(out) >= max_hints:
            break
    return out


def messages(card: SchemaCard, question: str, context: str = "", must: list[str] | None = None,
             hints: list[str] | None = None) -> list[dict]:
    user = f"Database schema:\n{schema_ddl(card, question, context, must)}\n\n"
    if hints:
        user += "Matching database values:\n" + "\n".join(hints) + "\n\n"
    if context.strip():
        user += f"Evidence: {context.strip()}\n"
    user += f"Question: {question.strip()}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def clean_sql(text: str) -> str:
    """The first SQL statement in a reply (code fences and trailing prose removed)."""
    text = re.sub(r"<\|(?:im_end|endoftext|im_start)\|>.*", "", text, flags=re.S)  # end markers some runtimes keep
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    s = (m.group(1) if m else text).strip()
    return s.split(";")[0].strip() if ";" in s else s
