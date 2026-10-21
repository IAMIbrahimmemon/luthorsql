"""Group selector and reviser prompts (round 5). Shared by training-data building and serving.

Candidates are grouped by the rows they return. The selector reads the question, the schema, and one
representative query from each of the two largest groups with a preview of its result, and answers
"A" or "B". The reviser reads the same and writes one corrected query; it may keep one candidate as is.
"""
from __future__ import annotations

import collections
import random

from ..sq.schema_card import SchemaCard
from .execute import signature
from .prompt import schema_ddl

SELECT_SYSTEM = ("You check SQL answers. Two SQLite queries were written for the same question and return "
                 "different results. Reply with the letter of the query whose result answers the question "
                 "correctly: A or B.")
REVISE_SYSTEM = ("You fix SQL. Two SQLite queries were written for the same question and disagree. Write the one "
                 "query that answers the question exactly (it may be one of them unchanged). Reply with the SQL only.")
PREVIEW_ROWS = 5
CELL = 40


def run_preview(con, sql: str, max_rows: int = 100_000) -> dict:
    """Execute on an open sqlite3 connection (with its own time limit) -> {runs, empty, sig, cols, n,
    rows (first few, as strings), set (all rows, for comparing with the gold; not serialisable)}."""
    try:
        cur = con.execute(sql)
        got = cur.fetchmany(max_rows)
        cols = [d[0] for d in (cur.description or [])]
    except Exception as e:
        return {"runs": False, "empty": True, "sig": "", "cols": [], "n": 0, "rows": [], "error": str(e)[:200]}
    return {"runs": True, "empty": not any(v is not None for r in got for v in r), "sig": signature(got),
            "cols": cols, "n": len(got), "rows": [[str(v)[:CELL] for v in r] for r in got[:PREVIEW_ROWS]],
            "set": set(map(tuple, got))}


def preview_text(p: dict) -> str:
    if not p["runs"]:
        return f"Error: {p.get('error', '')}"
    if p["n"] == 0:
        return "Result: no rows"
    head = " | ".join(p["cols"])
    body = "\n".join(" | ".join(r) for r in p["rows"])
    more = f"\n... ({p['n']} rows in total)" if p["n"] > len(p["rows"]) else f"\n({p['n']} rows)"
    return f"{head}\n{body}{more}"


def top_groups(cands: list[dict], k: int = 2) -> list[list[dict]]:
    """Groups of runnable candidates with the same result, largest first (ties: earliest candidate)."""
    groups = collections.OrderedDict()
    for c in cands:
        if c.get("runs"):
            groups.setdefault(c["sig"], []).append(c)
    return sorted(groups.values(), key=lambda g: -len(g))[:k]


def _body(card: SchemaCard, question: str, context: str, a: dict, b: dict) -> str:
    s = f"Database schema:\n{schema_ddl(card, question, context)}\n\n"
    if context.strip():
        s += f"Evidence: {context.strip()}\n"
    s += f"Question: {question.strip()}\n\n"
    for name, c in (("A", a), ("B", b)):
        s += f"Query {name}:\n{c['sql']}\n{preview_text(c['preview'])}\n\n"
    return s.rstrip()


def select_messages(card, question, context, a, b) -> list[dict]:
    return [{"role": "system", "content": SELECT_SYSTEM}, {"role": "user", "content": _body(card, question, context, a, b)}]


def revise_messages(card, question, context, a, b) -> list[dict]:
    return [{"role": "system", "content": REVISE_SYSTEM}, {"role": "user", "content": _body(card, question, context, a, b)}]


def build_items(card: SchemaCard, rec: dict, rnd: random.Random) -> list[dict]:
    """Training items for one question whose candidates carry `preview` and `ok`.
    Items use the sqlw_sft format: `messages` (system + user) and the target text in `sql`.
    select: the two largest groups, exactly one right -> answer its letter (order shuffled).
    revise: the two largest groups disagree -> target the gold SQL when neither is right, else the right
            representative unchanged (teaches 'keep the correct one')."""
    g = top_groups(rec["cands"])
    if len(g) < 2:
        return []
    a, b = g[0][0], g[1][0]
    if rnd.random() < 0.5:
        a, b = b, a
    q, ctx, out = rec["question"], rec.get("context", ""), []
    if a["ok"] != b["ok"]:
        out.append({"kind": "select", "id": rec["id"], "db_id": rec["db_id"],
                    "messages": select_messages(card, q, ctx, a, b), "sql": "A" if a["ok"] else "B"})
    target = a["sql"] if a["ok"] else b["sql"] if b["ok"] else rec["gold_sql"]
    out.append({"kind": "revise", "id": rec["id"], "db_id": rec["db_id"], "both_wrong": not (a["ok"] or b["ok"]),
                "messages": revise_messages(card, q, ctx, a, b), "sql": target})
    return out
