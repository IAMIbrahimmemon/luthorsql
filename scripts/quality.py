"""Quality rules for generated (question, plan) pairs, derived from the round-3 Claude audit
(2,000 questions judged by hand). Each rule targets a failure the auditors found repeatedly:

  fan_out     COUNT / SUM / AVG over a table that a join repeats ("how many users" counting their
              tweets). COUNT(*) of a table with a single key is rewritten to COUNT(DISTINCT key), which is
              what the question means; SUM / AVG over a fan-out cannot be repaired and are dropped.
  identifier  the question copies a schema identifier that isn't plain English (has '_', a digit or an
              internal capital: A_VAR18, hasPosition, lgIDLoser), or SQL words ("desc", "ascending"), or
              escape junk (a literal "\\n", "\\x96").
  limit       LIMIT without ORDER BY ("top 5" of an unordered list).
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sq.plan import ROWS, Plan, join_tree  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402

_SQLWORDS = re.compile(r"\b(desc|asc|ascending|descending|grouped by|group by|order by|sorted by|count\(|rows? in the|"
                       r"table|records? in)\b", re.I)
_JUNK = re.compile(r"\\[nrtx]|\\x[0-9a-f]{2}|[\x00-\x08\x0b\x0c\x0e-\x1f\x96]")


def _weird(ident: str) -> bool:
    return bool(re.search(r"_|\d|[a-z][A-Z]", ident)) and len(ident) > 2


def identifier_leak(question: str, card: SchemaCard, literals: set[str]) -> str | None:
    q = question
    for v in literals:
        q = q.replace(str(v), " ")
    if _JUNK.search(q):
        return "junk"
    if _SQLWORDS.search(q):
        return "sql_words"
    names = {t.name for t in card.tables} | {c.name for t in card.tables for c in t.columns}
    for n in names:
        if _weird(n) and re.search(rf"(?<![\w]){re.escape(n)}(?![\w])", q):
            return "identifier"
    return None


def _parents(card: SchemaCard) -> dict[str, set[str]]:
    """table -> tables that reference it (joining one of those repeats the table's rows)."""
    out: dict[str, set[str]] = {}
    for a, _, b, _ in list(card.fks) + [e for e in card.inferred_fks if not card._pk_to_pk(e)]:
        if a != b:
            out.setdefault(b, set()).add(a)
    return out


def fix_fan_out(plan: dict, card: SchemaCard) -> tuple[dict | None, str]:
    """Return (plan, status): status 'ok', 'rewritten' (COUNT(*) -> COUNT(DISTINCT key)) or 'dropped'."""
    p = Plan.from_dict(plan)
    if p.set_op != "none" or any(f.sub is not None for f in p.filters):
        return plan, "ok"
    g = card.graph()
    try:
        tables, _ = join_tree(g, [t for t in dict.fromkeys(p.tables() + list(p.require)) if t in g.columns])
    except Exception:
        return plan, "ok"
    if len(tables) < 2:
        return plan, "ok"
    children = _parents(card)
    pk = {t.name: [c.name for c in t.columns if c.pk] for t in card.tables}
    out = copy.deepcopy(plan)
    status = "ok"
    projs = out["projections"] + ([out["order"]] if out.get("order") else [])
    for x in projs:
        t, col, agg = x["col"]["table"], x["col"]["column"], x["agg"]
        if agg in ("none", "min", "max", "count_distinct"):
            continue
        if col != ROWS and agg == "count":
            agg = "count_col"
        repeated = any(c in tables for c in children.get(t, ()))
        if not repeated:
            continue
        if col == ROWS and len(pk.get(t, [])) == 1:
            x["col"] = {"table": t, "column": pk[t][0]}
            x["agg"] = "count_distinct"
            status = "rewritten"
        else:
            return None, "dropped"
    return out, status


def limit_without_order(plan: dict) -> bool:
    return plan.get("limit") not in (None,) and not plan.get("order")
