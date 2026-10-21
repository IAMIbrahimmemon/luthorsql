"""Near-miss negatives: one edit to a right SQL (sqlglot AST), for the val2 validator training lists.

Edits (each yields at most one mutant):
  distinct   toggle DISTINCT on the outer SELECT
  limit      outer LIMIT present -> removed; absent -> LIMIT 1 added
  x100       a projection with "* 100" -> the factor removed (= divide by 100); else the first numeric-looking
             projection (aggregate / arithmetic / CAST) multiplied by 100
  extra_col  one more column of the first projected column's table, appended to the outer SELECT
  cmp        the first < > <= >= comparison: < <-> <=, > <-> >=

Only SQL whose sqlglot round trip (sqlite dialect) reproduces the original text exactly is mutated, so the one edit is
the only textual difference: the validator cannot tell a mutant by its formatting. The caller executes each mutant
and keeps it only when it runs, is non-null and its result signature differs from every right class.
"""
from __future__ import annotations

import random

import sqlglot
from sqlglot import exp

KINDS = ("distinct", "limit", "x100", "extra_col", "cmp")
_AGG = (exp.AggFunc, exp.Cast, exp.Binary, exp.Round)
_CMP = {exp.LT: exp.LTE, exp.LTE: exp.LT, exp.GT: exp.GTE, exp.GTE: exp.GT}


def parse_identity(sql: str):
    """The sqlglot tree when the round trip is the identity, else None."""
    try:
        t = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if not isinstance(t, exp.Select) or t.sql(dialect="sqlite") != sql:
        return None
    return t


def _unalias(e):
    return e.this if isinstance(e, exp.Alias) else e


def _tables(sel: exp.Select) -> dict:
    """alias (or bare name) -> table name for the outer SELECT's FROM / JOIN tables."""
    out = {}
    srcs = []
    if sel.args.get("from_") is not None or sel.args.get("from") is not None:
        f = sel.args.get("from_") or sel.args.get("from")
        srcs.append(f.this)
    srcs += [j.this for j in sel.args.get("joins") or []]
    for t in srcs:
        if isinstance(t, exp.Table):
            out[t.alias_or_name] = t.name
            out.setdefault(t.name, t.name)
    return out


def m_distinct(t):
    t = t.copy()
    t.set("distinct", None if t.args.get("distinct") else exp.Distinct())
    return t


def m_limit(t):
    t = t.copy()
    if t.args.get("limit") is not None:
        t.set("limit", None)
        return t
    return t.limit(1, copy=False)


def m_x100(t):
    t = t.copy()
    projs = t.expressions
    for p in projs:
        for m in _unalias(p).find_all(exp.Mul):
            for side, other in (("expression", "this"), ("this", "expression")):
                v = m.args.get(side)
                if isinstance(v, exp.Literal) and v.this in ("100", "100.0"):
                    m.replace(m.args[other])
                    return t
    for i, p in enumerate(projs):
        inner = _unalias(p)
        if isinstance(inner, _AGG) and not isinstance(inner, (exp.Connector, exp.Predicate)):
            new = exp.Mul(this=inner.copy(), expression=exp.Literal.number(100))
            if isinstance(p, exp.Alias):
                p.set("this", new)
            else:
                p.replace(new)
            return t
    return None


def m_extra_col(t, columns: dict, rnd: random.Random):
    """columns: table name -> [column names] (schema order)."""
    t = t.copy()
    tabs = _tables(t)
    cols = list(t.expressions[0].find_all(exp.Column)) if t.expressions else []
    if not cols:
        return None
    c0 = cols[0]
    qual = c0.table
    if not qual:
        if len({v for v in tabs.values()}) != 1:
            return None
        name = next(iter(tabs.values()))
    else:
        name = tabs.get(qual)
    if not name or name not in columns:
        return None
    used = {c.name for p in t.expressions for c in p.find_all(exp.Column)}
    choices = [c for c in columns[name] if c not in used]
    if not choices:
        return None
    col = rnd.choice(choices)
    new = exp.Column(this=exp.to_identifier(col, quoted=True),
                     table=exp.to_identifier(qual, quoted=False) if qual else None)
    t.set("expressions", t.expressions + [new])
    return t


def m_cmp(t):
    t = t.copy()
    for node in t.find_all(*_CMP):
        node.replace(_CMP[type(node)](this=node.this.copy(), expression=node.expression.copy()))
        return t
    return None


def mutants(sql: str, columns: dict | None = None, seed: str = "") -> list[tuple[str, str]]:
    """[(kind, mutated sql)] for every edit that applies (distinct, in a seeded order); [] when the SQL does not
    round-trip through sqlglot unchanged."""
    t = parse_identity(sql)
    if t is None:
        return []
    rnd = random.Random(f"mut:{seed}")
    out, seen = [], {sql}
    for kind in KINDS:
        try:
            if kind == "distinct":
                m = m_distinct(t)
            elif kind == "limit":
                m = m_limit(t)
            elif kind == "x100":
                m = m_x100(t)
            elif kind == "extra_col":
                m = m_extra_col(t, columns or {}, rnd)
            else:
                m = m_cmp(t)
            s = m.sql(dialect="sqlite") if m is not None else None
        except Exception:
            s = None
        if s and s not in seen:
            seen.add(s)
            out.append((kind, s))
    rnd.shuffle(out)
    return out
