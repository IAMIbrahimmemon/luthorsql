"""Clause-level SQL diff for the VALHARD hard-pair miner (docs/paper_notes/valhard_2026-10-10.md).

classify(right_sql, wrong_sql) -> (type, bucket, n_components): which single edit turns the right SQL into the wrong
one, following the System-1 pick-gap taxonomy (docs/paper_notes/research/system1_pick_gap_2026-10-10.md §1):

  bucket grain  count_distinct  COUNT(x) vs COUNT(DISTINCT x), nothing else differs
                count_key       a different counted column / COUNT(*) vs COUNT(col)
                join_fanout     only the joined tables differ and the SELECT aggregates (COUNT/SUM/AVG/TOTAL)
  bucket sem    where_drop      the wrong WHERE is a strict subset of the right one's conjuncts
                where_extra     a strict superset
                literal         one conjunct differs only in its literal(s)
                where_op        one conjunct differs only in its comparison operator
                where_swap      one conjunct replaced by another
                join            only the joined tables differ (no aggregate)
                join_filter     the wrong SQL drops (or adds) a join together with the conjuncts that use it
                order_dir       ORDER BY keys equal, a direction differs
                order_expr      different ORDER BY keys
  bucket proj   proj_order      same output columns, other order
                proj_drop       the wrong SELECT misses column(s) of the right one
                proj_extra      the wrong SELECT has extra column(s)
                id_name         one output column swapped for an id <-> name column of the same table
                proj_swap       one output column swapped for another
                agg_func        aggregate function differs (COUNT vs SUM, MAX vs MIN, ...)
                proj_other      other SELECT-only differences (arithmetic, casts, several columns)
  bucket ties   ties            only LIMIT differs, or ORDER/LIMIT vs an = (SELECT MAX/MIN) filter: the gold convention
                                is 50/50 on DECISION, so these are never used as hard negatives
  bucket other  distinct | same | multi | unparsed

Aliases are resolved to table names (lower case) so T1/T2 numbering never counts as a difference.
"""
from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

HARD_BUCKETS = ("grain", "sem", "proj")
BUCKET = {"count_distinct": "grain", "count_key": "grain", "join_fanout": "grain",
          "where_drop": "sem", "where_extra": "sem", "literal": "sem", "where_op": "sem", "where_swap": "sem",
          "join": "sem", "join_filter": "sem", "order_dir": "sem", "order_expr": "sem",
          "proj_order": "proj", "proj_drop": "proj", "proj_extra": "proj", "id_name": "proj", "proj_swap": "proj",
          "agg_func": "proj", "proj_other": "proj", "ties": "ties"}
_AGG_FANOUT = (exp.Count, exp.Sum, exp.Avg)
_SUPERLATIVE = re.compile(r"=\s*\(\s*SELECT\s+(MAX|MIN)\s*\(", re.I)
_ID = re.compile(r"(^id$|_id$|id$|^id_|ID$|Id$|_ID$|code$|key$)", re.I)
_NAME = re.compile(r"(name|title|first|last|label)", re.I)


def parse(sql: str):
    try:
        t = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    return t if isinstance(t, exp.Select) else None


def _resolve(t: exp.Select) -> exp.Select:
    t = t.copy()
    amap = {tb.alias: tb.name for tb in t.find_all(exp.Table) if tb.alias}
    for col in t.find_all(exp.Column):
        if col.table and col.table in amap:
            col.set("table", exp.to_identifier(amap[col.table].lower()))
        elif col.table:
            col.set("table", exp.to_identifier(col.table.lower()))
    for tb in t.find_all(exp.Table):
        tb.set("alias", None)
    return t


def conjuncts(node) -> list:
    if node is None:
        return []
    out, stack = [], [node.this if isinstance(node, exp.Where) else node]
    while stack:
        x = stack.pop()
        if isinstance(x, exp.And):
            stack += [x.right, x.left]
        elif isinstance(x, exp.Paren) and isinstance(x.this, exp.And):
            stack.append(x.this)
        else:
            out.append(x)
    return out


def _s(e) -> str:
    return e.sql(dialect="sqlite") if e is not None else ""


def feats(sql: str) -> dict | None:
    """Clause features with aliases resolved; None when the SQL does not parse as one SELECT."""
    t = parse(sql)
    if t is None:
        return None
    r = _resolve(t)
    joins = r.args.get("joins") or []
    frm = r.args.get("from_") or r.args.get("from")
    tables = []
    if frm is not None and isinstance(frm.this, exp.Table):
        tables.append(frm.this.name.lower())
    tables += [j.this.name.lower() for j in joins if isinstance(j.this, exp.Table)]
    order = r.args.get("order")
    where = conjuncts(r.args.get("where"))
    sel = [_unalias(p) for p in r.expressions]
    return {
        "select": [_s(p) for p in sel],
        "sel_nodes": sel,
        "distinct": r.args.get("distinct") is not None,
        "tables": tuple(sorted(tables)),
        "joinconds": frozenset(_s(c) for j in joins for c in conjuncts(j.args.get("on"))),
        "where": frozenset(_s(c) for c in where),
        "where_nodes": {_s(c): c for c in where},
        "group": _s(r.args.get("group")), "having": _s(r.args.get("having")),
        "order": tuple((_s(o.this), bool(o.args.get("desc"))) for o in (order.expressions if order else [])),
        "limit": _s(r.args.get("limit")),
        "superlative": bool(_SUPERLATIVE.search(sql)),
        "fanout_agg": any(isinstance(n, _AGG_FANOUT) for p in sel for n in p.find_all(*_AGG_FANOUT)),
        "where_tables": {k: {c.table for c in v.find_all(exp.Column)} for k, v in ((_s(c), c) for c in where)},
    }


def _unalias(e):
    return e.this if isinstance(e, exp.Alias) else e


COMPONENTS = ("select", "distinct", "tables", "joinconds", "where", "group", "having", "order", "limit")


def components(fa: dict, fb: dict) -> list[str]:
    return [k for k in COMPONENTS if fa[k] != fb[k]]


def _literal_only(a, b) -> bool:
    """Same tree after every literal is blanked."""
    def blank(e):
        e = e.copy()
        for lit in list(e.find_all(exp.Literal)):
            lit.replace(exp.Literal.string("?"))
        return _s(e)
    return blank(a) == blank(b) and _s(a) != _s(b)


def _op_only(a, b) -> bool:
    return (isinstance(a, (exp.Binary, exp.Predicate)) and isinstance(b, (exp.Binary, exp.Predicate))
            and type(a) is not type(b) and _s(a.args.get("this")) == _s(b.args.get("this"))
            and _s(a.args.get("expression")) == _s(b.args.get("expression")))


def _count_parts(node):
    """[(distinct, arg sql)] for every COUNT in a projection node."""
    out = []
    for c in node.find_all(exp.Count):
        a = c.this
        if isinstance(a, exp.Distinct):
            out.append((True, ",".join(_s(x) for x in a.expressions)))
        else:
            out.append((False, _s(a)))
    return out


def _select_type(fr: dict, fw: dict) -> str:
    sr, sw = fr["select"], fw["select"]
    if sorted(sr) == sorted(sw):
        return "proj_order"
    if len(sw) < len(sr) and set(sw) < set(sr):
        return "proj_drop"
    if len(sw) > len(sr) and set(sw) > set(sr):
        return "proj_extra"
    if len(sr) != len(sw):
        return "proj_other"
    diff = [k for k in range(len(sr)) if sr[k] != sw[k]]
    if len(diff) != 1:
        return "proj_other"
    a, b = fr["sel_nodes"][diff[0]], fw["sel_nodes"][diff[0]]
    ca, cb = _count_parts(a), _count_parts(b)
    if ca and cb and len(ca) == len(cb):
        da = [d for d, _ in ca]
        db = [d for d, _ in cb]
        aa = [x for _, x in ca]
        ab = [x for _, x in cb]
        strip = lambda s: re.sub(r"COUNT\(\s*DISTINCT\s+", "COUNT(", s, flags=re.I)  # noqa: E731
        if aa == ab and da != db and strip(sr[diff[0]]) == strip(sw[diff[0]]):
            return "count_distinct"
        if aa != ab:
            blank = lambda n: re.sub(r"COUNT\((DISTINCT\s+)?[^()]*\)", "COUNT(?)", _s(n), flags=re.I)  # noqa: E731
            if blank(a) == blank(b):
                return "count_key"
    if isinstance(a, exp.Column) and isinstance(b, exp.Column):
        if a.table == b.table and ((_ID.search(a.name) and _NAME.search(b.name)) or (_NAME.search(a.name) and _ID.search(b.name))):
            return "id_name"
        return "proj_swap"
    if isinstance(a, exp.AggFunc) and isinstance(b, exp.AggFunc) and type(a) is not type(b) \
            and _s(a.this) == _s(b.this):
        return "agg_func"
    return "proj_other"


def classify(right_sql: str, wrong_sql: str, fr: dict | None = None, fw: dict | None = None) -> tuple[str, str, int]:
    """(type, bucket, number of differing components) for the edit right -> wrong. fr / fw: cached feats()."""
    fr = feats(right_sql) if fr is None else fr
    fw = feats(wrong_sql) if fw is None else fw
    if fr is None or fw is None:
        return "unparsed", "other", 99
    d = components(fr, fw)
    n = len(d)
    ds = set(d)
    if not d:
        return "same", "other", 0
    t = None
    if ds <= {"limit", "order"} and "limit" in ds:
        t = "ties"
    elif ds <= {"where", "order", "limit"} and fr["superlative"] != fw["superlative"]:
        t = "ties"
    elif ds == {"select"}:
        t = _select_type(fr, fw)
    elif ds == {"distinct"}:
        return "distinct", "other", n
    elif ds <= {"tables", "joinconds"}:
        t = "join_fanout" if (fr["fanout_agg"] or fw["fanout_agg"]) else "join"
    elif ds <= {"tables", "joinconds", "where"} and "tables" in ds:
        tr, tw = set(fr["tables"]), set(fw["tables"])
        extra = (tr - tw) or (tw - tr)
        wd = fr["where"] ^ fw["where"]
        uses = set()
        for k in wd:
            uses |= (fr["where_tables"].get(k) or fw["where_tables"].get(k) or set())
        if extra and uses and uses <= extra and ((fw["where"] < fr["where"] and tw < tr) or
                                                  (fr["where"] < fw["where"] and tr < tw)):
            t = "join_filter"
    elif ds == {"where"}:
        r, w = fr["where"], fw["where"]
        if w < r:
            t = "where_drop"
        elif r < w:
            t = "where_extra"
        elif len(r - w) == 1 and len(w - r) == 1:
            a, b = fr["where_nodes"][next(iter(r - w))], fw["where_nodes"][next(iter(w - r))]
            t = "literal" if _literal_only(a, b) else "where_op" if _op_only(a, b) else "where_swap"
    elif ds == {"order"}:
        if [k for k, _ in fr["order"]] == [k for k, _ in fw["order"]]:
            t = "order_dir"
        else:
            t = "order_expr"
    if t is None:
        return "multi", "other", n
    return t, BUCKET[t], n


def is_hard(kind: str) -> bool:
    return BUCKET.get(kind) in HARD_BUCKETS
