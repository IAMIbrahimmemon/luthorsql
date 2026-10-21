"""Synthetic minimal edits of a right SQL for VALHARD hard negatives (sqlglot AST, sqlite dialect).

Each kind is one edit of the pick-gap taxonomy (docs/paper_notes/research/system1_pick_gap_2026-10-10.md §3):

  grain  count_distinct  COUNT(x) <-> COUNT(DISTINCT x) (first COUNT over a column)
         count_key       the counted column -> another join-key column of the query (DISTINCT kept); COUNT(*) with a
                         join -> COUNT(DISTINCT <the FROM table's join key>)
         add_join        INNER JOIN one child table (a foreign key pointing at a table of the query) that the query
                         does not use: the join fans the rows out
         drop_join       remove a JOIN whose table is referenced nowhere but its ON clause
  sem    drop_where      remove one top-level WHERE conjunct (seeded; up to two different conjuncts)
         drop_join_filter remove a JOIN together with the WHERE conjuncts that use it (and nothing else uses it)
         flip_order      ASC <-> DESC on the first ORDER BY key
  proj   proj_drop       remove one output column (SELECT with >= 2 columns)
         proj_add        append the first ORDER BY expression (when not selected), else one more column of the first
                         projected column's table
         proj_reorder    swap the first two output columns
         id_name         a selected id-like column -> a name-like column of the same table, or the reverse

Not edited: SQL with LIMIT but no ORDER BY (arbitrary rows). Never touched: LIMIT (ties are a 50/50 gold convention on DECISION). Only SQL whose sqlglot round trip reproduces the
text exactly is edited, so the edit is the only textual difference (same rule as scripts/val2/mutate.py). The caller
executes every edit and keeps it only when it runs, is non-null, is not right, and its result is new for the list.

schema: {table: {"cols": [names in order], "pk": [names], "fks": [(col, ref_table, ref_col)]}} (see schema_of()).
"""
from __future__ import annotations

import random
import re
import sqlite3

from sqlglot import exp

import sqldiff

GRAIN = ("count_distinct", "count_key", "add_join", "drop_join")
SEM = ("drop_where", "drop_join_filter", "flip_order")
PROJ = ("proj_drop", "proj_add", "proj_reorder", "id_name")
KINDS = GRAIN + SEM + PROJ
BUCKET_OF = {**{k: "grain" for k in GRAIN}, **{k: "sem" for k in SEM}, **{k: "proj" for k in PROJ}}
_ID = re.compile(r"(^id$|_id$|id$|^id_|code$|key$)", re.I)
_NAME = re.compile(r"(name|title|^first|^last|label)", re.I)


def schema_of(path: str) -> dict:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tabs = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        out = {}
        for t in tabs:
            info = list(con.execute(f'PRAGMA table_info("{t}")'))
            fks = [(r[3], r[2], r[4]) for r in con.execute(f'PRAGMA foreign_key_list("{t}")')]
            out[t] = {"cols": [r[1] for r in info], "pk": [r[1] for r in info if r[5]], "fks": fks}
        return out
    finally:
        con.close()


def parse_identity(sql: str):
    t = sqldiff.parse(sql)
    if t is None or t.sql(dialect="sqlite") != sql:
        return None
    return t


def _from(t):
    f = t.args.get("from_") or t.args.get("from")
    return f.this if f is not None and isinstance(f.this, exp.Table) else None


def _aliases(t) -> dict:
    """alias (or bare name) -> table name for the outer FROM / JOIN tables."""
    out = {}
    srcs = [_from(t)] + [j.this for j in t.args.get("joins") or []]
    for tb in srcs:
        if isinstance(tb, exp.Table):
            out[tb.alias_or_name] = tb.name
    return out


def _col(name: str, qual: str | None):
    return exp.Column(this=exp.to_identifier(name, quoted=True), table=exp.to_identifier(qual) if qual else None)


def _join_keys(t) -> list:
    """Column nodes used in the outer joins' ON equalities, in order."""
    out = []
    for j in t.args.get("joins") or []:
        on = j.args.get("on")
        if on is not None:
            out += [c for c in on.find_all(exp.Column)]
    return out


def _uses(t, alias: str, skip=None) -> bool:
    """Is the alias referenced anywhere in t outside the node skip (and its subtree)?"""
    skip_ids = {id(n) for n in skip.walk()} if skip is not None else set()
    return any(c.table == alias and id(c) not in skip_ids for c in t.find_all(exp.Column))


def e_count_distinct(t, schema, rnd):
    t = t.copy()
    for c in t.find_all(exp.Count):
        a = c.this
        if isinstance(a, exp.Distinct) and len(a.expressions) == 1:
            c.set("this", a.expressions[0].copy())
            return t
        if isinstance(a, exp.Column):
            c.set("this", exp.Distinct(expressions=[a.copy()]))
            return t
    return None


def e_count_key(t, schema, rnd):
    t = t.copy()
    keys = _join_keys(t)
    cnt = next(iter(t.find_all(exp.Count)), None)
    if cnt is None or not keys:
        return None
    a = cnt.this
    if isinstance(a, exp.Star):
        f = _from(t)
        alias = f.alias_or_name if f is not None else None
        k = next((c for c in keys if c.table == alias), None)
        if k is None:
            return None
        cnt.set("this", exp.Distinct(expressions=[k.copy()]))
        return t
    inner = a.expressions[0] if isinstance(a, exp.Distinct) and len(a.expressions) == 1 else a
    if not isinstance(inner, exp.Column):
        return None
    cur = inner.sql(dialect="sqlite")
    alts = []
    for c in keys:
        s = c.sql(dialect="sqlite")
        if s != cur and s not in [x.sql(dialect="sqlite") for x in alts]:
            alts.append(c)
    if not alts:
        return None
    inner.replace(rnd.choice(alts).copy())
    return t


def e_add_join(t, schema, rnd):
    t = t.copy()
    al = _aliases(t)
    used = set(al.values())
    if not al:
        return None
    nums = [int(m.group(1)) for a in al for m in [re.fullmatch(r"T(\d+)", a)] if m]
    opts = []
    for alias, tab in al.items():
        for child, info in schema.items():
            if child in used:
                continue
            for col, ref_t, ref_c in info["fks"]:
                if ref_t == tab and (ref_c or (schema.get(tab, {}).get("pk") or [None])[0]):
                    opts.append((alias, tab, child, col, ref_c or schema[tab]["pk"][0]))
    if not opts:
        return None
    alias, tab, child, col, ref_c = rnd.choice(sorted(opts))
    new = f"T{max(nums) + 1}" if nums else None
    if new is None:  # unaliased query: only add when the child name cannot clash
        return None
    cond = exp.EQ(this=_col(ref_c, alias), expression=_col(col, new))
    j = exp.Join(this=exp.Table(this=exp.to_identifier(child, quoted=True), alias=exp.TableAlias(this=exp.to_identifier(new))),
                 on=cond, kind="INNER")
    t.set("joins", list(t.args.get("joins") or []) + [j])
    return t


def _drop_join(t, with_filter: bool, rnd):
    t = t.copy()
    joins = list(t.args.get("joins") or [])
    where = t.args.get("where")
    conj = sqldiff.conjuncts(where)
    opts = []
    for k, j in enumerate(joins):
        if not isinstance(j.this, exp.Table):
            continue
        alias = j.this.alias_or_name
        # other joins must not depend on this one through their ON
        if any(c.table == alias for jj in joins[k + 1:] for c in (jj.args.get("on").find_all(exp.Column)
                                                                    if jj.args.get("on") is not None else [])):
            continue
        refs = [c for c in t.find_all(exp.Column) if c.table == alias and not any(c is x for x in j.walk())]
        if not refs:
            if not with_filter:
                opts.append((k, []))
            continue
        if not with_filter:
            continue
        users = [c for c in conj if any(x.table == alias for x in c.find_all(exp.Column))]
        rest = [c for c in conj if c not in users]
        in_users = {id(x) for u in users for x in u.walk()}
        if users and all(id(c) in in_users for c in refs) and len(rest) >= 0:
            opts.append((k, users))
    if not opts:
        return None
    k, users = rnd.choice(opts)
    t.set("joins", [j for i, j in enumerate(joins) if i != k] or None)
    if users:
        rest = [c for c in sqldiff.conjuncts(t.args.get("where")) if c not in users]
        _set_where(t, rest)
    return t


def _set_where(t, conj):
    if not conj:
        t.set("where", None)
        return
    node = conj[0].copy()
    for c in conj[1:]:
        node = exp.And(this=node, expression=c.copy())
    t.set("where", exp.Where(this=node))


def e_drop_join(t, schema, rnd):
    return _drop_join(t, False, rnd)


def e_drop_join_filter(t, schema, rnd):
    return _drop_join(t, True, rnd)


def e_drop_where(t, schema, rnd, which: int = 0):
    t = t.copy()
    conj = sqldiff.conjuncts(t.args.get("where"))
    if not conj or which >= len(conj):
        return None
    order = list(range(len(conj)))
    rnd.shuffle(order)
    k = order[which]
    _set_where(t, [c for i, c in enumerate(conj) if i != k])
    return t


def e_flip_order(t, schema, rnd):
    t = t.copy()
    o = t.args.get("order")
    if o is None or not o.expressions:
        return None
    first = o.expressions[0]
    desc = not first.args.get("desc")
    first.set("desc", desc)
    first.set("nulls_first", not desc)  # SQLite's default null order, so no NULLS FIRST/LAST clause is printed
    return t


def e_proj_drop(t, schema, rnd):
    t = t.copy()
    if len(t.expressions) < 2:
        return None
    k = rnd.randrange(len(t.expressions))
    t.set("expressions", [e for i, e in enumerate(t.expressions) if i != k])
    return t


def e_proj_add(t, schema, rnd):
    t = t.copy()
    sel = [sqldiff._unalias(e).sql(dialect="sqlite") for e in t.expressions]
    o = t.args.get("order")
    if o is not None and o.expressions:
        ex = o.expressions[0].this
        if ex.sql(dialect="sqlite") not in sel and not ex.find(exp.AggFunc) and not isinstance(ex, exp.Literal):
            t.set("expressions", t.expressions + [ex.copy()])
            return t
    cols = list(t.expressions[0].find_all(exp.Column)) if t.expressions else []
    if not cols:
        return None
    al = _aliases(t)
    qual = cols[0].table
    name = al.get(qual) if qual else (next(iter(al.values())) if len(set(al.values())) == 1 else None)
    if not name or name not in schema:
        return None
    used = {c.name for e in t.expressions for c in e.find_all(exp.Column)}
    choices = [c for c in schema[name]["cols"] if c not in used]
    if not choices:
        return None
    t.set("expressions", t.expressions + [_col(rnd.choice(choices), qual or None)])
    return t


def e_proj_reorder(t, schema, rnd):
    t = t.copy()
    ex = list(t.expressions)
    if len(ex) < 2:
        return None
    ex[0], ex[1] = ex[1], ex[0]
    t.set("expressions", ex)
    return t


def e_id_name(t, schema, rnd):
    t = t.copy()
    al = _aliases(t)
    for k, e in enumerate(t.expressions):
        c = sqldiff._unalias(e)
        if not isinstance(c, exp.Column):
            continue
        tab = al.get(c.table) if c.table else (next(iter(al.values())) if len(set(al.values())) == 1 else None)
        if not tab or tab not in schema:
            continue
        cols = schema[tab]["cols"]
        sel = {x.name for x in t.find_all(exp.Column)}
        if _ID.search(c.name) or c.name in schema[tab]["pk"]:
            alt = [x for x in cols if _NAME.search(x) and x not in sel]
        elif _NAME.search(c.name):
            alt = [x for x in schema[tab]["pk"] + [x for x in cols if _ID.search(x)] if x not in sel]
        else:
            continue
        if not alt:
            continue
        c.replace(_col(alt[0], c.table or None))
        return t
    return None


FUNCS = {"count_distinct": e_count_distinct, "count_key": e_count_key, "add_join": e_add_join,
         "drop_join": e_drop_join, "drop_where": e_drop_where, "drop_join_filter": e_drop_join_filter,
         "flip_order": e_flip_order, "proj_drop": e_proj_drop, "proj_add": e_proj_add,
         "proj_reorder": e_proj_reorder, "id_name": e_id_name}


def edits(sql: str, schema: dict | None = None, seed: str = "", kinds=KINDS) -> list[tuple[str, str]]:
    """[(kind, edited sql)] for every edit that applies, in KINDS order (drop_where may give two); [] when the SQL does
    not round-trip through sqlglot unchanged. Deterministic for a given seed."""
    t = parse_identity(sql)
    if t is None:
        return []
    if t.args.get("limit") is not None and t.args.get("order") is None:
        return []  # LIMIT n without ORDER BY returns arbitrary rows: any edit could change the result by accident
    schema = schema or {}
    out, seen = [], {sql}
    for kind in kinds:
        reps = 2 if kind == "drop_where" else 1
        for w in range(reps):
            rnd = random.Random(f"valhard:{seed}:{kind}")  # drop_where: one permutation, conjunct w of it
            try:
                m = e_drop_where(t, schema, rnd, w) if kind == "drop_where" else FUNCS[kind](t, schema, rnd)
                s = m.sql(dialect="sqlite") if m is not None else None
            except Exception:
                s = None
            if s and s not in seen:
                seen.add(s)
                out.append((kind, s))
    return out
