"""Step B, part 1: sample valid plans from REAL public databases, for question writing.

  python scripts/sample_plans.py --cards DIR --db-root DIR --shapes PLANS.jsonl [...] --per-db 300 --out data/qgen/plans.jsonl

1. Plan *shapes* (how many projections, which aggregations, filter kinds, group/order/limit,
   HAVING, subquery) are learned from converted human questions (Spider / BIRD train plans),
   so sampled plans look like what people actually ask.
2. Each shape is filled with this database's own columns and stored values: text columns with
   few distinct values for grouping and '=' filters, numeric columns for sums and thresholds
   (taken from the column's real quantiles), date columns for year filters.
3. A plan is kept only if it compiles, runs (5 s limit) and returns a sensible result: rows,
   not all NULL, at most 200 rows unless aggregated/limited. Duplicates (same SQL) are dropped.
4. Each kept plan gets a plain-English rendering (from the schema descriptions) and its first
   result rows: the inputs the question-writing subagents see.
Public data only.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luthor.sq.plan import ROWS, Col, CompileError, Filter, Having, Plan, Proj, compile_plan  # noqa: E402
from luthor.sq.schema_card import SchemaCard, natural  # noqa: E402

NUM = re.compile(r"int|real|float|double|decimal|numeric|number", re.I)
DATE = re.compile(r"date|time|year", re.I)


# ---------------------------------------------------------------- shapes from real plans
def shape_of(p: dict) -> tuple:
    projs = tuple("count" if x["col"]["column"] == "*" else x["agg"] for x in p["projections"])
    fl = tuple(sorted(("sub" if f.get("sub") else f["op"]) + ("/year" if f.get("part", "none") != "none" else "")
                      for f in p.get("filters") or []))
    return (projs, p.get("compose", "none"), any(x.get("cond") for x in p["projections"]), fl,
            len(p.get("group_by") or []), p.get("order_dir", "none") != "none",
            "none" if p.get("limit") is None else ("one" if p["limit"] == 1 else "n"),
            len(p.get("having") or []), bool(p.get("distinct")))


def load_shapes(paths: list[Path]) -> list[tuple]:
    shapes = []
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if r.get("outcome") == "plan" and not r.get("twin_of") and not r.get("empty_result") \
                    and r["plan"].get("set_op", "none") == "none":
                shapes.append(shape_of(r["plan"]))
    return shapes


# ---------------------------------------------------------------- column typing
class DBInfo:
    def __init__(self, card: SchemaCard, con: sqlite3.Connection):
        self.card, self.con = card, con
        self.graph = card.graph()
        self.cols: dict[str, dict] = {}
        fk_cols = {(a, b) for a, b, _, _ in card.fks + card.inferred_fks} | {(c, d) for _, _, c, d in card.fks + card.inferred_fks}
        for t in card.tables:
            if t.name.lower().startswith("sqlite_"):
                continue  # SQLite's own bookkeeping tables
            for c in t.columns:
                idish = c.pk or (t.name, c.name) in fk_cols or bool(re.search(r"(^|_)id$|id$", c.name, re.I))
                kind = "num" if NUM.search(c.type or "") else "text"
                if DATE.search(c.name) or DATE.search(c.type or ""):
                    kind = "date"
                vals = card.values.get(f"{t.name}.{c.name}", [])
                self.cols[f"{t.name}.{c.name}"] = {"table": t.name, "col": c.name, "kind": kind, "id": idish,
                                                   "nd": c.n_distinct, "vals": vals, "rows": self._rows(t.name)}

    def _rows(self, t: str) -> int:
        if not hasattr(self, "_rowcache"):
            self._rowcache = {}
        if t not in self._rowcache:
            try:
                self._rowcache[t] = self.con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            except sqlite3.Error:
                self._rowcache[t] = 0
        return self._rowcache[t]

    def pick(self, rnd, tables, pred, avoid=()):
        opts = [k for k, v in self.cols.items() if v["table"] in tables and pred(v) and k not in avoid]
        return rnd.choice(opts) if opts else None

    def quantile(self, cid: str, q: float):
        t, c = cid.split(".", 1)
        try:
            n = self.con.execute(f'SELECT COUNT("{c}") FROM "{t}"').fetchone()[0]
            if not n:
                return None
            v = self.con.execute(f'SELECT "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL ORDER BY "{c}" '
                                 f'LIMIT 1 OFFSET {int(n * q)}').fetchone()
            return v[0] if v else None
        except sqlite3.Error:
            return None

    def neighbourhood(self, rnd, t: str, k: int = 2) -> list[str]:
        out, frontier = [t], [t]
        for _ in range(k):
            nxt = []
            for x in frontier:
                for m, *_ in self.graph.adj.get(x, []):
                    if m not in out:
                        out.append(m)
                        nxt.append(m)
            frontier = nxt
        return out[: 1 + rnd.randint(0, 2)] if len(out) > 1 else out


def col(cid: str) -> Col:
    t, c = cid.split(".", 1)
    return Col(t, c)


# ---------------------------------------------------------------- instantiate a shape
CAT = lambda v: v["kind"] == "text" and not v["id"] and v["nd"] is not None and 2 <= v["nd"] <= 60 and v["vals"]
_CODEY = re.compile(r"code|zip|phone|postal|year|_no$|number$|^no$|_num$|rank$|order$|seq", re.I)
# a quantity worth summing or averaging: numeric, not an id or a code-like number
NUMV = lambda v: v["kind"] == "num" and not v["id"] and not _CODEY.search(v["col"])
NAMEISH = lambda v: v["kind"] == "text" and not v["id"]
# a grouping column must repeat: otherwise every group has one row and "the top one" is a tie
GROUPABLE = lambda v: CAT(v) and v["nd"] <= 0.5 * max(1, v["rows"])


def literal_filter(rnd, db: DBInfo, tables, op: str):
    if op.endswith("/year"):
        cid = db.pick(rnd, tables, lambda v: v["kind"] == "date")
        if not cid:
            return None
        y = db.quantile(cid, rnd.random())
        m = re.match(r"(\d{4})", str(y or ""))
        if not m:
            return None
        return Filter(col(cid), op.split("/")[0] if op.split("/")[0] in ("=", ">", "<", ">=", "<=") else "=", [m.group(1)], "year")
    if op in ("=", "!=", "in", "not_in", "like", "like_prefix"):
        cid = db.pick(rnd, tables, CAT if op in ("=", "!=", "in", "not_in") else NAMEISH)
        if not cid:
            return None
        vals = db.cols[cid]["vals"] or []
        if not vals:
            return None
        if op in ("in", "not_in"):
            return Filter(col(cid), op, rnd.sample(vals, min(len(vals), rnd.randint(2, 3))))
        v = rnd.choice(vals)
        if op == "like":
            words = [w for w in re.split(r"\W+", v) if len(w) > 3]
            return Filter(col(cid), "like", [rnd.choice(words)]) if words else None
        if op == "like_prefix":
            return Filter(col(cid), "like_prefix", [v[: max(2, len(v) // 2)]])
        return Filter(col(cid), op, [v])
    if op in (">", ">=", "<", "<=", "between"):
        cid = db.pick(rnd, tables, NUMV)
        if not cid:
            return None
        if op == "between":
            a, b = db.quantile(cid, 0.25), db.quantile(cid, 0.75)
            return Filter(col(cid), "between", [a, b]) if a is not None and b is not None and a != b else None
        q = db.quantile(cid, rnd.choice([0.2, 0.5, 0.8]))
        if isinstance(q, float):
            q = round(q, 2)
        return Filter(col(cid), op, [q]) if isinstance(q, (int, float)) else None
    if op == "sub":
        cid = db.pick(rnd, tables, NUMV)
        if not cid:
            return None
        return Filter(col(cid), rnd.choice([">", "<"]), [], sub=Plan([Proj("avg", col(cid))]))
    return None


def instantiate(rnd, db: DBInfo, shape: tuple) -> Plan | None:
    projs, compose, has_cond, fops, n_group, has_order, limit_kind, n_having, distinct = shape
    if compose != "none" or has_cond:
        return None  # sampled separately later; keep the sampler simple and reliable
    anchors = [t.name for t in db.card.tables if len(t.columns) >= 2 and not t.name.lower().startswith("sqlite_")]
    if not anchors:
        return None
    anchor = rnd.choice(anchors)
    tables = db.neighbourhood(rnd, anchor)
    plan = Plan(projections=[], distinct=distinct)
    group_cols = []
    for _ in range(n_group):
        g = db.pick(rnd, tables, GROUPABLE)
        if not g or g in group_cols:
            return None
        group_cols.append(g)
    plan.group_by = [col(g) for g in group_cols]
    used: list[str] = list(group_cols)
    for i, kind in enumerate(projs):
        if kind == "none":
            if group_cols and i >= len(group_cols):
                return None  # grouped results may only show grouped columns or aggregates
            cid = group_cols[i] if i < len(group_cols) else db.pick(rnd, tables, NAMEISH, avoid=used)
            if not cid:
                return None
            if i >= len(group_cols):
                used.append(cid)
            plan.projections.append(Proj("none", col(cid)))
        elif kind == "count":
            plan.projections.append(Proj("count", Col(rnd.choice(tables), ROWS)))
        elif kind == "count_distinct":
            cid = db.pick(rnd, tables, NAMEISH, avoid=used)
            if not cid:
                return None
            plan.projections.append(Proj("count_distinct", col(cid)))
        else:
            cid = db.pick(rnd, tables, NUMV, avoid=used)
            if not cid:
                return None
            used.append(cid)
            plan.projections.append(Proj(kind, col(cid)))
    if len({p.col.id for p in plan.projections}) < len(plan.projections):
        return None
    shown = {p.col.id for p in plan.projections}
    for op in fops:
        f = literal_filter(rnd, db, tables, op)
        if f is None or (f.col.id in shown and f.sub is None) or any(g.col.id == f.col.id for g in plan.filters):
            return None  # filtering on what is shown, or twice on one column, isn't how people ask
        plan.filters.append(f)
    if has_order:
        if group_cols:  # grouped results are ordered by an aggregate
            p = next((p for p in plan.projections if p.agg != "none" or p.col.column == ROWS), None)
            plan.order = Proj("count", Col(group_cols[0].split(".")[0], ROWS)) if p is None else Proj(p.agg, p.col)
        else:
            cid = db.pick(rnd, tables, NUMV)
            if not cid:
                return None
            plan.order = Proj("none", col(cid))
        plan.order_dir = rnd.choice(["asc", "desc", "desc"])
    single_total = not plan.group_by and all(p.agg != "none" or p.col.column == ROWS for p in plan.projections)
    if limit_kind == "one" and not single_total:
        plan.limit = 1
    elif limit_kind == "n" and not single_total:
        plan.limit = rnd.choice([3, 5, 10])
    if single_total:
        plan.order, plan.order_dir = None, "none"  # sorting one total is meaningless
    # a listing may combine columns only from tables that are directly linked
    if not plan.group_by and all(p.agg == "none" for p in plan.projections):
        ptabs = list(dict.fromkeys(p.col.table for p in plan.projections))
        if len(ptabs) > 2 or (len(ptabs) == 2 and ptabs[1] not in [m for m, *_ in db.graph.adj.get(ptabs[0], [])]):
            return None
    for _ in range(n_having):
        if not group_cols:
            return None
        plan.having.append(Having("count", Col(rnd.choice(tables), ROWS), ">", [rnd.choice([1, 2, 3, 5])]))
    return plan


# ---------------------------------------------------------------- English rendering
def describe(db: DBInfo, c: Col) -> str:
    if c.column == ROWS:
        return f"rows of {natural(c.table)}"
    cd = next((x for t in db.card.tables if t.name == c.table for x in t.columns if x.name == c.column), None)
    d = f" ({cd.description[:80]})" if cd and cd.description else ""
    return f"{c.table}.{c.column}{d}"


def english(db: DBInfo, p: Plan) -> str:
    aggw = {"none": "", "count": "the number of ", "count_distinct": "the number of distinct ", "sum": "the total of ",
            "avg": "the average of ", "min": "the minimum of ", "max": "the maximum of "}
    out = "show " + ", ".join(aggw[x.agg] + describe(db, x.col) for x in p.projections)
    if p.distinct:
        out += " (distinct)"
    conds = []
    for f in p.filters:
        target = f"the year of {describe(db, f.col)}" if f.part == "year" else describe(db, f.col)
        if f.sub is not None:
            conds.append(f"{target} {f.op} the average of {describe(db, f.sub.projections[0].col)} over all rows")
        elif f.op == "between":
            conds.append(f"{target} between {f.values[0]!r} and {f.values[1]!r}")
        elif f.op in ("in", "not_in"):
            conds.append(f"{target} {'is one of' if f.op == 'in' else 'is none of'} {f.values!r}")
        elif f.op == "like":
            conds.append(f"{target} contains {f.values[0]!r}")
        elif f.op == "like_prefix":
            conds.append(f"{target} starts with {f.values[0]!r}")
        else:
            conds.append(f"{target} {f.op} {f.values[0]!r}")
    if conds:
        out += "; only rows where " + " and ".join(conds)
    if p.group_by:
        out += "; for each " + " and ".join(describe(db, g) for g in p.group_by)
    for h in p.having:
        out += f"; keep only groups where {aggw[h.agg]}{describe(db, h.col)} {h.op} {h.values[0]}"
    if p.order is not None:
        out += f"; sorted by {aggw[p.order.agg]}{describe(db, p.order.col)} " + ("highest first" if p.order_dir == "desc" else "lowest first")
    if p.limit:
        out += f"; only the top {p.limit}" if p.limit > 1 else "; only the first one"
    return out


# ---------------------------------------------------------------- main
def run(con, sql, timeout=5.0):
    deadline = time.time() + timeout
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10_000)
    try:
        return con.execute(sql).fetchmany(501)
    finally:
        con.set_progress_handler(None, 0)


def sample_db(card: SchemaCard, sqlite_path: Path, shapes: list[tuple], n: int, seed: int) -> list[dict]:
    rnd = random.Random(seed)
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    db = DBInfo(card, con)
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 25:
        tries += 1
        p = instantiate(rnd, db, rnd.choice(shapes))
        if p is None:
            continue
        try:
            sql = compile_plan(p, db.graph)
        except CompileError:
            continue
        if sql in seen:
            continue
        try:
            rows = run(con, sql)
        except Exception:
            continue
        aggregated = any(x.agg != "none" or x.col.column == ROWS for x in p.projections) and not p.group_by
        if not rows or all(all(v is None for v in r) for r in rows):
            continue
        if not aggregated and p.limit is None and len(rows) > 200:
            continue
        seen.add(sql)
        out.append({"db_id": card.db_id, "plan": p.to_dict(), "sql": sql, "english": english(db, p),
                    "result_head": [list(map(str, r)) for r in rows[:3]], "result_rows": len(rows)})
    con.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=Path, required=True, help="folder of SchemaCard json files")
    ap.add_argument("--db-root", type=Path, required=True, help="folder with <db>/<db>.sqlite")
    ap.add_argument("--shapes", type=Path, nargs="+", required=True, help="converted human plans (plans.jsonl)")
    ap.add_argument("--per-db", type=int, default=300)
    ap.add_argument("--dbs", nargs="*", help="only these db_ids")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    shapes = load_shapes(a.shapes)
    print(f"{len(shapes)} human plan shapes, {len(set(shapes))} distinct", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(a.out, "w") as fh:
        for cp in sorted(a.cards.glob("*.json")):
            if a.dbs and cp.stem not in a.dbs:
                continue
            sp = a.db_root / cp.stem / f"{cp.stem}.sqlite"
            if not sp.exists():
                continue
            rows = sample_db(SchemaCard.load(cp), sp, shapes, a.per_db, seed=hash(cp.stem) & 0xFFFF)
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
            total += len(rows)
            print(f"  {cp.stem}: {len(rows)} plans", flush=True)
    print(f"{total} plans -> {a.out}")


if __name__ == "__main__":
    main()
