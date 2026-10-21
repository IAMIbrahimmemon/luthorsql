"""Generator misses: covered questions where NO candidate of the R1 u R2 wide pool (k=32 each) is right.

Analysis only (no generator change, no GPU, no paid API). Never reads BIRD dev, sealed, gate or RESERVED data.

  uv run python scripts/analysis/gen_misses.py exec   [--workers 6]   # execute gold + every distinct candidate
  uv run python scripts/analysis/gen_misses.py report                 # classify, print tables -> runs/analysis/gen_misses/

Sets:
  dec    DECISION (data/sys1/heldout_v2/decision.json) questions of runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl
         that are grammar-covered and have no ok candidate (111).
  val4   the 4 val2 validation DBs (citeseer, disney, professional_basketball, trains): R1 u R2 cands_train.jsonl,
         covered, no ok candidate. (data/val2/val_lists.jsonl itself drops no-right lists, so it has none.)
  train  every other BIRD-train question of R1 u R2 cands_train.jsonl, covered, no ok candidate (in-sample).

Per question: gold and each distinct candidate are re-executed read-only (gold 30 s, candidate 5 s). The closest
candidate is chosen (a) by AST distance over normalised clause features (tables, join pairs, projections, predicates,
group/order/limit), tie-broken by sqlglot tree-diff size, and (b) by result overlap (best per-gold-column value Jaccard).
Error classes are assigned automatically from the clause diff against the AST-closest live candidate plus pool-level
checks (a gold literal / column / table that NO candidate uses), then overridden by MANUAL labels where read by hand.
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import re
import sqlite3
import time
from pathlib import Path

import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/analysis/gen_misses"
DBS = ROOT / "data/raw/bird_train/train_databases"
VAL_DBS = {"citeseer", "disney", "professional_basketball", "trains"}
FORBIDDEN = ("bird_dev", "gate.json", "sealed", "score_")


def _guard(p: Path) -> Path:
    assert not any(f in str(p) for f in FORBIDDEN), p
    return p


def load_jsonl(p):
    return [json.loads(x) for x in open(_guard(Path(p)))]


# ----------------------------------------------------------------------------------------------------------- sets
def union_train():
    by = {}
    for run in ("r1", "r2"):
        for r in load_jsonl(ROOT / f"runs/phase1/p1w/p1w/{run}/cands_train.jsonl"):
            if r["id"] not in by:
                by[r["id"]] = dict(r, cands=[])
            seen = {c["sql"] for c in by[r["id"]]["cands"]}
            for c in r["cands"]:
                if c["sql"] not in seen:
                    by[r["id"]]["cands"].append(dict(c, src=run))
                    seen.add(c["sql"])
    return by


def miss_sets():
    dec_ids = set(json.load(open(_guard(ROOT / "data/sys1/heldout_v2/decision.json")))["ids"])
    reserved = set(json.load(open(_guard(ROOT / "data/sys1/heldout_v2/reserved.json")))["ids"])
    dec = [r for r in load_jsonl(ROOT / "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl")
           if r["id"] in dec_ids and r["id"] not in reserved]
    sets = {"dec": [r for r in dec if r["covered"] and not any(c.get("ok") for c in r["cands"])]}
    tr = union_train()
    tm = [r for r in tr.values() if r["covered"] and not any(c.get("ok") for c in r["cands"])]
    sets["val4"] = [r for r in tm if r["db_id"] in VAL_DBS]
    sets["train"] = [r for r in tm if r["db_id"] not in VAL_DBS]
    # context for rank curves
    sets["_dec_all"] = dec
    return sets


# ------------------------------------------------------------------------------------------------------- execution
def _run(con, box, sql, secs, cap=2000):
    box["t"] = time.time() + secs
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(100_000)
        return {"st": "ok", "n": len(rows), "ncol": len(cur.description or []),
                "rows": [tuple(r) for r in rows[:cap]], "set": hash(frozenset(map(tuple, rows)))}
    except Exception as e:  # noqa: BLE001
        return {"st": "timeout" if "interrupted" in str(e) else "error", "err": str(e)[:120]}


def exec_one(task):
    qid, db, gold, sqls = task
    con = sqlite3.connect(f"file:{DBS / db / (db + '.sqlite')}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    box = {"t": 0.0}
    con.set_progress_handler(lambda: 1 if time.time() > box["t"] else 0, 10_000)
    out = {"gold": _run(con, box, gold, 30.0), "cands": {}}
    for s in sqls:
        if s and s not in out["cands"]:
            out["cands"][s] = _run(con, box, s, 5.0)
    return qid, out


def cmd_exec(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    sets = miss_sets()
    tasks = []
    for name in ("dec", "val4", "train"):
        tasks += [(r["id"], r["db_id"], r["gold_sql"], [c["sql"] for c in r["cands"]]) for r in sets[name]]
    from multiprocessing import get_context
    res = {}
    t0 = time.time()
    with get_context("spawn").Pool(workers) as p:
        for i, (qid, o) in enumerate(p.imap_unordered(exec_one, tasks)):
            res[qid] = o
            if i % 50 == 0:
                print(f"{i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    pickle.dump(res, open(OUT / "exec.pkl", "wb"))
    print(f"done {len(res)} in {time.time() - t0:.0f}s")


# -------------------------------------------------------------------------------------------------- schema / values
_SCHEMA = {}


def schema(db):
    if db not in _SCHEMA:
        con = sqlite3.connect(f"file:{DBS / db / (db + '.sqlite')}?mode=ro", uri=True)
        tabs = [t for (t,) in con.execute("select name from sqlite_master where type='table'")]
        cols = {t.lower(): [c[1].lower() for c in con.execute(f'pragma table_info("{t}")')] for t in tabs}
        _SCHEMA[db] = (con, cols)
    return _SCHEMA[db]


def value_in_db(db, table, col, lit):
    """'exact' | 'nocase' | 'absent' | '?' for a string literal compared with table.col."""
    con, cols = schema(db)
    if not table or table not in cols or col not in cols[table]:
        return "?"
    try:
        if con.execute(f'select 1 from "{table}" where "{col}" = ? limit 1', (lit,)).fetchone():
            return "exact"
        if con.execute(f'select 1 from "{table}" where lower(trim("{col}")) = lower(trim(?)) limit 1', (lit,)).fetchone():
            return "nocase"
        return "absent"
    except Exception:  # noqa: BLE001
        return "?"


# --------------------------------------------------------------------------------------------------- AST features
AGGS = (exp.Count, exp.Sum, exp.Avg, exp.Max, exp.Min)


def parse(sql):
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        return None


def qualify(tree, db):
    """Replace every column by table.col (lower-case real table names), drop table aliases."""
    _, cols = schema(db)
    tree = tree.copy()
    for sel in tree.find_all(exp.Select):
        amap = {}
        for t in sel.find_all(exp.Table):
            if t.find_ancestor(exp.Select) is not sel:
                continue
            amap[(t.alias_or_name or t.name).lower()] = t.name.lower()
            amap[t.name.lower()] = t.name.lower()
        tabs = list(dict.fromkeys(amap.values()))
        for c in sel.find_all(exp.Column):
            if c.find_ancestor(exp.Select) is not sel:
                continue
            name = c.name.lower()
            tq = c.table.lower() if c.table else None
            real = amap.get(tq) if tq else next((t for t in tabs if name in cols.get(t, [])), tabs[0] if tabs else None)
            c.replace(exp.column(name, table=real or tq or "?", quoted=False))
    for t in tree.find_all(exp.Table):
        t.set("alias", None)
        t.set("this", exp.to_identifier(t.name.lower()))
    return tree


def lit_skel(e):
    e = e.copy()
    for lit in e.find_all(exp.Literal):
        lit.replace(exp.Placeholder())
    return e.sql()


def conjuncts(e):
    if e is None:
        return []
    if isinstance(e, exp.And):
        return conjuncts(e.left) + conjuncts(e.right)
    if isinstance(e, exp.Paren):
        return conjuncts(e.this)
    return [e]


def features(sql, db):
    t = parse(sql)
    if t is None:
        return None
    try:
        q = qualify(t, db)
    except Exception:  # noqa: BLE001
        return None
    top = q if isinstance(q, exp.Select) else (q.find(exp.Select) or q)
    f = {"tree": q}
    f["tables"] = frozenset(x.name for x in q.find_all(exp.Table))
    joins = set()
    for j in q.find_all(exp.Join):
        for eq in (j.args.get("on").find_all(exp.EQ) if j.args.get("on") else []):
            if isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column):
                joins.add(frozenset((eq.left.sql(), eq.right.sql())))
    f["joins"] = frozenset(joins)
    projs = [p.unalias() if isinstance(p, exp.Alias) else p for p in top.expressions]
    f["proj"] = tuple(p.sql() for p in projs)
    f["proj_cols"] = frozenset(c.sql() for p in projs for c in p.find_all(exp.Column))
    f["aggs"] = tuple(sorted(("COUNT_DISTINCT" if isinstance(a, exp.Count) and a.find(exp.Distinct) else a.key.upper())
                             for p in projs for a in p.find_all(*AGGS)))
    f["arith"] = any(p.find(exp.Div, exp.Mul, exp.Sub, exp.Add, exp.Cast) is not None for p in projs)
    f["distinct"] = bool(top.args.get("distinct"))
    where = top.args.get("where")
    preds = conjuncts(where.this) if where else []
    having = top.args.get("having")
    preds_h = conjuncts(having.this) if having else []
    f["preds"] = frozenset(p.sql() for p in preds)
    f["pred_skel"] = frozenset(lit_skel(p) for p in preds)
    f["having"] = frozenset(lit_skel(p) for p in preds_h)
    f["group"] = frozenset(g.sql() for g in (top.args.get("group").expressions if top.args.get("group") else []))
    o = top.args.get("order")
    f["order"] = tuple(x.sql() for x in o.expressions) if o else ()
    lim = top.args.get("limit")
    f["limit"] = lim.sql() if lim else ""
    f["subq"] = sum(1 for s in q.find_all(exp.Select)) - 1
    f["setop"] = q.find(exp.Union, exp.Intersect, exp.Except) is not None
    f["all_cols"] = frozenset(c.sql() for c in q.find_all(exp.Column))
    lits = []
    for p in list(q.find_all(exp.Literal)):
        if p.is_string:
            cmp_ = p.find_ancestor(exp.EQ, exp.Like, exp.In, exp.NEQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.Between)
            col = cmp_.find(exp.Column) if cmp_ is not None else None
            lits.append((p.this, col.table if col is not None else None, col.name if col is not None else None,
                         "like" if isinstance(cmp_, exp.Like) else "eq"))
    f["str_lits"] = tuple(lits)
    f["num_lits"] = frozenset(p.this for p in q.find_all(exp.Literal) if not p.is_string)
    return f


CLAUSES = ("tables", "joins", "proj", "aggs", "pred_skel", "preds", "having", "group", "order", "limit", "distinct", "subq")


def ast_dist(g, c):
    d = 0.0
    for k in CLAUSES:
        a, b = g[k], c[k]
        if isinstance(a, frozenset):
            d += len(a ^ b)
        elif a != b:
            d += 1 + (abs(len(a) - len(b)) if isinstance(a, tuple) else 0)
    return d


def tree_diff(g, c):
    try:
        return sum(1 for e in sqlglot.diff(g["tree"], c["tree"], delta_only=True))
    except Exception:  # noqa: BLE001
        return 999


# ------------------------------------------------------------------------------------------------ result matching
def _norm(v):
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, str):
        return v.strip()
    return v


def colsets(rows, ncol):
    return [set(_norm(r[i]) for r in rows) for i in range(ncol)]


def jac(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


def overlap(g, c):
    """Mean over gold columns of the best value-set Jaccard against any candidate column (shape-agnostic)."""
    if g.get("st") != "ok" or c.get("st") != "ok" or not g["ncol"] or not c["ncol"]:
        return 0.0
    gc, cc = colsets(g["rows"], g["ncol"]), colsets(c["rows"], c["ncol"])
    return sum(max(jac(a, b) for b in cc) for a in gc) / len(gc)


def lenient(g, c):
    """Name of the first lenient rule under which the candidate matches the gold, else ''."""
    if g.get("st") != "ok" or c.get("st") != "ok" or not g["rows"] or not c["rows"]:
        return ""
    G = {tuple(map(_norm, r)) for r in g["rows"]}
    C = [tuple(map(_norm, r)) for r in c["rows"]]
    if G == set(C):
        return "int_float_round"
    if g["ncol"] == c["ncol"]:
        import itertools
        if g["ncol"] <= 4:
            for perm in itertools.permutations(range(c["ncol"])):
                if {tuple(r[i] for i in perm) for r in C} == G:
                    return "col_order"

        def scale(rows, k):
            return {tuple(round(v * k, 2) if isinstance(v, (int, float)) and not isinstance(v, bool) else v for v in r)
                    for r in rows}
        for k, nm in ((100, "x100"), (0.01, "div100")):
            if scale(C, k) == scale(G, 1):
                return nm
        if {tuple(round(v, 1) if isinstance(v, float) else v for v in r) for r in C} == \
                {tuple(round(v, 1) if isinstance(v, float) else v for v in r) for r in G}:
            return "round1"
    if c["ncol"] > g["ncol"]:  # gold columns are a projection of the candidate's (extra column)
        import itertools
        for idx in itertools.permutations(range(c["ncol"]), g["ncol"]):
            if {tuple(r[i] for i in idx) for r in C} == G:
                return "extra_cols"
    if g["ncol"] > c["ncol"]:
        import itertools
        for idx in itertools.permutations(range(g["ncol"]), c["ncol"]):
            if {tuple(r[i] for i in idx) for r in G} == set(C):
                return "missing_cols"
    if G and set(C) and (set(C) < G or G < set(C)):
        return "subset" if set(C) < G else "superset"
    return ""


def scaled_match(g, c, factors):
    if g.get("st") != "ok" or c.get("st") != "ok" or not g["rows"] or g["ncol"] != c["ncol"] or g["n"] != c["n"]:
        return False

    def num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    G = {tuple(round(v, 4) if num(v) else _norm(v) for v in r) for r in g["rows"]}
    for f in factors:
        if {tuple(round(v * f, 4) if num(v) else _norm(v) for v in r) for r in c["rows"]} == G:
            return True
    return False


def rows_set(e):
    return {tuple(map(_norm, r)) for r in e["rows"]} if e.get("st") == "ok" and e["n"] <= len(e["rows"]) else None


# ----------------------------------------------------------------------------------------------- classification
EV_LIT = re.compile(r"'([^']+)'")


def classify(r, ex):
    """Automatic multi-label classes for one miss + diagnostic record."""
    db = r["db_id"]
    g = features(r["gold_sql"], db)
    gres = ex["gold"]
    rec = {"id": r["id"], "db": db, "labels": set(), "notes": []}
    cands = []
    for c in r["cands"]:
        e = ex["cands"].get(c["sql"]) or {"st": "none"}
        f = features(c["sql"], db) if c["sql"] else None
        live = e.get("st") == "ok" and e.get("n", 0) > 0 and any(v is not None for row in e["rows"] for v in row)
        cands.append({"c": c, "e": e, "f": f, "live": live})
    rec["n_cands"] = len(cands)
    rec["n_live"] = sum(x["live"] for x in cands)
    rec["n_empty"] = sum(x["e"].get("st") == "ok" and x["e"].get("n", 0) == 0 for x in cands)
    rec["n_err"] = sum(x["e"].get("st") in ("error", "timeout", "none") for x in cands)
    rec["n_sigs"] = len({x["e"]["set"] for x in cands if x["live"]})
    rec["gold_st"] = gres.get("st")
    rec["gold_n"] = gres.get("n")
    rec["gold_ncol"] = gres.get("ncol")
    if gres.get("st") == "ok":
        vals = [v for row in gres["rows"] for v in row]
        if not gres["rows"]:
            rec["gold_shape"] = "empty"
        elif all(v is None for v in vals):
            rec["gold_shape"] = "null"
        elif gres["n"] == 1 and gres["ncol"] == 1 and vals[0] in (0, 0.0):
            rec["gold_shape"] = "zero"
        else:
            rec["gold_shape"] = "ok"
    else:
        rec["gold_shape"] = gres.get("st")
    for x in cands:
        x["ov"] = overlap(gres, x["e"]) if x["live"] else 0.0
        x["len"] = lenient(gres, x["e"]) if x["live"] else ""
        x["ad"] = ast_dist(g, x["f"]) if (g and x["f"]) else 99
    rec["lenient"] = sorted({x["len"] for x in cands if x["len"]})
    # result-level arithmetic mutation: candidate numbers x factor (x100, /100, /N for N named in evidence/question)
    nums = {float(v) for v in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])", (r.get("context") or "") + " " + r["question"])}
    factors = {100.0, 0.01} | {1.0 / v for v in nums if v > 1}
    rec["arith_mut"] = any(scaled_match(gres, x["e"], factors) for x in cands if x["live"])
    deg = [x for x in cands if x["e"].get("st") == "ok"]
    rec["degenerate"] = bool(deg) and all(
        x["e"]["n"] == 0 or all(v in (None, 0, 0.0) for row in x["e"]["rows"] for v in row) for x in deg)
    pool = [x for x in cands if x["live"]] or [x for x in cands if x["f"]]
    if not g or not pool:
        rec["labels"].add("unparsed")
        return rec, None
    best_ast = min(pool, key=lambda x: (x["ad"], -x["ov"]))
    near = [x for x in pool if x["ad"] <= best_ast["ad"] + 0.5]
    if len(near) > 1:
        best_ast = min(near, key=lambda x: (tree_diff(g, x["f"]), -x["ov"]))
    best_res = max(pool, key=lambda x: (x["ov"], -x["ad"]))
    rec["best_ast"] = {"sql": best_ast["c"]["sql"], "rank": best_ast["c"]["rank"], "src": best_ast["c"].get("src"),
                       "ad": best_ast["ad"], "ov": round(best_ast["ov"], 3), "res": best_ast["c"].get("res")}
    rec["best_res"] = {"sql": best_res["c"]["sql"], "rank": best_res["c"]["rank"], "ad": best_res["ad"],
                       "ov": round(best_res["ov"], 3), "res": best_res["c"].get("res")}
    c = best_ast["f"]
    L = rec["labels"]
    # table / join path
    if g["tables"] != c["tables"]:
        L.add("join_path")
    elif g["joins"] != c["joins"]:
        L.add("join_path")
    # projection
    if g["proj"] != c["proj"]:
        if g["aggs"] != c["aggs"]:
            ga, ca = collections.Counter(g["aggs"]), collections.Counter(c["aggs"])
            L.add("count_distinct" if {"COUNT", "COUNT_DISTINCT"} >= set((ga - ca) | (ca - ga)) and ga and ca else "aggregation")
        if g["arith"] or c["arith"]:
            if g["arith"] != c["arith"] or g["num_lits"] != c["num_lits"] or g["aggs"] == c["aggs"]:
                if g["arith"]:
                    L.add("arithmetic")
        if g["proj_cols"] != c["proj_cols"]:
            if len(g["proj"]) != len(c["proj"]):
                L.add("projection_count")
            if (g["proj_cols"] - c["proj_cols"]) and (c["proj_cols"] - g["proj_cols"]) or len(g["proj"]) == len(c["proj"]):
                L.add("projection_column")
    # filters
    ms, ex_ = g["pred_skel"] - c["pred_skel"], c["pred_skel"] - g["pred_skel"]
    if ms and not ex_:
        L.add("missing_filter")
    elif ex_ and not ms:
        L.add("extra_filter")
    elif ms and ex_:
        L.add("wrong_filter")
    elif g["preds"] != c["preds"]:
        L.add("wrong_literal")
    if g["having"] != c["having"] or g["group"] != c["group"]:
        L.add("grouping")
    if g["order"] != c["order"] or g["limit"] != c["limit"]:
        L.add("order_limit")
    if g["distinct"] != c["distinct"] and "projection_column" not in L:
        L.add("distinct")
    if g["subq"] != c["subq"] or g["setop"] != c["setop"]:
        L.add("nesting")
    # literals: values in the gold that no candidate writes
    pool_text = " ".join(x["c"]["sql"] or "" for x in cands)
    lit_rows = []
    for v, tab, col, kind in g["str_lits"]:
        if f"'{v}'" in pool_text or f"'{v.replace(chr(39), chr(39) * 2)}'" in pool_text:
            continue
        status = value_in_db(db, tab, col, v) if kind == "eq" else "like"
        near = [cv for cv, ct, cc, _ in c["str_lits"] if cc == col]
        lit_rows.append((v, f"{tab}.{col}", status, near[:2], [s for s in (value_in_db(db, tab, col, n) for n in near[:2])]))
    rec["missing_lits"] = lit_rows
    if lit_rows:
        L.add("literal_never_generated")
    # pool-level schema coverage
    used = set()
    for x in cands:
        if x["f"]:
            used |= x["f"]["all_cols"]
    miss_cols = sorted(g["all_cols"] - used)
    rec["cols_never_used"] = miss_cols
    if miss_cols:
        L.add("column_never_generated")
    # evidence: a quoted evidence literal or evidence-named column the gold uses but the closest candidate does not
    ev = r.get("context") or ""
    ev_lits = set(EV_LIT.findall(ev))
    g_lits = {v for v, *_ in g["str_lits"]}
    c_lits = {v for v, *_ in c["str_lits"]}
    if (ev_lits & g_lits) - c_lits:
        L.add("evidence_ignored")
    if re.search(r"divide|\*\s*100|percent|subtract|multiply|/", ev, re.I) and g["arith"] and not c["arith"]:
        L.add("evidence_ignored")
    rec["schema_size"] = sum(len(v) for v in schema(db)[1].values())
    return rec, best_ast


# -------------------------------------------------------------------------------------------- manual (read by hand)
# Labels read by hand for every DECISION miss: these override the automatic ones in the report (see the doc for
# definitions). Keys: id -> (labels, short note). Filled from the per-question dump (report --dump).
MANUAL_PATH = ROOT / "scripts/analysis/gen_misses_manual.json"


def verified_changes():
    ver = {}
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    for f in ("bird-verified-train.json", "bird-verified-val.json"):
        for x in json.load(open(ROOT / "data/raw/bird_verified" / f)):
            i = int(x["question_id"])
            ver[i] = (" ".join(x["SQL"].split()).rstrip(";") != " ".join(tr[i]["SQL"].split()).rstrip(";"), x["SQL"])
    return ver


def rank_curve(dec_all):
    """Oracle as a function of k per run (from candidate rank): how much do ranks 8..31 add?"""
    out = {}
    cov = [r for r in dec_all if r["covered"]]
    for k in (1, 2, 4, 8, 16, 24, 32):
        hit = 0
        for r in cov:
            ok = [c for c in r["cands"] if c.get("ok") and c["rank"] < k]
            hit += bool(ok)
        out[k] = hit
    return out, len(cov)


def cmd_report(dump):
    ex = pickle.load(open(OUT / "exec.pkl", "rb"))
    sets = miss_sets()
    ver = verified_changes()
    noise_drop = set(json.load(open(ROOT / "data/sys1/gold_noise_drop.json"))["ids"])
    manual = json.load(open(MANUAL_PATH)) if MANUAL_PATH.exists() else {}
    allrecs = {}
    for name in ("dec", "val4", "train"):
        recs = []
        for r in sets[name]:
            rec, _ = classify(r, ex[r["id"]])
            n = int(r["id"].rsplit("_", 1)[1])
            rec["verified_changed"] = ver.get(n, (None, None))[0]
            rec["verified_sql"] = ver.get(n, (None, None))[1] if ver.get(n, (False,))[0] else None
            rec["noise_drop"] = r["id"] in noise_drop
            rec["verified_rescue"] = None
            if rec["verified_sql"]:
                con = sqlite3.connect(f"file:{DBS / r['db_id'] / (r['db_id'] + '.sqlite')}?mode=ro", uri=True)
                con.text_factory = lambda b: b.decode("utf-8", "replace")
                box = {"t": 0.0}
                con.set_progress_handler(lambda: 1 if time.time() > box["t"] else 0, 10_000)
                v = _run(con, box, rec["verified_sql"], 30.0)
                vs = rows_set(v)
                rec["verified_rescue"] = bool(vs) and any(
                    rows_set(e) == vs for e in ex[r["id"]]["cands"].values() if e.get("st") == "ok")
            if r["id"] in manual:
                rec["fix"] = manual[r["id"]].get("fix")
            rec["auto"] = sorted(rec["labels"])
            if r["id"] in manual:
                rec["labels"] = list(manual[r["id"]]["labels"])  # first = primary
                rec["note"] = manual[r["id"]].get("note", "")
            if not isinstance(rec["labels"], list):
                rec["labels"] = sorted(rec["labels"])
            rec.update(question=r["question"], evidence=r["context"], gold=r["gold_sql"])
            recs.append(rec)
        allrecs[name] = recs
        with open(OUT / f"{name}.jsonl", "w") as f:
            for rec in recs:
                f.write(json.dumps(rec, default=list) + "\n")
    for name, recs in allrecs.items():
        n = len(recs)
        print(f"\n=== {name}: {n} misses ===")
        lab = collections.Counter(l for r in recs for l in r["labels"])
        auto = collections.Counter(l for r in recs for l in r["auto"])
        print("labels (final):", ", ".join(f"{k} {v}" for k, v in lab.most_common()))
        print("labels (auto): ", ", ".join(f"{k} {v}" for k, v in auto.most_common()))
        print("cheap signals: zero-live", sum(r["n_live"] == 0 for r in recs),
              "| all-live-same-result", sum(r["n_sigs"] == 1 for r in recs),
              "| any-empty-cand", sum(r["n_empty"] > 0 for r in recs),
              "| lenient-right", sum(bool(r["lenient"]) for r in recs),
              collections.Counter(x for r in recs for x in r["lenient"]))
        print("gold shape", collections.Counter(r["gold_shape"] for r in recs),
              "| verified changed", sum(bool(r["verified_changed"]) for r in recs),
              "| has verified entry", sum(r["verified_changed"] is not None for r in recs),
              "| noise_drop", sum(r["noise_drop"] for r in recs))
        print("by db", collections.Counter(r["db"] for r in recs).most_common(12))
        strict = {"col_order", "int_float_round", "x100", "div100", "round1", "extra_cols"}
        print("lenient strict (col order / numeric / extra col)", sum(bool(set(r["lenient"]) & strict) for r in recs),
              "| col_order only", sum("col_order" in r["lenient"] for r in recs),
              "| arith mutation (x100,/100,/N evidence) matches", sum(r["arith_mut"] for r in recs),
              "| degenerate pool (all empty/0/NULL)", sum(r["degenerate"] for r in recs),
              "| Verified gold matched by a candidate", sum(bool(r["verified_rescue"]) for r in recs))
        if any("fix" in r for r in recs):
            print("primary label", collections.Counter(r["labels"][0] for r in recs).most_common())
            print("fix", collections.Counter(r.get("fix") for r in recs).most_common())
            print("fix x db", {d: dict(collections.Counter(r.get("fix") for r in recs if r["db"] == d))
                               for d in sorted({r["db"] for r in recs})})
    curve, ncov = rank_curve(sets["_dec_all"])
    print("\nDECISION oracle vs per-run rank cutoff (R1-ranked union list, covered n=%d):" % ncov, curve)
    if dump:
        for rec in allrecs[dump]:
            print(f"\n## {rec['id']} {rec['db']} | {rec['question']}\n EV: {rec['evidence']}\n GOLD: {rec['gold']}")
            print(f" gold {rec['gold_shape']} n={rec['gold_n']}x{rec['gold_ncol']} | live {rec['n_live']}/{rec['n_cands']} "
                  f"sigs {rec['n_sigs']} | lenient {rec['lenient']} | verified_changed {rec['verified_changed']}")
            if rec.get("verified_sql"):
                print(" VERIFIED:", rec["verified_sql"][:400])
            for k in ("best_ast", "best_res"):
                if rec.get(k):
                    b = rec[k]
                    print(f" {k} r{b['rank']} ad={b['ad']} ov={b['ov']} [{(b['res'] or '')[:60]}]: {b['sql'][:420]}")
            print(" auto:", rec["auto"], "| lits", rec.get("missing_lits"), "| cols", rec.get("cols_never_used"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("exec", "report"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    if a.cmd == "exec":
        cmd_exec(a.workers)
    else:
        cmd_report(a.dump)
