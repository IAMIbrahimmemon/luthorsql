"""System-1 pick gap on DECISION (Mac-only, existing scores; no GPU, no API).

System 1 = R1 grammar generator (k=32) + R1 validator (rr). Pick = max rr over live candidates (runs and nonnull; all
if none live). Oracle = any right candidate. Uncovered DECISION questions count wrong.

  .venv/bin/python scripts/analysis/system1_pick_gap.py > runs/analysis/system1_pick_gap.out

Reads ONLY runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl restricted to data/sys1/heldout_v2/decision.json ids (rows
with other ids are dropped at load time, before anything else looks at them).

Part 1: anatomy of every pick-miss (oracle right, pick wrong): rr rank of the best right candidate, rr margin, whether
the right candidate is the greedy (rank 0), self-consistency class sizes, preview relation and a clause-level SQL diff
(sqlglot, aliases resolved to table names, WHERE conjuncts as a set) -> one actionable bucket each.
Part 2: deterministic pick rules on the same scores; every fitted parameter is fitted leave-one-DB-out (grid, ties ->
the value closest to the baseline). DECISION pick and exact DB sign-flip p (8 DBs, 256 patterns) vs the baseline.
Note: ok = set(rows) == set(gold rows), so row order and duplicate rows never matter; column order and LIMIT
truncation do.
"""
from __future__ import annotations

import ast
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402

MANUAL = ROOT / "scripts/analysis/system1_pick_gap_manual.json"
UNION = ROOT / "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl"
DECISION = ROOT / "data/sys1/heldout_v2/decision.json"
OUT_JSONL = ROOT / "runs/analysis/system1_pick_gap_misses.jsonl"


# ----------------------------------------------------------------------------------------------------------- load
def load():
    dec = json.load(open(DECISION))
    ids = list(dec["ids"])
    idset = set(ids)
    Q = {}
    for line in open(UNION):
        r = json.loads(line)
        if r["id"] not in idset:
            continue  # never look at non-DECISION rows
        r["cands"] = [c for c in r["cands"] if c["src"] in ("r1", "both")]
        Q[r["id"]] = r
    return ids, Q


def live_of(C):
    L = [c for c in C if c["runs"] and c["nonnull"]]
    return L or C


RES_RE = re.compile(r"^(\d+) rows? x (\d+) cols? \((.*?)\); first: (.*)$")


def parse_res(s):
    s = s or ""
    if not s:
        return {"err": True, "n": None, "k": None, "types": None, "first": None}
    if s.startswith("0 rows"):
        return {"err": False, "n": 0, "k": None, "types": None, "first": None}
    m = RES_RE.match(s)
    if not m:
        return {"err": False, "n": None, "k": None, "types": None, "first": s}
    first = m.group(4)
    try:
        first = ast.literal_eval(first)
        if not isinstance(first, tuple):
            first = (first,)
    except Exception:
        pass
    return {"err": False, "n": int(m.group(1)), "k": int(m.group(2)), "types": m.group(3), "first": first}


# -------------------------------------------------------------------------------------------------- SQL features
AGG_RE = re.compile(r"\b(COUNT|SUM|AVG|MAX|MIN|TOTAL)\(", re.I)


def _norm_aliases(e):
    amap = {}
    for t in e.find_all(exp.Table):
        if t.alias:
            amap[t.alias] = t.name
    for col in e.find_all(exp.Column):
        if col.table and col.table in amap:
            col.set("table", exp.to_identifier(amap[col.table]))
    for t in e.find_all(exp.Table):
        t.set("alias", None)
    return e


def _conj(node):
    if node is None:
        return frozenset()
    out = []
    stack = [node.this if isinstance(node, exp.Where) else node]
    while stack:
        x = stack.pop()
        if isinstance(x, exp.And):
            stack += [x.left, x.right]
        elif isinstance(x, exp.Paren):
            stack.append(x.this)
        else:
            out.append(x.sql(dialect="sqlite"))
    return frozenset(out)


def feats(sql):
    try:
        e = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if not isinstance(e, exp.Select):
        return {"raw": sql}
    e = _norm_aliases(e.copy())
    joins = e.args.get("joins") or []
    tables = sorted(t.name.lower() for t in e.find_all(exp.Table))
    jconds = frozenset(s for j in joins for s in _conj(j.args.get("on")))
    g = lambda k: (e.args.get(k).sql(dialect="sqlite") if e.args.get(k) is not None else "")
    return {"select": [x.sql(dialect="sqlite") for x in e.expressions],
            "distinct": e.args.get("distinct") is not None,
            "from": (tuple(tables), jconds),
            "where": _conj(e.args.get("where")),
            "group": g("group"), "having": g("having"), "order": g("order"), "limit": g("limit")}


CLAUSES = ("select", "distinct", "from", "where", "group", "having", "order", "limit")


def clause_diff(fa, fb):
    if fa is None or fb is None or "raw" in fa or "raw" in fb:
        return None
    return [k for k in CLAUSES if fa[k] != fb[k]]


def strip_distinct(xs):
    return [re.sub(r"\bDISTINCT\s+", "", x, flags=re.I) for x in xs]


def agg_neutral(xs):
    return [AGG_RE.sub("AGG(", x) for x in xs]


SUPERLATIVE = re.compile(r"=\s*\(SELECT\s+(MAX|MIN)\(", re.I)


def bucket(p, r, fp, fr, rp, rr_):
    """Actionable bucket for (pick p, best right r)."""
    d = clause_diff(fp, fr)
    if d is None:
        return "unparsed", d
    dset = set(d)
    if not d:
        return "same-structure (literal/alias only)", d
    if dset <= {"select", "distinct"}:
        sp, sr = fp["select"], fr["select"]
        if sorted(sp) == sorted(sr) and sp != sr:
            return "projection: column order", d
        if set(sr) < set(sp):
            return "projection: pick has extra columns", d
        if set(sp) < set(sr):
            return "projection: pick misses columns", d
        if strip_distinct(sp) == strip_distinct(sr):
            return "COUNT(DISTINCT) vs COUNT", d
        if agg_neutral(sp) == agg_neutral(sr):
            return "aggregate function (COUNT/SUM/AVG...)", d
        if len(sp) == len(sr) and any(re.search(r"CAST|ROUND|\*\s*100|/|REAL|IIF|CASE", x, re.I) for x in sp + sr):
            return "arithmetic / cast / ratio expression", d
        if len(sp) != len(sr):
            return "projection: different columns (count differs)", d
        return "projection: wrong column(s)", d
    if dset <= {"order", "limit", "distinct"}:
        return "LIMIT / ties (ORDER-LIMIT only)", d
    if dset <= {"where", "order", "limit", "distinct"} and (
            bool(SUPERLATIVE.search(p["sql"])) != bool(SUPERLATIVE.search(r["sql"]))
            or (fp["limit"] != fr["limit"] and "where" in dset and (SUPERLATIVE.search(p["sql"]) or SUPERLATIVE.search(r["sql"])))):
        return "LIMIT / ties (MAX-subquery vs LIMIT)", d
    if dset == {"where"}:
        return "WHERE predicate only", d
    if "from" in dset:
        return "join path / tables", d
    return "multi-clause (" + "+".join(d) + ")", d


def preview_relation(rp, rr_):
    if rp["n"] == 0 and rr_["n"] == 0:
        return "both empty"
    if rr_["n"] == 0:
        return "right is empty"
    if rp["n"] == 0:
        return "pick is empty"
    if rp["n"] is None or rr_["n"] is None:
        return "unknown"
    fp, fr = rp["first"], rr_["first"]
    if rp["k"] == rr_["k"]:
        if fp == fr and rp["n"] != rr_["n"]:
            return "same first row, row count differs"
        if fp == fr:
            return "same first row and count (other rows differ)"
        if rp["n"] == rr_["n"] and isinstance(fp, tuple) and isinstance(fr, tuple) and sorted(map(str, fp)) == sorted(map(str, fr)):
            return "same values, column order"
        if (isinstance(fp, tuple) and isinstance(fr, tuple) and len(fp) == len(fr) == 1
                and all(isinstance(x, (int, float)) for x in fp + fr)):
            a, b = fp[0], fr[0]
            if a and b and (abs(a / b - 100) < 1e-6 or abs(b / a - 100) < 1e-6):
                return "scalar x100 (percent vs ratio)"
            if a and b and abs(round(a) - round(b)) == 0 and a != b:
                return "scalar rounding/type"
            return "different scalar"
        return "different rows"
    if isinstance(fp, tuple) and isinstance(fr, tuple):
        if rp["k"] > rr_["k"] and set(map(str, fr)) <= set(map(str, fp)):
            return "pick has extra columns (superset row)"
        if rp["k"] < rr_["k"] and set(map(str, fp)) <= set(map(str, fr)):
            return "pick misses columns (subset row)"
    return "different shape"


LIST_Q = re.compile(r"^\s*(please\s+)?(list|name|give|provide|show|state|identify|what are|which)\b", re.I)


def degenerate(res):
    if res["n"] == 0:
        return True
    f = res["first"]
    if isinstance(f, tuple) and all(v is None or v == 0 or v == 0.0 for v in f) and res["n"] == 1:
        return True
    return False


# ------------------------------------------------------------------------------------------------------ analysis
def analyse(ids, Q, out):
    n = len(ids)
    pk = orc = gr = grl = 0
    misses = []
    for i in ids:
        r = Q.get(i)
        if not r or not r["cands"]:
            continue
        C = r["cands"]
        L = live_of(C)
        p = max(L, key=lambda c: c["rr"])
        g = min(C, key=lambda c: c["rank"])
        gl = min(L, key=lambda c: c["rank"])
        pk += p["ok"]; orc += any(c["ok"] for c in C); gr += g["ok"]; grl += gl["ok"]
        if any(c["ok"] for c in C) and not p["ok"]:
            misses.append((r, C, L, p))
    out(f"DECISION n={n}  covered={sum(1 for i in ids if i in Q and Q[i]['cands'])}")
    out(f"baseline pick {100*pk/n:.2f}  oracle {100*orc/n:.2f}  gap {100*(orc-pk)/n:.2f} ({orc-pk} q)")
    out(f"greedy rank-0 (any) {100*gr/n:.2f}   greedy among live {100*grl/n:.2f}")

    recs = []
    for r, C, L, p in misses:
        rights = [c for c in C if c["ok"]]
        rl = [c for c in rights if c in L]
        best = max(rl or rights, key=lambda c: c["rr"])
        order = sorted(L, key=lambda c: -c["rr"])
        pos = next((k + 1 for k, c in enumerate(order) if c is best), None)
        sigc = Counter(c["sig"] for c in L)
        cls_best = {}
        for c in L:
            cls_best[c["sig"]] = max(cls_best.get(c["sig"], -1e9), c["rr"])
        cls_order = sorted(cls_best, key=lambda s: -cls_best[s])
        cpos = cls_order.index(best["sig"]) + 1 if best["sig"] in cls_best else None
        rp, rr_ = parse_res(p["res"]), parse_res(best["res"])
        fp, fr = feats(p["sql"]), feats(best["sql"])
        b, d = bucket(p, best, fp, fr, rp, rr_)
        rel = preview_relation(rp, rr_)
        if not rl:
            b = "right candidate not live (empty/NULL result)"
        recs.append({
            "id": r["id"], "db": r["db_id"], "q": r["question"], "ev": r["context"],
            "bucket": b, "clauses": d, "relation": rel,
            "right_rr_pos": pos, "right_class_pos": cpos, "n_classes": len(cls_best), "margin": round(p["rr"] - best["rr"], 3),
            "right_is_greedy": any(c["rank"] == min(x["rank"] for x in C) for c in rights),
            "greedy_ok": min(C, key=lambda c: c["rank"])["ok"],
            "pick_rank": p["rank"], "right_rank": best["rank"], "right_min_rank": min(c["rank"] for c in rights),
            "n_live": len(L), "pick_class": sigc[p["sig"]], "right_class": sigc[best["sig"]],
            "pick_cost": p["cost"], "right_cost": best["cost"],
            "right_degenerate": degenerate(rr_), "list_q": bool(LIST_Q.match(r["question"])),
            "pick_sql": p["sql"], "right_sql": best["sql"], "pick_res": p["res"], "right_res": best["res"],
            "pick_rr": p["rr"], "right_rr": best["rr"]})
    return recs


def describe(recs, out):
    N = len(recs)
    out(f"\n=== Part 1: {N} pick-misses (oracle right, pick wrong) ===")
    pos = Counter(min(r["right_rr_pos"] or 99, 6) for r in recs)
    out("rr position of best right (live) candidate: " + ", ".join(
        f"{'#'+str(k) if k < 6 else '#6+'}: {v}" for k, v in sorted(pos.items())))
    cp = Counter(min(r["right_class_pos"] or 99, 6) for r in recs)
    out("rank of the right RESULT CLASS among distinct live results (by best rr): " + ", ".join(
        f"{'#'+str(k) if k < 6 else '#6+/not live'}: {v}" for k, v in sorted(cp.items())) +
        f"; distinct live results per miss: median {sorted(r['n_classes'] for r in recs)[N//2]}")
    ms = sorted(r["margin"] for r in recs)
    q = lambda f: ms[min(int(f * N), N - 1)]
    out(f"rr margin pick - best right: median {q(.5):.2f}, IQR [{q(.25):.2f}, {q(.75):.2f}], "
        f"<0.5: {sum(m < .5 for m in ms)}, <1: {sum(m < 1 for m in ms)}, <2: {sum(m < 2 for m in ms)}")
    out(f"right candidate is the generator greedy (rank 0): {sum(r['right_is_greedy'] for r in recs)}/{N}; "
        f"best-right generator rank median {sorted(r['right_min_rank'] for r in recs)[N//2]}, "
        f"pick rank median {sorted(r['pick_rank'] for r in recs)[N//2]}")
    out(f"self-consistency: right class > pick class {sum(r['right_class'] > r['pick_class'] for r in recs)}, "
        f"equal {sum(r['right_class'] == r['pick_class'] for r in recs)}, "
        f"smaller {sum(r['right_class'] < r['pick_class'] for r in recs)}")
    out(f"right result degenerate (0 rows or single 0/NULL): {sum(r['right_degenerate'] for r in recs)}")
    out("per DB: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['db'] for r in recs).items())))
    out("\npreview relation (pick vs best right):")
    for k, v in Counter(r["relation"] for r in recs).most_common():
        out(f"  {v:3d}  {k}")
    out("\nbuckets (clause diff pick vs best right):")
    for k, v in Counter(r["bucket"] for r in recs).most_common():
        sub = [r for r in recs if r["bucket"] == k]
        out(f"  {v:3d}  {k}   [greedy-right {sum(r['right_is_greedy'] for r in sub)}, margin<1 "
            f"{sum(r['margin'] < 1 for r in sub)}, right-class>pick {sum(r['right_class'] > r['pick_class'] for r in sub)}]")
    if MANUAL.exists():
        lab = json.load(open(MANUAL))
        out("\nmanual theme x verdict (scripts/analysis/system1_pick_gap_manual.json; real = validator should prefer the "
            "right one, conv = gold convention decides, noise = right one is accidental/worse):")
        T = defaultdict(list)
        for r in recs:
            t, v = lab[r["id"].split("_")[-1]]
            r["theme"], r["verdict"] = t, v
            T[t].append(r)
        out(f"  {'theme':8s} {'n':>3s} {'real':>5s} {'conv':>5s} {'noise':>5s} {'greedyR':>8s} {'marg<1':>7s} "
            f"{'Rcls>P':>7s} {'Rpos2':>6s}")
        for t, rs in sorted(T.items(), key=lambda kv: -len(kv[1])):
            cv = Counter(r["verdict"] for r in rs)
            out(f"  {t:8s} {len(rs):3d} {cv['real']:5d} {cv['conv']:5d} {cv['noise']:5d} "
                f"{sum(r['right_is_greedy'] for r in rs):8d} {sum(r['margin'] < 1 for r in rs):7d} "
                f"{sum(r['right_class'] > r['pick_class'] for r in rs):7d} {sum(r['right_rr_pos'] == 2 for r in rs):6d}")
        cv = Counter(r["verdict"] for r in recs)
        out(f"  {'all':8s} {N:3d} {cv['real']:5d} {cv['conv']:5d} {cv['noise']:5d}")
        out("  per DB real/conv/noise: " + ", ".join(
            f"{d} {sum(r['verdict']=='real' for r in recs if r['db']==d)}/{sum(r['verdict']=='conv' for r in recs if r['db']==d)}/"
            f"{sum(r['verdict']=='noise' for r in recs if r['db']==d)}" for d in sorted({r['db'] for r in recs})))
    out("\nexamples per bucket:")
    for k, _ in Counter(r["bucket"] for r in recs).most_common():
        out(f"--- {k}")
        for r in [x for x in recs if x["bucket"] == k][:3]:
            out(f"  [{r['db']}] {r['q'][:120]}  (margin {r['margin']:.2f}, right pos {r['right_rr_pos']}, "
                f"classes pick {r['pick_class']}/right {r['right_class']})")
            out(f"    PICK  {r['pick_sql'][:260]}  -> {r['pick_res'][:70]}")
            out(f"    RIGHT {r['right_sql'][:260]}  -> {r['right_res'][:70]}")


# ------------------------------------------------------------------------------------------------------- rules
def prep(ids, Q):
    """Per question: live cands with derived features."""
    P = {}
    for i in ids:
        r = Q.get(i)
        if not r or not r["cands"]:
            continue
        L = live_of(r["cands"])
        sigc = Counter(c["sig"] for c in L)
        gmin = min(c["rank"] for c in L)
        lq = bool(LIST_Q.match(r["question"]))
        rows = []
        for c in L:
            res = parse_res(c["res"])
            rows.append({"rr": c["rr"], "ok": c["ok"], "cost": c["cost"], "rank": c["rank"], "sig": c["sig"],
                         "n_sig": sigc[c["sig"]], "greedy": c["rank"] == gmin, "nrows": res["n"],
                         "limit1": bool(re.search(r"LIMIT\s+1\b", c["sql"], re.I)),
                         "superl": bool(SUPERLATIVE.search(c["sql"])),
                         "cdist": "COUNT(DISTINCT" in c["sql"].upper(),
                         "ckey": re.sub(r"COUNT\(DISTINCT\s+", "COUNT(", c["sql"], flags=re.I)})
        keys = Counter(x["ckey"] for x in rows)
        for x in rows:
            x["cdist_twin"] = x["cdist"] and keys[x["ckey"]] > 1
        P[i] = {"db": r["db_id"], "cands": rows, "listq": lq}
    return P


def pick_ok(q, rule, th):
    C = q["cands"]
    if rule == "rr":
        return max(C, key=lambda c: c["rr"])["ok"]
    if rule == "greedy":
        return next(c for c in C if c["greedy"])["ok"]
    if rule == "cost":
        return max(C, key=lambda c: c["rr"] - th * c["cost"])["ok"]
    if rule == "rank":
        return max(C, key=lambda c: c["rr"] - th * c["rank"])["ok"]
    if rule == "sc_log":
        return max(C, key=lambda c: c["rr"] + th * math.log(c["n_sig"]))["ok"]
    if rule == "sc_frac":
        return max(C, key=lambda c: c["rr"] + th * c["n_sig"] / len(C))["ok"]
    if rule == "class_lse":  # class score = T * logsumexp(rr / T) over the class; T -> 0 = baseline
        if th <= 0:
            return pick_ok(q, "rr", 0)
        S = defaultdict(list)
        for c in C:
            S[c["sig"]].append(c["rr"] / th)
        sc = {s: th * (max(v) + math.log(sum(math.exp(x - max(v)) for x in v))) for s, v in S.items()}
        top = max(sc, key=sc.get)
        return max((c for c in C if c["sig"] == top), key=lambda c: c["rr"])["ok"]
    if rule == "class_mean":  # class score = mean rr + th * log size
        S = defaultdict(list)
        for c in C:
            S[c["sig"]].append(c["rr"])
        sc = {s: max(v) + th * math.log(len(v)) for s, v in S.items()}
        top = max(sc, key=sc.get)
        return max((c for c in C if c["sig"] == top), key=lambda c: c["rr"])["ok"]
    if rule == "greedy_margin":
        b = max(C, key=lambda c: c["rr"])
        g = next(c for c in C if c["greedy"])
        return (g if b["rr"] - g["rr"] < th else b)["ok"]
    if rule == "empty_pen":
        return max(C, key=lambda c: c["rr"] - th * (c["nrows"] == 0))["ok"]
    if rule == "list_onerow_pen":
        return max(C, key=lambda c: c["rr"] - th * (q["listq"] and c["nrows"] == 1))["ok"]
    if rule == "limit1_pen":  # penalise LIMIT 1 when a superlative (= MAX/MIN subquery) twin exists
        has_sup = any(c["superl"] and not c["limit1"] for c in C)
        return max(C, key=lambda c: c["rr"] - th * (c["limit1"] and has_sup))["ok"]
    if rule == "cdist_twin_bonus":  # prefer COUNT(DISTINCT ...) over its plain-COUNT twin
        return max(C, key=lambda c: c["rr"] + th * c["cdist_twin"])["ok"]
    if rule == "cdist_bonus":
        return max(C, key=lambda c: c["rr"] + th * c["cdist"])["ok"]
    if rule == "cost+sc":
        lam, mu = th
        return max(C, key=lambda c: c["rr"] - lam * c["cost"] + mu * math.log(c["n_sig"]))["ok"]
    raise ValueError(rule)


GRIDS = {
    "cost": [0, .05, .1, .2, .3, .5, .75, 1, 1.5, 2, 3],
    "rank": [0, .02, .05, .1, .2, .3, .5],
    "sc_log": [0, .1, .25, .5, .75, 1, 1.5, 2, 3, 4],
    "sc_frac": [0, .5, 1, 2, 3, 5, 8],
    "class_lse": [0, .1, .25, .5, 1, 2, 4],
    "class_mean": [0, .1, .25, .5, 1, 1.5, 2],
    "greedy_margin": [0, .1, .25, .5, 1, 1.5, 2, 3],
    "empty_pen": [0, .5, 1, 2, 4, 8],
    "list_onerow_pen": [0, .5, 1, 2, 4],
    "limit1_pen": [0, .25, .5, 1, 2, 4],
    "cdist_twin_bonus": [0, .25, .5, 1, 2, 4],
    "cdist_bonus": [0, .25, .5, 1, 2, 4],
    "cost+sc": [(a, b) for a in [0, .1, .2, .3, .5, 1] for b in [0, .25, .5, 1, 1.5, 2]],
}


def dist0(th):
    return sum(abs(x) for x in th) if isinstance(th, tuple) else abs(th)


def lodo(ids, P, rule, grid, db):
    dbs = sorted(set(db.values()))
    res, chosen = {}, {}
    for h in dbs:
        tr = [i for i in ids if db[i] != h and i in P]
        best = max(grid, key=lambda th: (sum(pick_ok(P[i], rule, th) for i in tr), -dist0(th)))
        chosen[h] = best
        for i in ids:
            if db[i] == h:
                res[i] = bool(i in P and pick_ok(P[i], rule, best))
    return res, chosen


def rules(ids, Q, out):
    P = prep(ids, Q)
    db = {i: (Q[i]["db_id"] if i in Q else "?") for i in ids}
    # questions absent from the union file: DB from decision.json counts not available; use universe-free fallback
    n = len(ids)
    base = {i: bool(i in P and pick_ok(P[i], "rr", 0)) for i in ids}
    out(f"\n=== Part 2: deterministic pick rules (DECISION n={n}; baseline {100*sum(base.values())/n:.2f}) ===")
    out("rule                 in-sample best (param)        LODO pick   delta   DB sign-flip p  McNemar(+/-)  LODO params")
    g = {i: bool(i in P and pick_ok(P[i], "greedy", 0)) for i in ids}
    c = cluster_signflip(base, g, db); qq = question_signflip(base, g)
    out(f"{'greedy (rank 0 live)':20s} {'-':29s} {100*sum(g.values())/n:8.2f}  {c['delta']:+6.2f}  {c['p']:13.3f}  "
        f"+{qq['wins']}/-{qq['losses']}")
    summary = {"greedy": (100 * sum(g.values()) / n, c["delta"], c["p"])}
    for rule, grid in GRIDS.items():
        ins = {th: sum(bool(i in P and pick_ok(P[i], rule, th)) for i in ids) for th in grid}
        bt = max(grid, key=lambda th: (ins[th], -dist0(th)))
        res, ch = lodo(ids, P, rule, grid, db)
        c = cluster_signflip(base, res, db); qq = question_signflip(base, res)
        out(f"{rule:20s} {100*ins[bt]/n:6.2f} ({str(bt):20s})  {100*sum(res.values())/n:8.2f}  {c['delta']:+6.2f}  "
            f"{c['p']:13.3f}  +{qq['wins']}/-{qq['losses']}   {sorted(Counter(map(str, ch.values())).items())}")
        summary[rule] = (100 * sum(res.values()) / n, c["delta"], c["p"])
    # fixed, un-fitted reference points
    out("\nfixed-parameter curves (in-sample, no fitting; for shape only):")
    for rule in ("cost", "sc_log", "class_lse", "greedy_margin"):
        out(f"  {rule:14s} " + "  ".join(f"{th}:{100*sum(bool(i in P and pick_ok(P[i], rule, th)) for i in ids)/n:.2f}"
                                         for th in GRIDS[rule]))
    return summary


def twin_stats(ids, Q, out):
    """Convention check across ALL covered DECISION questions: when the live pool holds two candidates that differ only
    by COUNT(DISTINCT x) vs COUNT(x) (or only by a trailing LIMIT 1 after an = (SELECT MAX/MIN) filter) and give
    different results, which one matches gold, and which does the validator prefer?"""
    out("\n=== twin conventions (all covered DECISION questions; pairs with different results) ===")
    for name, key in (("COUNT(DISTINCT) vs COUNT", lambda s: re.sub(r"COUNT\(DISTINCT\s+", "COUNT(", s, flags=re.I)),
                      ("superlative + LIMIT 1 vs no LIMIT", lambda s: re.sub(r"\s+LIMIT\s+1\s*$", "", s, flags=re.I)
                       if SUPERLATIVE.search(s) else s)):
        a_ok = b_ok = neither = vr_a = n = one = vr_right = 0
        for i in ids:
            r = Q.get(i)
            if not r or not r["cands"]:
                continue
            L = live_of(r["cands"])
            groups = defaultdict(list)
            for c in L:
                groups[key(c["sql"])].append(c)
            for k, g in groups.items():
                A = [c for c in g if c["sql"] != k]  # the variant (DISTINCT / with LIMIT)
                B = [c for c in g if c["sql"] == k]  # the plain form
                if not A or not B or A[0]["sig"] == B[0]["sig"]:
                    continue
                a, b = max(A, key=lambda c: c["rr"]), max(B, key=lambda c: c["rr"])
                n += 1; a_ok += a["ok"]; b_ok += b["ok"]; neither += not (a["ok"] or b["ok"]); vr_a += a["rr"] > b["rr"]
                if a["ok"] != b["ok"]:
                    one += 1; vr_right += (a["rr"] > b["rr"]) == bool(a["ok"])
                break
        out(f"  {name}: {n} questions; variant right {a_ok}, plain right {b_ok}, neither {neither}; "
            f"validator prefers the variant in {vr_a}; exactly one right in {one}, validator on the right side in {vr_right}")


def top2_budget(ids, Q, out):
    """Sizing a margin-gated top-2 comparison inside System 1 (same validator, e.g. a pairwise head): for questions
    whose top-2 distinct live results are within tau in rr, how many have (top1 right, top2 wrong) = risk and
    (top1 wrong, top2 right) = gain. Break-even pairwise accuracy = risk / (risk + gain)."""
    n = len(ids)
    out("\n=== top-2 distinct-result budget (margin-gated comparator sizing) ===")
    t2 = 0
    rows = []
    for i in ids:
        r = Q.get(i)
        if not r or not r["cands"]:
            continue
        L = live_of(r["cands"])
        best = {}
        for c in L:
            if c["sig"] not in best or c["rr"] > best[c["sig"]]["rr"]:
                best[c["sig"]] = c
        order = sorted(best.values(), key=lambda c: -c["rr"])
        a = order[0]
        b = order[1] if len(order) > 1 else None
        t2 += a["ok"] or bool(b and b["ok"])
        if b:
            rows.append((a["rr"] - b["rr"], a["ok"], b["ok"]))
    out(f"top-2 distinct-result oracle {100*t2/n:.2f} (pick 57.29)")
    for tau in (0.25, 0.5, 1, 2, 1e9):
        sub = [x for x in rows if x[0] < tau]
        risk = sum(1 for _, a, b in sub if a and not b)
        gain = sum(1 for _, a, b in sub if b and not a)
        out(f"  margin < {tau if tau < 1e8 else 'inf':>4}: {len(sub):3d} q gated; top1-only right {risk}, top2-only right "
            f"{gain}, both wrong {sum(1 for _, a, b in sub if not a and not b)}; max gain {100*gain/n:+.2f} pts; "
            f"break-even pairwise acc {risk/(risk+gain) if risk+gain else float('nan'):.2f}")


def main():
    ids, Q = load()
    # DB for uncovered questions: from the union file if present, else unknown -> they are wrong for every rule anyway,
    # but the sign-flip needs a DB label; take it from decision.json database_counts by elimination is impossible, so
    # read db_id from the BIRD train json listed as the DECISION source.
    dec = json.load(open(DECISION))
    miss = [i for i in ids if i not in Q]
    if miss:
        train = json.load(open(ROOT / dec["sources"]["train_json"]["path"]))
        dbmap = {f"bird_train_train_{k}": t["db_id"] for k, t in enumerate(train)}
        for i in miss:
            Q[i] = {"id": i, "db_id": dbmap.get(i, "?"), "cands": [], "question": ""}
        got = Counter(Q[i]["db_id"] for i in ids)
        assert got == Counter(dec["database_counts"]), (got, dec["database_counts"])
    lines = []
    out = lambda s: (print(s), lines.append(s))
    recs = analyse(ids, Q, out)
    describe(recs, out)
    twin_stats(ids, Q, out)
    top2_budget(ids, Q, out)
    rules(ids, Q, out)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUT_JSONL}")


if __name__ == "__main__":
    main()
