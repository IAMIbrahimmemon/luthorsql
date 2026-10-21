"""Why did Phase 3 (v4desc: grammar v4 + BIRD column descriptions + constant-arithmetic upweight) lose to R3a on
DECISION (52.51 vs 53.77), and on car_retails above all (52 -> 41)? Mac-only, CPU, no model calls.

  uv run --offline python scripts/analysis/v4desc_regression.py          # all sections, writes runs/analysis/v4desc_regression/
  uv run --offline python scripts/analysis/v4desc_regression.py --no-exec  # skip the SQLite re-execution proxy (section D)

Reads only DECISION ids (data/sys1/heldout_v2/decision.json via phase1_eval.IDS). Never reads BIRD dev outputs, RESERVED,
sealed sets or GATE lines (training-log lines containing "GATE" are dropped before parsing; the logs' mixed
DECISION+RESERVED "eval step" lines are not parsed either).

Sections
  A  per-DB: coverage, hits (own k=8 + own validator), oracle@8, greedy, pick failures, lost/won; v3 retrain spread
  B  distinct-decoding oracle@k (k=32 --distinct lists): v4desc vs R1's own k=32 (p1w union, src r1|both)
  C  join edges outside the FK graph (declared + inferred) in candidates / greedy / picks, per arm
  D  join-guard proxy: re-route v4desc candidates' top-level joins (a) over the v3 FK route, (b) keep a non-FK hint pair
     only when both columns have the same name; re-execute against ORIGINAL gold (BIRD EX set equality); validator
     scores kept as they were (approximation)
  E  component match vs gold (greedy and any@8): tables, projection columns/order/count, ops, WHERE columns/values
  F  schema text with/without descriptions per DB; serving/validator budgets; are wrong columns described ones?
  G  v3 vs v4 gold plans on DECISION (coverage, plan dict, compiled SQL)
  H  training mix, epochs reached, OOM skips (train_r1.log, GATE lines dropped)
  I  car_retails + other losing DBs: per-question lost/won with automatic labels (jsonl)
  J  car_retails description audit
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import re
import sqlite3
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("LUTHOR_GRAMMAR", "v3")  # FK routing as v3 compiles it (section D); schema_text is grammar-free
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_eval import IDS, db_of, load, single_hits  # noqa: E402

from luthor.sq import bird_desc  # noqa: E402
from luthor.sq.plan import join_tree  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sq2.serialize import SERVE_MAX_CHARS, schema_text  # noqa: E402
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402

OUT = ROOT / "runs/analysis/v4desc_regression"
CTL = ROOT / "runs/phase1/seed/r1/rr/dev_scored.jsonl"
ARM = ROOT / "runs/phase1/v4desc/r1/rr/dev_scored.jsonl"
WIDE4 = ROOT / "runs/phase1/v4desc/r1/rr/wide/cands_dev_k32.jsonl"
WIDE1 = ROOT / "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl"
V3_RUNS = {"R1(seed0)": "runs/r1r2/r1/rr/dev_scored.jsonl", "R3a": "runs/phase1/seed/r1/rr/dev_scored.jsonl",
           "R3b seedfull": "runs/phase1/seedfull/r1/rr/dev_scored.jsonl", "teacher": "runs/phase1/teacher/r1/rr/dev_scored.jsonl",
           "verified": "runs/phase1/verified/r1/rr/dev_scored.jsonl"}
VALIDATOR_BUDGET = 9000  # rerank_sq2 --budget-chars (both runs)
TAB = re.compile(r'(?:FROM|JOIN)\s+"([^"]+)"\s+AS\s+(\w+)')
EQ = re.compile(r'(\w+)\."([^"]+)"\s*=\s*(\w+)\."([^"]+)"')
ON = re.compile(r'\bON\b(.*?)(?=\bINNER JOIN\b|\bLEFT JOIN\b|\bJOIN\b|\bWHERE\b|\bGROUP BY\b|\bORDER BY\b|\)|$)', re.S)

DB = db_of()
DBS = sorted(set(DB.values()))
CARDS = {d: SchemaCard.load(ROOT / f"data/sys1/bird_train/cards/{d}.json") for d in DBS}  # descriptions off here
FK = {d: {frozenset([(a, b), (x, y)]) for a, b, x, y in list(c.fks) + list(c.inferred_fks)} for d, c in CARDS.items()}
LOG: list[str] = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def pick_idx(cands):
    live = [j for j, c in enumerate(cands) if c.get("runs") and c.get("nonnull")] or list(range(len(cands)))
    return max(live, key=lambda j: cands[j].get("rr", -1e9))


def greedy(S):
    return {i: bool(S.get(i)) and bool(S[i]["cands"]) and bool(S[i]["cands"][0].get("ok")) for i in IDS}


def join_pairs(sql):
    if not sql:
        return []
    al = {a: t for t, a in TAB.findall(sql)}
    return [((al[a], ca), (al[b], cb)) for m in ON.finditer(sql) for a, ca, b, cb in EQ.findall(m.group(1))
            if a in al and b in al]


def nonfk(d, sql):
    return [p for p in join_pairs(sql) if frozenset(p) not in FK[d]]


def by_db(h, d):
    return sum(h[i] for i in IDS if DB[i] == d)


# ---------------------------------------------------------------------------------------------------------- A
def section_a(C, N):
    say("\n## A. per-DB (DECISION, own k=8 list + own validator)")
    hc, oc = single_hits(C)
    hn, on = single_hits(N)
    gc, gn = greedy(C), greedy(N)
    say("db            n  cov R3a/v4d | hits R3a v4d | oracle@8 R3a v4d | greedy R3a v4d | pickfail R3a v4d | lost won")
    for d in DBS + ["ALL"]:
        ids = [i for i in IDS if d == "ALL" or DB[i] == d]
        s = lambda h: sum(h[i] for i in ids)
        pf = lambda h, o: sum(o[i] and not h[i] for i in ids)
        say(f"{d:13s} {len(ids):3d}  {sum(i in C for i in ids):3d}/{sum(i in N for i in ids):3d} | {s(hc):4d} {s(hn):4d} | "
            f"{s(oc):4d} {s(on):4d} | {s(gc):4d} {s(gn):4d} | {pf(hc, oc):3d} {pf(hn, on):3d} | "
            f"{sum(hc[i] and not hn[i] for i in ids):3d} {sum(hn[i] and not hc[i] for i in ids):3d}")
    say("\nv3 retrain spread (same recipe family, grammar v3, no descriptions): hits (oracle@8) per DB")
    H = {}
    for k, p in list(V3_RUNS.items()) + [("v4desc", str(ARM.relative_to(ROOT)))]:
        H[k] = single_hits(load(ROOT / p))
    say("%-13s" % "run" + "".join("%15s" % d[:13] for d in DBS) + "   total%")
    for k, (h, o) in H.items():
        say("%-13s" % k + "".join("%9d (%3d)" % (by_db(h, d), by_db(o, d)) for d in DBS) + "  %6.2f" % (100 * sum(h.values()) / len(IDS)))
    v3 = [k for k in H if k != "v4desc"]
    say("%-13s" % "v3 mean" + "".join("%15.1f" % st.mean(by_db(H[k][0], d) for k in v3) for d in DBS) +
        "  %6.2f" % st.mean(100 * sum(H[k][0].values()) / len(IDS) for k in v3))
    say("%-13s" % "v3 min-max" + "".join("%15s" % f"{min(by_db(H[k][0], d) for k in v3)}-{max(by_db(H[k][0], d) for k in v3)}" for d in DBS))
    return hc, oc, hn, on


# ---------------------------------------------------------------------------------------------------------- B
def section_b():
    say("\n## B. distinct decoding oracle@k (k=32 --distinct lists; R1 = its own candidates in the p1w union)")
    W4, W1 = load(WIDE4), load(WIDE1)
    o = lambda r, k, src=None: any(c.get("ok") for c in r["cands"] if c["rank"] < k and (not src or c.get("src") in src))
    for d in DBS + ["ALL"]:
        ids = [i for i in IDS if d == "ALL" or DB[i] == d]
        say(f"{d:13s} " + "  ".join(f"k{k}: R1 {sum(o(W1[i], k, {'r1', 'both'}) for i in ids if i in W1):3d} "
                                     f"v4d {sum(o(W4[i], k) for i in ids if i in W4):3d}" for k in (1, 4, 8, 16, 32)))
    C8, N8 = load(ROOT / "runs/phase1/seed/r1/rr/cands_dev.jsonl"), load(ROOT / "runs/phase1/v4desc/r1/rr/cands_dev.jsonl")
    nu = lambda r: len({c["sql"] for c in r["cands"][:8]})
    both = [i for i in IDS if i in C8 and i in N8]
    say(f"unique SQL per plain k=8 list (covered by both): R3a {st.mean(nu(C8[i]) for i in both):.2f}, "
        f"v4desc {st.mean(nu(N8[i]) for i in both):.2f} (the --distinct lists have 8 by construction)")


# ---------------------------------------------------------------------------------------------------------- C
def section_c(C, N):
    say("\n## C. join edges not in the FK graph (declared + inferred), DECISION")
    say("db            gold | R3a cands% greedy pick(wrong/right) | v4d cands% greedy pick(wrong/right)")
    tot = collections.Counter()
    for d in DBS + ["ALL"]:
        a = collections.Counter()
        for i in IDS:
            if d != "ALL" and DB[i] != d:
                continue
            r0 = C.get(i) or N.get(i)
            if r0:
                a["gold"] += bool(nonfk(DB[i], r0["gold_sql"]))
            for nm, S in (("C", C), ("N", N)):
                r = S.get(i)
                if not r:
                    continue
                cs = r["cands"]
                a[nm + "c"] += len(cs)
                a[nm + "b"] += sum(bool(nonfk(DB[i], c.get("sql"))) for c in cs)
                a[nm + "g"] += bool(nonfk(DB[i], cs[0].get("sql")))
                p = cs[pick_idx(cs)]
                bad = bool(nonfk(DB[i], p.get("sql")))
                a[nm + "pw"] += bad and not p.get("ok")
                a[nm + "pr"] += bad and bool(p.get("ok"))
        say(f"{d:13s} {a['gold']:4d} | {100 * a['Cb'] / max(1, a['Cc']):5.1f}% {a['Cg']:4d} {a['Cpw']:3d}/{a['Cpr']:<3d} | "
            f"{100 * a['Nb'] / max(1, a['Nc']):5.1f}% {a['Ng']:4d} {a['Npw']:3d}/{a['Npr']:<3d}")
    pairs = collections.Counter()
    oks = collections.Counter()
    for i in IDS:
        r = N.get(i)
        if not r:
            continue
        for c in r["cands"]:
            for p in {tuple(sorted(x)) for x in nonfk(DB[i], c.get("sql"))}:
                pairs[(DB[i], p)] += 1
                oks[(DB[i], p)] += bool(c.get("ok"))
    say("most frequent non-FK pairs in v4desc candidates (count, of which in a right candidate):")
    for k, v in pairs.most_common(15):
        same = k[1][0][1].lower() == k[1][1][1].lower()
        say(f"  {v:3d} {oks[k]:3d}  {k[0]}: {k[1][0][0]}.{k[1][0][1]} = {k[1][1][0]}.{k[1][1][1]}{'  (same name)' if same else ''}")


# ---------------------------------------------------------------------------------------------------------- D
def run_sql(d, sql, sec=30.0):
    con = sqlite3.connect(f"file:{ROOT}/data/sys1/dbs/{d}/{d}.sqlite?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    t0 = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - t0 > sec else 0, 10000)
    try:
        return set(map(tuple, con.execute(sql).fetchmany(100000)))
    except Exception:
        return None
    finally:
        con.close()


def reroute(d, sql, keep_same_name: bool):
    """The candidate's top-level FROM/JOIN block re-routed: FK graph (declared + inferred) plus, with keep_same_name,
    the candidate's own non-FK pairs whose two columns share a name (cheap edges, as v4's hint pairs). None when the
    candidate has no offending pair, has a subquery / self-join, or cannot be routed."""
    if not sql or sql.count("SELECT") != 1:
        return None
    m = re.search(r"\bFROM\b(.*?)(?=\bWHERE\b|\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|$)", sql, re.S)
    blk = m.group(1)
    tabs = TAB.findall("FROM" + blk)
    if len(tabs) < 2 or len({t for t, _ in tabs}) != len(tabs):
        return None
    al = {a: t for t, a in tabs}
    pairs = [((al[a], ca), (al[b], cb)) for a, ca, b, cb in EQ.findall(blk) if a in al and b in al]
    bad = [p for p in pairs if frozenset(p) not in FK[d]]
    same = lambda p: p[0][1].lower() == p[1][1].lower()
    offending = [p for p in bad if not (keep_same_name and same(p))]
    if not offending:
        return None
    g = CARDS[d].graph()
    if keep_same_name:
        for (ta, ca), (tb, cb) in bad:
            if same(((ta, ca), (tb, cb))) and ta != tb:
                g.adj[ta].append((tb, ca, cb, 0.1))
                g.adj[tb].append((ta, cb, ca, 0.1))
    try:
        order, edges = join_tree(g, [t for t, _ in tabs], None, None)
    except Exception:
        return None
    ta = {t: a for t, a in tabs}
    k = len(tabs)
    for t in order:
        if t not in ta:
            k += 1
            ta[t] = f"T{k}x"
    new = f' "{order[0]}" AS {ta[order[0]]}'
    for a, ca, b, cb in edges:
        new += f' INNER JOIN "{b}" AS {ta[b]} ON {ta[a]}."{ca}" = {ta[b]}."{cb}"'
    return sql[:m.start(1)] + new + " " + sql[m.end(1):]


def section_d(C, N, hc, oc, do_exec: bool):
    say("\n## D. join-guard proxy (re-execution, ORIGINAL gold; validator scores unchanged)")
    cache = OUT / "joinfix_proxy.json"
    if do_exec:
        res, agree, golds = {}, collections.Counter(), {}
        for mode in ("v3route", "same_name"):
            res[mode] = {}
            for i in IDS:
                r = N.get(i)
                if not r:
                    continue
                d = DB[i]
                fx = [reroute(d, c.get("sql"), mode == "same_name") for c in r["cands"]]
                if not any(fx):
                    continue
                if i not in golds:
                    golds[i] = run_sql(d, r["gold_sql"])
                g = golds[i]
                rec = []
                for c, f in zip(r["cands"], fx):
                    if f is None:
                        rec.append(None)
                        continue
                    if mode == "v3route":
                        o = run_sql(d, c["sql"])
                        agree[str((o is not None and o == g, bool(c.get("ok"))))] += 1
                    out = run_sql(d, f)
                    rec.append({"sql": f, "ok": bool(out is not None and g is not None and out == g), "runs": out is not None,
                                "nonnull": bool(out) and any(any(v is not None for v in row) for row in out)})
                res[mode][i] = rec
        res["ex_agreement_original_sql"] = dict(agree)
        cache.write_text(json.dumps(res))
    res = json.loads(cache.read_text())
    say(f"my EX vs stored ok flag on the original candidates (mine, stored): {res['ex_agreement_original_sql']}")
    gc = greedy(C)
    hn, on = single_hits(N)
    gn = greedy(N)
    for mode in ("v3route", "same_name"):
        F = copy.deepcopy(N)
        flips = collections.Counter()
        for i, rec in res[mode].items():
            for c, x in zip(F[i]["cands"], rec):
                if x:
                    flips["to_right" if x["ok"] and not c.get("ok") else ("to_wrong" if c.get("ok") and not x["ok"] else "same")] += 1
                    c.update(x)
        hf, of = single_hits(F)
        gf = greedy(F)
        say(f"-- mode {mode}: {sum(1 for v in res[mode].values() for x in v if x)} candidates re-routed {dict(flips)}")
        say("db            greedy R3a v4d fix | hits R3a v4d fix | oracle@8 R3a v4d fix")
        for d in DBS + ["ALL"]:
            ids = [i for i in IDS if d == "ALL" or DB[i] == d]
            s = lambda h: sum(h[i] for i in ids)
            say(f"{d:13s} {s(gc):4d} {s(gn):4d} {s(gf):4d} | {s(hc):4d} {s(hn):4d} {s(hf):4d} | {s(oc):4d} {s(on):4d} {s(of):4d}")
        c = cluster_signflip(hc, hf, DB)
        q = question_signflip(hc, hf)
        say(f"   proxy DECISION {100 * sum(hf.values()) / len(IDS):.2f} vs R3a {100 * sum(hc.values()) / len(IDS):.2f}: "
            f"delta {c['delta']:+.2f} (DB p={c['p']:.2f}, McNemar +{q['wins']}/-{q['losses']}); greedy "
            f"{100 * sum(gf.values()) / len(IDS):.2f} vs v4desc {100 * sum(gn.values()) / len(IDS):.2f}")


# ---------------------------------------------------------------------------------------------------------- E
def comp(sql):
    import sqlglot
    from sqlglot import exp
    try:
        t = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    sel = t.find(exp.Select)
    if sel is None:
        return None
    proj, ops = [], set()
    for e in sel.expressions:
        proj.append(tuple(sorted(c.name.lower() for c in e.find_all(exp.Column))) or ("*",))
        for k, o in ((exp.Sub, "-"), (exp.Add, "+"), (exp.Mul, "*"), (exp.Div, "/")):
            if e.find(k):
                ops.add(o)
    w = t.find(exp.Where)
    return {"tables": frozenset(x.name.lower() for x in t.find_all(exp.Table)), "proj_cols": frozenset(proj),
            "proj_order": tuple(proj), "n_proj": len(proj), "ops": frozenset(ops),
            "where_cols": frozenset(c.name.lower() for c in w.find_all(exp.Column)) if w else frozenset(),
            "where_vals": frozenset(str(x.this).lower() for x in w.find_all(exp.Literal)) if w else frozenset()}


KEYS = ["tables", "proj_cols", "proj_order", "n_proj", "ops", "where_cols", "where_vals"]


def section_e(C, N):
    say("\n## E. component match vs gold, questions covered by both (greedy | any of the k=8 list)")
    both = [i for i in IDS if i in C and i in N]
    for name, ids in (("car_retails", [i for i in both if DB[i] == "car_retails"]),
                      ("other 7 DBs", [i for i in both if DB[i] != "car_retails"])):
        for nm, S in (("R3a", C), ("v4d", N)):
            g0, ak, n = collections.Counter(), collections.Counter(), 0
            for i in ids:
                G = comp(S[i]["gold_sql"])
                if not G:
                    continue
                n += 1
                cs = [comp(c["sql"]) if c.get("sql") else None for c in S[i]["cands"]]
                for k in KEYS:
                    g0[k] += bool(cs[0]) and cs[0][k] == G[k]
                    ak[k] += any(c and c[k] == G[k] for c in cs)
            say(f"  {name:12s} {nm} n={n}  greedy: " + " ".join(f"{k}={g0[k]}" for k in KEYS) +
                "  | any@8: " + " ".join(f"{k}={ak[k]}" for k in KEYS))


# ---------------------------------------------------------------------------------------------------------- F
def section_f(C, N):
    import sqlglot
    from sqlglot import exp
    say("\n## F. schema text with / without BIRD descriptions (sq2 serialize.schema_text)")
    D = bird_desc.load_dir(bird_desc.DEFAULT_DIR)
    described = {}
    say(f"db            chars no-desc  desc   +%   described/cols  > validator budget {VALIDATOR_BUDGET} (no/yes)  > serve {SERVE_MAX_CHARS}")
    for d in DBS:
        c = SchemaCard.load(ROOT / f"data/sys1/bird_train/cards/{d}.json")
        l0 = len(schema_text(c).text)
        n = bird_desc.apply(c, D)
        l1 = len(schema_text(c).text)
        described[d] = {(t.name.lower(), x.name.lower()) for t in c.tables for x in t.columns if x.description}
        say(f"{d:13s} {l0:13d} {l1:6d} {100 * (l1 - l0) / l0:5.1f}%  {n:3d}/{sum(len(t.columns) for t in c.tables):<3d}"
            f"            {str(l0 > VALIDATOR_BUDGET):5s}/{str(l1 > VALIDATOR_BUDGET):5s}                 {l1 > SERVE_MAX_CHARS}")
    say("(generator: whole schema whenever <= 60,000 chars -> no table is ever trimmed on these DBs; validator "
        "tables_for keeps every table a candidate uses, then fills to 9,000 chars)")

    def cols(sql):
        try:
            t = sqlglot.parse_one(sql, read="sqlite")
        except Exception:
            return set()
        al = {(x.alias or x.name).lower(): x.name.lower() for x in t.find_all(exp.Table)}
        tabs = set(al.values())
        out = set()
        for c in t.find_all(exp.Column):
            tb = al.get(c.table.lower()) if c.table else (next(iter(tabs)) if len(tabs) == 1 else None)
            if tb:
                out.add((tb, c.name.lower()))
        return out
    say("greedy-candidate columns vs gold columns: share that are described (if descriptions pulled the model to wrong "
        "columns, v4d's intruders would be described more often than its base rate)")
    for scope in ("car_retails", "other"):
        for nm, S in (("R3a", C), ("v4d", N)):
            k = collections.Counter()
            for i in IDS:
                if (DB[i] == "car_retails") != (scope == "car_retails") or i not in C or i not in N:
                    continue
                g, p, dd = cols(S[i]["gold_sql"]), cols(S[i]["cands"][0]["sql"]), described[DB[i]]
                for x in p - g:
                    k["i"] += 1
                    k["id"] += x in dd
                for x in g - p:
                    k["m"] += 1
                    k["md"] += x in dd
                for x in p:
                    k["u"] += 1
                    k["ud"] += x in dd
            say(f"  {scope:11s} {nm}: wrong (intruder) cols {k['i']:3d} ({100 * k['id'] / max(1, k['i']):.0f}% described), "
                f"missed gold cols {k['m']:3d} ({100 * k['md'] / max(1, k['m']):.0f}%), base rate {100 * k['ud'] / max(1, k['u']):.0f}%")


# ---------------------------------------------------------------------------------------------------------- G
def section_g():
    say("\n## G. gold plans v3 (data/sys1/bird_train) vs v4 (data/plans_v4/train/bird_train), DECISION ids only")
    ids = set(IDS)

    def rd(p):
        return {r["id"]: r for r in map(json.loads, open(p)) if r.get("id") in ids}
    v3, v4 = rd(ROOT / "data/sys1/bird_train/plans.jsonl"), rd(ROOT / "data/plans_v4/train/bird_train/plans.jsonl")
    empty = (None, [], "none", 0, False)
    say("db            v3cov v4cov both plan_same sql_same  differing plan keys")
    for d in DBS:
        a, keys = collections.Counter(), collections.Counter()
        for i in IDS:
            if DB[i] != d:
                continue
            p3, p4 = v3.get(i), v4.get(i)
            c3, c4 = bool(p3 and p3["outcome"] == "plan"), bool(p4 and p4["outcome"] == "plan")
            a["v3"] += c3
            a["v4"] += c4
            if c3 and c4:
                a["both"] += 1
                a["sql"] += p3["compiled_sql"] == p4["compiled_sql"]
                P3, P4 = p3["plan"], p4["plan"]
                ks = [k for k in set(P3) | set(P4) if P3.get(k) != P4.get(k) and not (P3.get(k) in empty and P4.get(k) in empty)]
                a["plan"] += not ks
                keys.update(ks)
        say(f"{d:13s} {a['v3']:5d} {a['v4']:5d} {a['both']:4d} {a['plan']:9d} {a['sql']:8d}  {dict(keys)}")


# ---------------------------------------------------------------------------------------------------------- H
def section_h():
    say("\n## H. training run facts (train_r1.log, lines containing GATE dropped; dev 'eval step' lines not parsed)")
    for run in ("seed", "v4desc"):
        p = ROOT / f"runs/phase1/{run}/logs/train_r1.log"
        rows = mix = oom = last = None
        with open(p, errors="replace") as f:
            for line in f:
                if "GATE" in line or line.startswith("eval step"):
                    continue
                if line.startswith("training rows by source"):
                    mix = line.strip()
                elif re.match(r"^train \d+ / \d+ dbs", line):
                    rows = int(line.split()[1])
                elif line.startswith("OOM:"):
                    oom = line.strip()
                elif re.match(r"^step \d+/\d+ ep \d+", line):
                    last = line
        m = re.match(r"^step (\d+)/(\d+) ep (\d+).*?oom_skipped \d+/(\d+)q", last)
        step, total, ep, seen = (int(x) for x in m.groups())
        say(f"{run}: {mix}")
        say(f"   train rows {rows}; last logged step {step}/{total} in epoch {ep}; questions drawn {seen} "
            f"-> {seen / rows:.2f} passes at that step, ~{seen / rows * total / step:.2f} at the end")
        say(f"   {oom}")


# ---------------------------------------------------------------------------------------------------------- I
def labels(d, gold, cand):
    G, P = comp(gold), comp(cand)
    out = []
    if nonfk(d, cand) and not nonfk(d, gold):
        out.append("join_pair")
    if G and P:
        for k in ("tables", "n_proj", "proj_cols", "ops", "where_cols", "where_vals"):
            if G[k] != P[k]:
                out.append(k)
    return out


def section_i(C, N, hc, oc, hn, on):
    say("\n## I. per-question lost / won (written to runs/analysis/v4desc_regression/diff_<db>.jsonl)")
    losing = ["car_retails", "student_loan", "law_episode", "university", "cs_semester"]
    summary = {}
    for d in losing:
        rows, cnt = [], collections.Counter()
        for i in IDS:
            if DB[i] != d or hc[i] == hn[i]:
                continue
            r0 = C.get(i) or N.get(i)
            rec = {"id": i, "lost" if hc[i] else "won": True, "question": r0["question"], "evidence": r0.get("context"),
                   "gold": r0["gold_sql"], "oracle8_R3a": oc[i], "oracle8_v4d": on[i]}
            for nm, S in (("R3a", C), ("v4d", N)):
                r = S.get(i)
                if not r:
                    rec[nm] = None
                    continue
                p = pick_idx(r["cands"])
                rec[nm] = {"pick": r["cands"][p]["sql"], "pick_rank": p, "greedy": r["cands"][0]["sql"],
                           "greedy_ok": bool(r["cands"][0].get("ok")),
                           "ok_ranks": [j for j, c in enumerate(r["cands"]) if c.get("ok")],
                           "pick_labels": labels(d, r0["gold_sql"], r["cands"][p]["sql"])}
            if hc[i]:
                kind = "uncovered" if rec["v4d"] is None else ("pick" if on[i] else "recall")
                rec["loss_kind"] = kind
                cnt[kind] += 1
                if rec["v4d"]:
                    cnt.update("pick:" + x for x in rec["v4d"]["pick_labels"])
            rows.append(rec)
        (OUT / f"diff_{d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary[d] = cnt
        say(f"{d:13s} lost {sum(1 for r in rows if r.get('lost'))} won {sum(1 for r in rows if r.get('won'))}  {dict(cnt)}")
    return summary


# ---------------------------------------------------------------------------------------------------------- J
def section_j():
    say("\n## J. car_retails description audit (what the arm's cards show, bird_desc.describe)")
    D = bird_desc.load_dir(bird_desc.DEFAULT_DIR)["car_retails"]
    c = SchemaCard.load(ROOT / "data/sys1/bird_train/cards/car_retails.json")
    seen = collections.defaultdict(list)
    for t in c.tables:
        for x in t.columns:
            e = D.get(bird_desc._norm(t.name), {}).get(bird_desc._norm(x.name))
            if not e:
                continue
            s = bird_desc.describe(x.name, e)
            if not s:
                continue
            flags = []
            if "string" in s.lower() and x.type.lower().startswith("int"):
                flags.append("says string, column is integer")
            if re.search(r'"[A-Za-z]+"', s) or "+" in s or " - " in s.lower() or " x " in s:
                flags.append("formula / names another column")
            if bird_desc._norm(s).startswith(bird_desc._norm(e[0])) and len(s) - len(e[0]) < 4:
                flags.append("restates expanded name only")
            seen[s].append(f"{t.name}.{x.name}")
            say(f"  {t.name}.{x.name}: {s!r}" + (f"   [{'; '.join(flags)}]" if flags else ""))
    for s, where in seen.items():
        if len(where) > 1:
            say(f"  identical text on {len(where)} columns {where}: {s!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true", help="reuse runs/analysis/v4desc_regression/joinfix_proxy.json")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    C, N = load(CTL), load(ARM)
    hc, oc, hn, on = section_a(C, N)
    section_b()
    section_c(C, N)
    section_d(C, N, hc, oc, not a.no_exec)
    section_e(C, N)
    section_f(C, N)
    section_g()
    section_h()
    section_i(C, N, hc, oc, hn, on)
    section_j()
    (OUT / "report.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
