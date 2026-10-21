"""Join-route guard proxy on saved v4desc candidates (Mac CPU, no model). Diagnostic only: DECISION is used to REPORT
the effect of a guard whose rule was fixed on training data (scripts/grammar/join_guard_check.py + the BIRD-train
lists below), never to tune it.

For each saved candidate SQL: convert it back to a grammar v4 plan (luthor.sq.convert, the gold converter), compile it
with the guard off (must reproduce the candidate SQL byte for byte, else the candidate is left as it is) and with each
guard (LUTHOR_JOIN_GUARD = same / name / direct, src/luthor/sq/plan.py). When the guarded SQL differs, re-execute it
against the ORIGINAL gold (BIRD EX: row-set equality, as v4desc_regression.py section D) and replace the candidate's
ok / runs / nonnull flags. The validator scores are kept (approximation: a re-routed candidate keeps its old score)
and freed beam slots are not refilled, so this is a lower bound on re-decoding with the guard.

  uv run --offline python scripts/analysis/join_guard_proxy.py            # DECISION k=8 + k=32 wide + BIRD-train lists
  uv run --offline python scripts/analysis/join_guard_proxy.py --no-train # skip the BIRD-train lists (~5 min)

Inputs (DECISION ids only, rows of any other id are skipped while parsing; no BIRD dev file is opened):
  runs/phase1/v4desc/r1/rr/dev_scored.jsonl        v4desc own k=8 list + own validator scores
  runs/phase1/v4desc/r1/rr/wide/cands_dev_k32.jsonl v4desc k=32 --distinct list (no scores; oracle only)
  runs/phase1/v4desc/r1/rr/cands_train.jsonl       v4desc k=8 lists on BIRD-train (5,022 q; rule selection side)
  controls: runs/phase1/seed (R3a), runs/r1r2/r1 (R1), runs/phase1/seedfull (R3b) dev_scored.jsonl
Writes runs/analysis/grammar_v4_fix/join_guard_proxy.{json,txt}.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["LUTHOR_GRAMMAR"] = "v4"  # v4desc's grammar (before the first luthor import; workers inherit it)
os.environ.pop("LUTHOR_JOIN_GUARD", None)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = ROOT / "runs/analysis/grammar_v4_fix"
RUN = ROOT / "runs/phase1/v4desc/r1/rr"
CARDS = ROOT / "data/plans_v4/train/bird_train/cards"
DBS = {"decision": "data/sys1/dbs/{db}/{db}.sqlite", "train": "data/raw/bird_train/train_databases/{db}/{db}.sqlite"}
CONTROLS = {"R3a": "runs/phase1/seed/r1/rr/dev_scored.jsonl", "R1": "runs/r1r2/r1/rr/dev_scored.jsonl",
            "R3b": "runs/phase1/seedfull/r1/rr/dev_scored.jsonl"}
GUARDS = ("same", "name", "direct")
LOG: list[str] = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def load_ids(p, ids):
    out = {}
    with open(p) as f:
        for line in f:
            m = re.match(r'\{"id": "([^"]+)"', line)
            if m and m.group(1) not in ids:  # never parse rows outside the requested split
                continue
            r = json.loads(line)
            if r["id"] in ids:
                out[r["id"]] = r
    return out


def run_sql(con, sql, sec=30.0):
    t0 = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - t0 > sec else 0, 10000)
    try:
        return set(map(tuple, con.execute(sql).fetchmany(100000)))
    except Exception:  # noqa: BLE001
        return None
    finally:
        con.set_progress_handler(None, 0)


def reroute_db(args):
    """One database: {sql: {guard: new_sql or None (unchanged)}} plus round-trip stats; re-executes changed SQL."""
    db, items, tmpl = args
    import luthor.sq.plan as pl
    from luthor.sq.convert import convert
    from luthor.sq.schema_card import SchemaCard
    graph = SchemaCard.load(CARDS / f"{db}.json").graph()
    con = sqlite3.connect(f"file:{ROOT / tmpl.format(db=db)}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    stats = collections.Counter()
    memo, golds, out = {}, {}, {}
    for qid, gold, sqls in items:
        res = {}
        for sql in sqls:
            if not sql or sql in res:
                continue
            if sql not in memo:
                conv = convert(sql, graph)
                m = {"rt": "no_plan"}
                if conv.plan is not None:
                    try:
                        off = pl.compile_plan(conv.plan, graph)
                    except pl.CompileError:
                        off = None
                    m["rt"] = "exact" if off == sql else "inexact"
                    if off == sql:
                        for g in GUARDS + ("v3route",):
                            if g == "v3route":  # the same plan routed as grammar v3 does (hint pairs never edges)
                                pl.JOIN_HINT_MODE = "disconnected"
                            else:
                                pl.JOIN_GUARD = g
                            try:
                                new = pl.compile_plan(conv.plan, graph)
                            except pl.CompileError:
                                new = "<compile error>"
                            finally:
                                pl.JOIN_GUARD, pl.JOIN_HINT_MODE = "off", "pairs"
                            m[g] = None if new == sql else new
                memo[sql] = m
            m = memo[sql]
            stats["rt:" + m["rt"]] += 1
            r = {"rt": m["rt"]}
            for g in GUARDS + ("v3route",):
                new = m.get(g)
                if not new:
                    continue
                if qid not in golds:
                    golds[qid] = run_sql(con, gold)
                o = None if new == "<compile error>" else run_sql(con, new)
                r[g] = {"sql": new, "ok": bool(o is not None and golds[qid] is not None and o == golds[qid]),
                        "runs": o is not None, "nonnull": bool(o) and any(any(v is not None for v in row) for row in o)}
            res[sql] = r
        out[qid] = res
    con.close()
    return out, stats


def reroute_all(rows, tmpl, workers):
    by_db = collections.defaultdict(list)
    for i, r in rows.items():
        by_db[r["db_id"]].append((i, r["gold_sql"], [c.get("sql") for c in r["cands"]]))
    res, stats = {}, collections.Counter()
    with ProcessPoolExecutor(workers) as ex:
        for o, s in ex.map(reroute_db, [(d, it, tmpl) for d, it in sorted(by_db.items())]):
            res.update(o)
            stats.update(s)
    return res, stats


def apply(rows, res, guard):
    """A copy of rows with guarded candidates' flags replaced; flips counter."""
    F = copy.deepcopy(rows)
    flips = collections.Counter()
    for i, r in F.items():
        for c in r["cands"]:
            x = res.get(i, {}).get(c.get("sql"), {}).get(guard)
            if not x:
                continue
            flips["to_right" if x["ok"] and not c.get("ok") else "to_wrong" if c.get("ok") and not x["ok"] else "same"] += 1
            c.update(x)
    return F, flips


def pick_ok(cands, key="rr"):
    live = [j for j, c in enumerate(cands) if c.get("runs") and c.get("nonnull")] or list(range(len(cands)))
    return bool(live) and bool(cands[max(live, key=lambda j: cands[j].get(key, -1e9))].get("ok"))


def metrics(S, ids, k=None, scored=True):
    h = {i: bool(S.get(i)) and scored and pick_ok(S[i]["cands"]) for i in ids}
    o = {i: bool(S.get(i)) and any(c.get("ok") for c in S[i]["cands"][:k]) for i in ids}
    g = {i: bool(S.get(i)) and bool(S[i]["cands"]) and bool(S[i]["cands"][0].get("ok")) for i in ids}
    return h, o, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from phase1_eval import IDS, db_of  # DECISION ids (HELDOUT_SPLIT default)
    from luthor.stats import cluster_signflip, question_signflip
    ids = list(IDS)
    idset = set(ids)
    N = load_ids(RUN / "dev_scored.jsonl", idset)
    W = load_ids(RUN / "wide/cands_dev_k32.jsonl", idset)
    ctl = {k: load_ids(ROOT / p, idset) for k, p in CONTROLS.items()}
    DB = {i: d for i, d in db_of().items() if i in idset}
    dbs = sorted(set(DB.values()))
    summary = {}

    say("## DECISION, v4desc own k=8 list + own validator (scores kept), guard re-route + re-execution vs ORIGINAL gold")
    res, st = reroute_all(N, DBS["decision"], a.workers)
    say(f"round trip of candidate SQL (convert -> compile, guard off == candidate SQL): {dict(st)}")
    hc, oc, gc = metrics(ctl["R3a"], ids, 8)
    base = metrics(N, ids, 8)
    arms = {"v4desc": (N, collections.Counter())}
    for g in GUARDS:
        arms[f"v4desc+{g}"] = apply(N, res, g)
    say("arm                 hits   DEC%  oracle@8 greedy | vs R3a 53.77: delta, DB p, McNemar | flips")
    for name, (S, fl) in arms.items():
        h, o, gr = metrics(S, ids, 8)
        c = cluster_signflip(hc, h, DB)
        q = question_signflip(hc, h)
        summary[name] = {"hits": sum(h.values()), "pct": round(100 * sum(h.values()) / len(ids), 2),
                         "oracle8": sum(o.values()), "greedy": sum(gr.values()), "delta_vs_R3a": round(c["delta"], 2),
                         "p_db": round(c["p"], 3), "mcnemar": [q["wins"], q["losses"]], "flips": dict(fl),
                         "per_db": {d: [sum(h[i] for i in ids if DB.get(i) == d), sum(o[i] for i in ids if DB.get(i) == d),
                                        sum(gr[i] for i in ids if DB.get(i) == d)] for d in dbs}}
        say(f"{name:18s} {sum(h.values()):5d} {100 * sum(h.values()) / len(ids):6.2f} {sum(o.values()):8d} {sum(gr.values()):6d} | "
            f"{c['delta']:+.2f}, p={c['p']:.2f}, +{q['wins']}/-{q['losses']} | {dict(fl)}")
    for k, S in ctl.items():
        h, o, gr = metrics(S, ids, 8)
        summary[k] = {"hits": sum(h.values()), "pct": round(100 * sum(h.values()) / len(ids), 2), "oracle8": sum(o.values()),
                      "greedy": sum(gr.values()),
                      "per_db": {d: [sum(h[i] for i in ids if DB.get(i) == d), sum(o[i] for i in ids if DB.get(i) == d),
                                     sum(gr[i] for i in ids if DB.get(i) == d)] for d in dbs}}
        say(f"{k:18s} {sum(h.values()):5d} {100 * sum(h.values()) / len(ids):6.2f} {sum(o.values()):8d} {sum(gr.values()):6d}")
    say("\nper DB hits / oracle@8 / greedy")
    say("%-13s" % "db" + "".join("%20s" % k for k in ["R3a", "R1", "v4desc"] + [f"+{g}" for g in GUARDS]))
    for d in dbs:
        say("%-13s" % d + "".join("%20s" % "/".join(map(str, summary[k]["per_db"][d]))
                                  for k in ["R3a", "R1", "v4desc"] + [f"v4desc+{g}" for g in GUARDS]))

    say("\n## DECISION, v4desc k=32 --distinct list (no validator scores): oracle@k")
    resw, stw = reroute_all(W, DBS["decision"], a.workers)
    say(f"round trip: {dict(stw)}")
    for name, S in [("v4desc", W)] + [(f"v4desc+{g}", apply(W, resw, g)[0]) for g in GUARDS]:
        row = {k: sum(metrics(S, ids, k, scored=False)[1].values()) for k in (1, 4, 8, 16, 32)}
        summary.setdefault("wide", {})[name] = row
        say(f"{name:18s} " + "  ".join(f"@{k} {v}" for k, v in row.items()))

    say("\n## route expansion (one generator, decode time): each candidate also compiled with v3 FK routing; a question")
    say("counts when the list or any route variant is right (k=8 list: its 8 + up to 8 variants; k=32: oracle@32)")
    for nm, S, R, k in (("k=8 v4desc", N, res, 8), ("k=32 v4desc", W, resw, 32)):
        for base_g in ("off", "direct"):
            B = S if base_g == "off" else apply(S, R if S is N else resw, base_g)[0]
            o = 0
            extra = 0
            for i in ids:
                r = B.get(i)
                if not r:
                    continue
                cs = r["cands"][:k]
                orig = S[i]["cands"][:k]
                var = [R.get(i, {}).get(c.get("sql"), {}).get("v3route") for c in orig]
                extra += sum(bool(v) for v in var)
                o += any(c.get("ok") for c in cs) or any(v and v["ok"] for v in var)
            base = sum(metrics(B, ids, k, scored=False)[1].values())
            summary.setdefault("route_expansion", {})[f"{nm} {base_g}"] = {"oracle": base, "with_v3route": o, "variants": extra}
            say(f"{nm:12s} guard {base_g:6s}: oracle {base} -> {o} with v3-route variants ({extra} variant SQL added)")

    if not a.no_train:
        say("\n## BIRD-train v4desc k=8 lists (training data; in-sample generator): flips per guard")
        T = {}
        with open(RUN / "cands_train.jsonl") as f:
            for line in f:
                r = json.loads(line)
                T[r["id"]] = r
        rest, stt = reroute_all(T, DBS["train"], a.workers)
        say(f"round trip: {dict(stt)}")
        tids = list(T)
        for name, S in [("v4desc", T)] + [(f"v4desc+{g}", apply(T, rest, g)) for g in GUARDS]:
            S, fl = (S, {}) if name == "v4desc" else S
            _, o, gr = metrics(S, tids, 8, scored=False)
            summary.setdefault("train", {})[name] = {"n": len(tids), "oracle8": sum(o.values()), "greedy": sum(gr.values()),
                                                     "flips": dict(fl)}
            say(f"{name:18s} n={len(tids)} greedy {sum(gr.values())} oracle@8 {sum(o.values())} flips {dict(fl)}")

    (OUT / "join_guard_proxy.json").write_text(json.dumps(summary, indent=1))
    (OUT / "join_guard_proxy.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
