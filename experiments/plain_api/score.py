"""Score the plain-API cascade runs on RESERVED (experiments/plain_api/PREREG.md).

Accuracy: strict EX vs the original BIRD-train gold (read-only SQLite, 30 s, set of rows). Cost: each question's
logged API usage at Opus 5.5 list prices, summed per database; C = System 1 on kept questions + A1 on routed ones.

  python3 experiments/plain_api/score.py --runs runs/phase2/plain_api/r1 [r2 r3] --out experiments/plain_api/result.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "cost_confirmation"))
from luthor.stats import cluster_signflip  # noqa: E402
from score import TRAIN, run  # noqa: E402  (the registered experiment's executor and BIRD-train path, LUTHOR_TRAIN_JSON / LUTHOR_TRAIN_DBS)

def _first(*ps):  # development layout first, then the public release layout
    return next((p for p in ps if p.exists()), ps[-1])


S1 = _first(ROOT / "runs/phase2/cost_confirmation/s1_reserved.jsonl", ROOT / "experiments/cost_confirmation/artifacts/s1_reserved.jsonl")
IDS = _first(ROOT / "data/sys1/heldout_v2/reserved.json", ROOT / "splits/reserved_ids.json")
S1_RENT = 0.00085  # $/question, rented H100 (registered experiment), sensitivity only
T = 3.0


def score_run(d: Path, gold, dbid, s1, res_cache):
    ids = sorted(gold)
    routed = {i for i in ids if s1[i]["margin"] is None or s1[i]["margin"] < T}
    cands = {i: {c["label"]: c["sql"] for c in s1[i].get("candidates", [])} for i in ids}
    rows = {}
    for arm in ("A0", "A1"):
        rows[arm] = {}
        for l in open(d / f"answers_{arm}.jsonl"):
            r = json.loads(l)
            rows[arm].setdefault(r["question_id"], r)
    missing = {arm: len(set(ids) - set(rows[arm])) for arm in rows}
    pred = {"A0": {i: rows["A0"][i]["sql"] or None for i in rows["A0"]},
            "A1": {i: (cands[i].get(r["pick"]) if r["pick"].upper() != "NONE" else None) or r["sql"] or None
                   for i, r in rows["A1"].items()}}
    pred["C"] = {i: (pred["A1"].get(i) if i in routed else s1[i]["sql"]) for i in ids}
    pred["S1"] = {i: s1[i]["sql"] for i in ids}
    tasks = sorted({(dbid[i], s) for i in ids for s in [gold[i]] + [p.get(i) for p in pred.values()] if s} - set(res_cache))
    with ProcessPoolExecutor(8) as ex:
        res_cache.update(dict(ex.map(run, tasks, chunksize=16)))

    def ok(i, s):
        g = res_cache.get((dbid[i], gold[i]))
        p = res_cache.get((dbid[i], s)) if s else None
        return g is not None and p is not None and p == g

    R = {arm: {i: ok(i, pred[arm].get(i)) for i in ids} for arm in pred}
    dbs = sorted(set(dbid.values()))
    cost = {k: defaultdict(float) for k in ("A0", "A1", "C", "A1_rent", "C_rent")}
    for i in ids:
        d0 = dbid[i]
        cost["A0"][d0] += rows["A0"].get(i, {}).get("dollars", 0.0)
        a1 = rows["A1"].get(i, {}).get("dollars", 0.0)
        cost["A1"][d0] += a1
        cost["A1_rent"][d0] += a1 + S1_RENT
        cost["C"][d0] += a1 if i in routed else 0.0
        cost["C_rent"][d0] += (a1 if i in routed else 0.0) + S1_RENT
    rng = random.Random(20261026)

    def saving(arm, base, sample):
        cb, ca = sum(cost[base][x] for x in sample), sum(cost[arm][x] for x in sample)
        return 1 - ca / cb if cb else float("nan")

    def with_ci(arm, base):
        boot = sorted(saving(arm, base, [rng.choice(dbs) for _ in dbs]) for _ in range(10000))
        return {"point": round(saving(arm, base, dbs), 4), "ci95": [round(boot[249], 4), round(boot[9749], 4)]}

    tok = {arm: defaultdict(int) for arm in rows}
    for arm in rows:
        for r in rows[arm].values():
            for k, v in r["usage"].items():
                tok[arm][k] += v
    n = len(ids)
    return {
        "n": n, "routed": len(routed), "missing": missing,
        "not_submitted": {arm: sum(not r["submitted"] for r in rows[arm].values()) for arm in rows},
        "ex": {arm: round(100 * sum(v.values()) / n, 2) for arm, v in R.items()},
        "cost_per_q": {k: round(sum(v.values()) / n, 5) for k, v in cost.items()},
        "saving_C_vs_A0": with_ci("C", "A0"), "saving_C_vs_A1": with_ci("C", "A1"),
        "saving_C_vs_A0_s1_rented": with_ci("C_rent", "A0"), "saving_C_vs_A1_s1_rented": with_ci("C_rent", "A1_rent"),
        "C_minus_A0": cluster_signflip(R["A0"], R["C"], dbid), "C_minus_A1": cluster_signflip(R["A1"], R["C"], dbid),
        "A1_minus_A0": cluster_signflip(R["A0"], R["A1"], dbid),
        "tokens": {arm: dict(v) for arm, v in tok.items()},
        "turns_per_q": {arm: round(sum(r["turns"] for r in rows[arm].values()) / max(1, len(rows[arm])), 2) for arm in rows},
        "sql_per_q": {arm: round(sum(r["sql_executions"] for r in rows[arm].values()) / max(1, len(rows[arm])), 2) for arm in rows},
        "routed_vs_kept_A0_cost_per_q": {
            "routed": round(sum(rows["A0"][i]["dollars"] for i in routed if i in rows["A0"]) / len(routed), 5),
            "kept": round(sum(rows["A0"][i]["dollars"] for i in set(ids) - routed if i in rows["A0"]) / (n - len(routed)), 5)},
        "cost_per_db": {k: {x: round(v[x], 4) for x in dbs} for k, v in cost.items()},
        "_R": R,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tr = json.load(open(TRAIN))
    ids = json.load(open(IDS))["ids"]
    gold = {i: tr[int(i.rsplit("_", 1)[1])]["SQL"] for i in ids}
    dbid = {i: tr[int(i.rsplit("_", 1)[1])]["db_id"] for i in ids}
    s1 = {json.loads(l)["id"]: json.loads(l) for l in open(S1)}
    assert set(s1) == set(ids)
    cache = {}
    out = {"prereg": "experiments/plain_api/PREREG.md", "runs": {}}
    for d in a.runs:
        r = score_run(d, gold, dbid, s1, cache)
        r.pop("_R")
        out["runs"][d.name] = r
        print(d.name, json.dumps({k: r[k] for k in ("ex", "cost_per_q", "saving_C_vs_A0", "saving_C_vs_A1", "missing")}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
