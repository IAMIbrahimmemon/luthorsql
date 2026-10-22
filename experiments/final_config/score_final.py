"""Score step 3 of experiments/final_config/PREREG.md: C* end-to-end on RESERVED vs the existing B1 and C runs 1-3.

  .venv/bin/python experiments/final_config/score_final.py --box <s2fin> --ref <s2res> <s2res_r2> <s2res_r3> \
      --tasks <dir> --out experiments/final_config/result_reserved.json
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cost_confirmation"))
sys.path.insert(0, str(ROOT / "experiments/top3_at_30"))
sys.path.insert(0, str(ROOT / "src"))
import score  # noqa: E402
from gate_eval import suspicious  # noqa: E402
from luthor.stats import cluster_signflip  # noqa: E402

S1_Q, T = 0.000849, 3.0


def answers(g):
    out = {}
    for f in sorted(glob.glob(g)):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            out.setdefault(r.get("question_id"), r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--ref", type=Path, nargs=3, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tr = json.load(open(score.TRAIN))
    s1 = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "runs/phase2/cost_confirmation/s1_reserved.jsonl")}
    ids = sorted(s1)
    db = {i: s1[i]["db_id"] for i in ids}
    gold = {i: score.run((db[i], tr[int(i.rsplit("_", 1)[1])]["SQL"]))[1] for i in ids}
    ok = lambda i, sql: bool(sql) and gold[i] is not None and score.run((db[i], sql))[1] == gold[i]
    base_routed = {i for i in ids if s1[i]["margin"] is None or s1[i]["margin"] < T}
    star_routed = base_routed | {i for i in ids if i not in base_routed and suspicious(s1[i])}

    def arm_ok(box, arm, routed, top):
        ans = answers(str(box / f"answers/{arm}/*.jsonl"))
        res, missing = {}, 0
        for i in ids:
            if i not in routed:
                res[i] = ok(i, s1[i]["sql"])
                continue
            r = ans.get(i)
            if r is None:
                missing += 1
                res[i] = False
                continue
            cands = {c["label"]: c["sql"] for c in s1[i].get("candidates", [])[:top]}
            p = str(r.get("pick", "NONE")).strip()
            res[i] = ok(i, cands.get(p) if p.upper() != "NONE" else r.get("sql"))
        return res, missing

    star, star_missing = arm_ok(a.box, "C", star_routed, 3)
    cruns = [arm_ok(b, "C", base_routed, 8)[0] for b in a.ref]
    n = len(ids)
    ex = lambda d: round(100 * sum(d.values()) / n, 2)
    c_mean = sum(ex(r) for r in cruns) / 3

    def cost_db(box, arm):
        out = defaultdict(float)
        for chunk, t in score.usage(a.tasks, box).get(arm, {}).items():
            out[chunk.rsplit("_", 1)[0]] += score.dollars(t)
        return out

    dbs = sorted(set(db.values()))
    nq = {d: sum(db[i] == d for i in ids) for d in dbs}
    star_cost = cost_db(a.box, "C")
    b1_cost = defaultdict(float)
    for b in a.ref:
        for d, v in cost_db(b, "B1").items():
            b1_cost[d] += v / 3
    for d in dbs:
        star_cost[d] += S1_Q * nq[d]
        b1_cost[d] += S1_Q * nq[d]
    sav = lambda smp: 1 - sum(star_cost[d] for d in smp) / sum(b1_cost[d] for d in smp)
    rng = random.Random(20261022)
    boot = sorted(sav([rng.choice(dbs) for _ in dbs]) for _ in range(10000))
    out = {
        "n": n, "routed": len(star_routed), "routed_base": len(base_routed), "missing_answers": star_missing,
        "ex_cstar": ex(star), "ex_c_runs": [ex(r) for r in cruns], "ex_c_mean": round(c_mean, 2),
        "cstar_minus_c_mean": round(ex(star) - c_mean, 2),
        "cstar_vs_c_run": [cluster_signflip(r, star, db) for r in cruns],
        "cost_per_q": {"cstar": round(sum(star_cost.values()) / n, 5), "b1_mean": round(sum(b1_cost.values()) / n, 5)},
        "saving_vs_b1": {"point": round(sav(dbs), 4), "ci95": [round(boot[249], 4), round(boot[9749], 4)]},
        "projected_test_ex": round(81.6 + ex(star) - c_mean, 2),
        "pass": {"saving_ge_30": sav(dbs) >= 0.30, "noninferior_2pt": ex(star) >= c_mean - 2.0},
    }
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
