"""Research problem 08-03, attack 1: pick from the R1 ∪ R2 candidate pool with one or both validators.

  python scripts/analysis/ens1_pick.py heldout          # choose the rule (held-out training DBs only)
  python scripts/analysis/ens1_pick.py dev RULE         # one paired check of the frozen RULE on ORIGINAL dev (all 1,534)

Inputs: data/ens1/out/<set>_by_{r1,r2}.jsonl (the pool scored by each validator on the GPU box).
Rules (all pick among candidates that run and return non-null rows, else among all, like pick_rerank):
  r2_own   R2 validator on R2's own list (the published pick; baseline)
  v2_pool  R2 validator over the pool        v1_pool  R1 validator over the pool
  sum_z    sum of both validators' scores, each z-normalised within the question
  sum_raw  plain sum of both validators' scores
Statistics: luthor.stats (exact DB-level sign-flip with inverted CI; McNemar for "these questions only").
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402

RULES = ("r2_own", "v2_pool", "v1_pool", "sum_z", "sum_raw")


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


def picks(set_):
    by1 = load(ROOT / f"data/ens1/out/{set_}_by_r1.jsonl")
    by2 = load(ROOT / f"data/ens1/out/{set_}_by_r2.jsonl")
    # R2's own list from its original candidate file (the pool's src tags are unreliable: a query repeated inside
    # R1's list was tagged "both")
    raw2 = load(ROOT / "runs/r1r2/r2/rr" / ("cands_dev.jsonl" if set_ == "heldout" else "cands_bird_dev.jsonl"))
    out = {k: {} for k in RULES}
    db = {}
    for i, r in by2.items():
        db[i] = r["db_id"]
        c2, c1 = r["cands"], by1[i]["cands"]
        assert [c["sql"] for c in c1] == [c["sql"] for c in c2]
        s1 = np.array([c["rr"] for c in c1]) if c1 else np.zeros(0)
        s2 = np.array([c["rr"] for c in c2]) if c2 else np.zeros(0)
        z = lambda s: (s - s.mean()) / (s.std() + 1e-6) if len(s) > 1 else s * 0
        live = [j for j, c in enumerate(c2) if c.get("runs") and c.get("nonnull")] or list(range(len(c2)))
        mine = {c.get("sql") for c in raw2[i]["cands"]}
        own_all = [j for j, c in enumerate(c2) if c["sql"] in mine]
        own = [j for j in own_all if c2[j].get("runs") and c2[j].get("nonnull")] or own_all
        score = {"r2_own": (s2, own), "v2_pool": (s2, live), "v1_pool": (s1, live),
                 "sum_z": (z(s1) + z(s2), live), "sum_raw": (s1 + s2, live)}
        for k, (s, cand) in score.items():
            out[k][i] = bool(cand) and bool(c2[max(cand, key=lambda j: s[j])].get("ok"))
    oracle = {i: any(c.get("ok") for c in by2[i]["cands"]) for i in by2}
    return out, db, oracle


def report(set_, rules):
    P, db, orc = picks(set_)
    n = len(db)
    base = P["r2_own"]
    print(f"{set_}: n={n}, {len(set(db.values()))} databases, pool oracle {100 * sum(orc.values()) / n:.2f}")
    for k in rules:
        c = cluster_signflip(base, P[k], db)
        q = question_signflip(base, P[k])
        print(f"  {k:8s} {100 * sum(P[k].values()) / n:6.2f}  vs r2_own {c['delta']:+5.2f}  "
              f"DB sign-flip p={c['p']:.3f} CI [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]  McNemar p={q['p']:.3f} "
              f"(+{q['wins']}/-{q['losses']})")


if __name__ == "__main__":
    if sys.argv[1] == "heldout":
        report("heldout", RULES)
    elif sys.argv[1] == "dev":
        rule = sys.argv[2]
        assert rule in RULES
        report("dev", ["r2_own", rule])
