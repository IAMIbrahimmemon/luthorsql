"""Size and power of the paired tests in luthor.stats on ORIGINAL BIRD dev's real structure (all 1,534, 11 DBs).
MEASURE ONLY: real per-question outcomes give the discordance patterns; nothing is fitted.

Nulls (true effect 0), rejection rate at nominal 5% (want <= 5%):
  N1  question-level sign flip of a real pair's discordant questions (the 07-07 null)
  N2  heterogeneous DBs: each DB gets a true effect +h or -h points (random signs, so symmetric), sum ~0, plus churn
Power: true +0.5 / +1 / +2 / +3 points spread at random (07-07's simulation), with 2% two-way churn.
  .venv/bin/python -I scripts/analysis/paired_test_size.py
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "src"))
from luthor.stats import cluster_signflip, percentile_bootstrap, question_signflip  # noqa: E402


def L(p):
    return [json.loads(l) for l in open(R / p)]


sysm, dbof = {}, {}
for tag, base in (("R1", "runs/r1r2/r1/rr"), ("R2", "runs/r1r2/r2/rr")):
    for r in L(base + "/bird_dev_scored.stack_cons.jsonl"):
        dbof[r["id"]] = r["db"]
        sysm.setdefault(tag + "_val", {})[r["id"]] = bool(r["res"]["V:-"][0])
        sysm.setdefault(tag + "_valdrop", {})[r["id"]] = bool(r["res"]["V:VAL+DROP"][0])
ids = sorted(dbof)
N = len(ids)
rng = np.random.default_rng(0)


def tests(a, b):
    lo, hi = percentile_bootstrap(a, b, dbof, B=1000, seed=int(rng.integers(1 << 30)))
    return {"pct_boot": lo > 0 or hi < 0,
            "cluster_flip": cluster_signflip(a, b, dbof, grid_step=0.001)["p"] < 0.05,
            "mcnemar": question_signflip(a, b)["p"] < 0.05}


def rate(fn, sims):
    hits = {}
    for _ in range(sims):
        a, b = fn()
        for k, v in tests(a, b).items():
            hits[k] = hits.get(k, 0) + v
    return {k: round(100 * v / sims, 1) for k, v in hits.items()}


print("N1 question-level sign-flip null on real pairs (rejection %, nominal 5):")
for x, y in (("R2_val", "R2_valdrop"), ("R1_valdrop", "R2_valdrop"), ("R1_val", "R2_val")):
    A, B = sysm[x], sysm[y]

    def n1():
        a, b = {}, {}
        for i in ids:
            if rng.random() < 0.5:
                a[i], b[i] = A[i], B[i]
            else:
                a[i], b[i] = B[i], A[i]
        return a, b
    print(f"  {x} vs {y}: {rate(n1, 400)}")

base = np.array([sysm["R2_valdrop"][i] for i in ids], bool)
dbs = sorted(set(dbof.values()))
di = np.array([dbs.index(dbof[i]) for i in ids])


def perturb(gain_pts=0.0, db_effect_pts=0.0, churn=0.02):
    new = base.copy()
    k = int(round(churn * N))
    new[rng.choice(np.flatnonzero(~new), k, replace=False)] = True
    new[rng.choice(np.flatnonzero(new & base), k, replace=False)] = False
    if db_effect_pts:
        for g in range(len(dbs)):
            m = int(round(db_effect_pts / 100 * (di == g).sum()))
            if rng.random() < 0.5:
                pool = np.flatnonzero((~new) & (di == g))
                new[rng.choice(pool, min(m, len(pool)), replace=False)] = True
            else:
                pool = np.flatnonzero(new & (di == g))
                new[rng.choice(pool, min(m, len(pool)), replace=False)] = False
    if gain_pts:
        new[rng.choice(np.flatnonzero(~new), int(round(gain_pts / 100 * N)), replace=False)] = True
    return dict(zip(ids, base)), dict(zip(ids, new))


for h in (2.0, 4.0):
    print(f"N2 heterogeneous null, per-DB effects +/-{h} pts with random signs: {rate(lambda: perturb(db_effect_pts=h), 300)}")
print("power, true gain spread at random, 2% churn (detection %):")
for g in (0.5, 1.0, 2.0, 3.0):
    print(f"  +{g} pts: {rate(lambda: perturb(gain_pts=g), 300)}")
print("power with per-DB heterogeneity +/-2 pts on top of the gain:")
for g in (1.0, 2.0, 3.0):
    print(f"  +{g} pts: {rate(lambda: perturb(gain_pts=g, db_effect_pts=2.0), 300)}")
