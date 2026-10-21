"""Held-out -> ORIGINAL-dev mapping (research problems 06-11, 08-07): every system with both a held-out number and an
ORIGINAL BIRD dev (2024-06-27 gold, all 1,534) number, a fitted map dev = a + b * heldout, the plain offset, and their
residuals. Dev numbers are measurements only; this map is for *reading* held-out results, never for choosing on dev.

  python scripts/analysis/ho_dev_map.py > scripts/analysis/ho_dev_map.out

Held-out definition for the main table ("HO1000"): runs/<run>/rr/dev_scored.jsonl, 1,000 BIRD-train questions on the
9 held-out DBs, ALL grammar-v3-covered, gold = the compiled v3 plan SQL (not the ORIGINAL train SQL), retail_world
included (40 rows). Dev side ("DEV1534"): all 1,534 questions run, uncovered ones counted as run (R1/R2/ENS1 GPU files;
D0 from reports/fanout_r2/zero_gpu/hybrid_1534.jsonl). Rows marked `new` were computed for this report by re-running
the frozen reports/fanout_r1/stack/unify.py (STACK_CONSERVATIVE=1) on the held-out file and are not recorded elsewhere;
rows marked `rc` are recomputed below from files on disk when the file exists.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# (system, metric, HO1000 %, DEV1534 %, provenance, sources)
PAIRS = [
    ("D0", "first pick", 57.90, 43.42, "rc", "runs/sys1/r6c/sys1_full/rr/dev_metrics.json; docs/SYSTEM1_FANOUT_R2.md:24"),
    ("D0", "exec fallback", 59.90, 45.96, "new(dev)", "runs/sys1/logs_c/rr.log:61; hybrid_1534.jsonl"),
    ("D0", "fallback+VAL+DROP", 62.70, 51.83, "new", "unify.py STACK_CONSERVATIVE=1"),
    ("D0", "validator", 64.70, 51.30, "rc", "runs/sys1/logs_c/rr.log:335,350; FANOUT_R2:25"),
    ("D0", "validator+VAL+DROP", 66.10, 55.08, "new(ho)", "FANOUT_R2:27; hybrid_1534.stack_cons.jsonl"),
    ("D0", "validator+VAL+ADD+W+DROP", 67.00, 55.41, "rc", "reports/fanout_r1/stack/cons.out:3; FANOUT_R1:42"),
    ("D0", "oracle@8", 69.10, 54.69, "rc", "sys1_full dev_metrics.json; hybrid_1534.jsonl"),
    ("R1", "first pick", 59.70, 48.44, "rc", "runs/r1r2/r1/rr/dev_metrics.json; docs/SYSTEM1_R1R2_RESULTS.md:26"),
    ("R1", "exec fallback", 60.80, 50.59, "rc", "runs/r1r2/r1/logs/rrscore_r1.log:14"),
    ("R1", "fallback+VAL+DROP", 61.70, 52.80, "new", "unify.py STACK_CONSERVATIVE=1"),
    ("R1", "validator", 67.40, 55.28, "rc", "08_ckpt_union.out:2,8; R1R2_RESULTS"),
    ("R1", "validator+VAL+DROP", 68.40, 56.78, "new(ho)", "runs/r1r2/r1/rr/bird_dev_scored.stack_cons.jsonl; R1R2_RESULTS:29"),
    ("R1", "validator+VAL+ADD+W+DROP", 69.10, 56.91, "new(ho)", "r1 bird_dev_scored.stack_cons.jsonl"),
    ("R1", "oracle@8", 70.30, 59.45, "rc", "r1 dev_scored.jsonl / bird_dev_scored.jsonl"),
    ("R2", "first pick", 60.60, 48.83, "rc", "runs/r1r2/r2/rr/dev_metrics.json; R1R2_RESULTS:26"),
    ("R2", "exec fallback", 61.30, 51.37, "rc", "runs/r1r2/r2/logs/rrscore_r2.log:14; R1R2_RESULTS:31"),
    ("R2", "fallback+VAL+DROP", 62.50, 53.13, "new(ho)", "R1R2_RESULTS:31; reports/research/ONE_FORWARD_DEV.md:15"),
    ("R2", "validator", 67.40, 55.28, "rc", "03_slot_model.md:27; 08_ckpt_union.out"),
    ("R2", "validator+VAL+DROP", 68.80, 56.71, "new(ho)", "runs/r1r2/r2/rr/bird_dev_scored.stack_cons.jsonl; R1R2_RESULTS:29"),
    ("R2", "validator+VAL+ADD+W+DROP", 69.00, 56.58, "new(ho)", "r2 bird_dev_scored.stack_cons.jsonl"),
    ("R2", "oracle@8", 71.20, 60.23, "rc", "r2 dev_metrics.json; R1R2_RESULTS:28"),
    ("ENS1", "validator (sum_z)", 70.10, 58.21, "rc", "docs/SYSTEM1_ENS1_RESULTS.md:23,35; data/ens1/out"),
    ("ENS1", "validator+VAL+DROP", 71.00, 59.06, "new(ho)", "ENS1_RESULTS:36; data/ens1/out/dev_sum_z_scored.stack_cons.jsonl"),
    ("ENS1", "validator+VAL+ADD+W+DROP", 71.30, 58.93, "new(ho)", "dev_sum_z_scored.stack_cons.jsonl"),
]
# D0 repair configs behind the "dev gains are 1.95x held-out gains" claim: HO1000 default (non-conservative) VAL vs
# DEV1534u0 (only the 1,297 covered questions run, uncovered wrong). reports/fanout_r2/proof_protocol/proxies.out:11-22.
REPAIR = [("V:-", 64.70, 49.80), ("VAL", 66.00, 53.52), ("W", 64.70, 49.80), ("DUP", 65.00, 49.87), ("VG", 65.10, 51.24),
          ("DROP", 64.90, 51.04), ("ADD", 65.60, 50.00), ("ADD+W+DROP", 65.90, 51.30), ("VAL+W+DROP", 66.40, 54.30),
          ("VAL+ADD+W", 67.00, 53.91), ("VAL+ADD+DROP", 67.20, 54.43), ("VAL+ADD+W+DROP", 67.30, 54.56)]


def _rows(p):
    return [json.loads(l) for l in open(REPO / p)]


def recompute():
    """Re-derive the cheap rows from files on disk; returns [(label, table value, recomputed)]."""
    out = []
    first = lambda r: bool(r["cands"]) and bool(r["cands"][0].get("ok"))  # noqa: E731
    orc = lambda r: any(c.get("ok") for c in r["cands"])  # noqa: E731
    for run in ("r1", "r2"):
        ho = REPO / f"runs/r1r2/{run}/rr/dev_scored.jsonl"
        dv = REPO / f"runs/r1r2/{run}/rr/bird_dev_scored.jsonl"
        st = REPO / f"runs/r1r2/{run}/rr/bird_dev_scored.stack_cons.jsonl"
        tab = {(s, m): (h, d) for s, m, h, d, *_ in PAIRS if s == run.upper()}
        if ho.exists():
            H = _rows(ho)
            out.append((f"{run} HO first", tab[(run.upper(), "first pick")][0], 100 * np.mean([first(r) for r in H])))
            out.append((f"{run} HO oracle@8", tab[(run.upper(), "oracle@8")][0], 100 * np.mean([orc(r) for r in H])))
        if dv.exists():
            D = _rows(dv)
            out.append((f"{run} dev first (n={len(D)}, /1534)", tab[(run.upper(), "first pick")][1], 100 * sum(map(first, D)) / 1534))
            out.append((f"{run} dev oracle@8 (/1534)", tab[(run.upper(), "oracle@8")][1], 100 * sum(map(orc, D)) / 1534))
        if st.exists():
            S = _rows(st)
            out.append((f"{run} dev validator (/1534)", tab[(run.upper(), "validator")][1], 100 * sum(r["res"]["V:-"][0] for r in S) / 1534))
            out.append((f"{run} dev VAL+DROP (/1534)", tab[(run.upper(), "validator+VAL+DROP")][1],
                        100 * sum(r["res"]["V:VAL+DROP"][0] for r in S) / 1534))
    return out


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    return a, b, res


def main():
    print(__doc__.strip().split("\n")[0])
    print("\n== provenance check: rows recomputed from files on disk (table value vs recomputed) ==")
    for lab, t, r in recompute():
        print(f"  {lab:32s} table {t:6.2f}  recomputed {r:6.2f}  {'OK' if abs(t - r) < 0.006 else 'MISMATCH'}")

    print("\n== pairs: HO1000 (v3-covered only, plan-rendered gold, retail_world in) vs ORIGINAL dev, all 1,534 ==")
    print(f"  {'system':5s} {'metric':26s} {'held-out':>8s} {'dev':>7s} {'dev-ho':>7s}  prov")
    for s, m, h, d, prov, src in PAIRS:
        print(f"  {s:5s} {m:26s} {h:8.2f} {d:7.2f} {d - h:+7.2f}  {prov}")
    x = [p[2] for p in PAIRS]
    y = [p[3] for p in PAIRS]
    sysl = [p[0] for p in PAIRS]
    a, b, res = ols(x, y)
    off = np.mean(np.subtract(y, x))
    res_off = np.subtract(y, x) - off
    n = len(PAIRS)
    print(f"\n== fits over all {n} pairs (points are NOT independent: 4 systems x nested metrics) ==")
    print(f"  linear: dev = {a:+.2f} + {b:.3f} * heldout   residual SD {res.std(ddof=2):.2f}  max |res| {abs(res).max():.2f}")
    print(f"  offset: dev = heldout {off:+.2f}                 residual SD {res_off.std(ddof=1):.2f}  max |res| {abs(res_off).max():.2f}")
    # leave-one-system-out
    print("\n  leave-one-system-out prediction error (fit on the other systems, predict this one's rows):")
    for s in sorted(set(sysl)):
        tr = [i for i in range(n) if sysl[i] != s]
        te = [i for i in range(n) if sysl[i] == s]
        a2, b2, _ = ols([x[i] for i in tr], [y[i] for i in tr])
        off2 = np.mean([y[i] - x[i] for i in tr])
        e_lin = [y[i] - (a2 + b2 * x[i]) for i in te]
        e_off = [y[i] - (x[i] + off2) for i in te]
        print(f"    {s:5s} n={len(te)}  linear mean err {np.mean(e_lin):+5.2f} (RMSE {math.sqrt(np.mean(np.square(e_lin))):.2f})"
              f"   offset mean err {np.mean(e_off):+5.2f} (RMSE {math.sqrt(np.mean(np.square(e_off))):.2f})")
    # system-cluster bootstrap of slope
    rnd = random.Random(0)
    S = sorted(set(sysl))
    bs = []
    for _ in range(2000):
        pick = [rnd.choice(S) for _ in S]
        idx = [i for s in pick for i in range(n) if sysl[i] == s]
        if len({x[i] for i in idx}) > 2:
            bs.append(ols([x[i] for i in idx], [y[i] for i in idx])[1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  slope b: system-cluster bootstrap 95% range [{lo:.2f}, {hi:.2f}] (only 4 clusters: a rough range, not a CI)")

    print("\n== per metric across systems: does a held-out DIFFERENCE carry over? (dev delta / held-out delta) ==")
    metrics = sorted({p[1] for p in PAIRS})
    T = {(p[0], p[1]): (p[2], p[3]) for p in PAIRS}
    ratios = []
    for m in metrics:
        segs = []
        for s1, s2 in (("D0", "R1"), ("R1", "R2"), ("D0", "R2"), ("R2", "ENS1")):
            if (s1, m) in T and (s2, m) in T:
                dh = T[(s2, m)][0] - T[(s1, m)][0]
                dd = T[(s2, m)][1] - T[(s1, m)][1]
                segs.append(f"{s1}->{s2} ho {dh:+.2f} dev {dd:+.2f}")
                if abs(dh) >= 1.0:
                    ratios.append(dd / dh)
        print(f"  {m:26s} " + " | ".join(segs))
    print(f"  dev/held-out delta ratio over system pairs with |held-out delta| >= 1: median {np.median(ratios):.2f}, "
          f"range [{min(ratios):.2f}, {max(ratios):.2f}], n={len(ratios)}")
    print("  sign agreement: R1->R2 first pick ho +0.90 / dev +0.39; validator ho 0.00 / dev 0.00; "
          "VAL+DROP ho +0.40 / dev -0.07 (sign flips at sub-point deltas)")

    print("\n== within one system: D0 repair configs (HO1000 default VAL vs DEV1534u0, proxies.out:11-22) ==")
    xr = [r[1] for r in REPAIR]
    yr = [r[2] for r in REPAIR]
    ar, br, rr = ols(xr, yr)
    gh = np.subtract(xr[1:], xr[0])
    gd = np.subtract(yr[1:], yr[0])
    print(f"  linear: dev = {ar:+.2f} + {br:.2f} * heldout   residual SD {rr.std(ddof=2):.2f};  "
          f"mean gain over V:- held-out {gh.mean():+.2f}, dev {gd.mean():+.2f} (ratio {gd.mean() / gh.mean():.2f})")

    print("\n== held-out definitions in the repo (which set each number used) ==")
    print("  HO1000  runs/*/rr/dev_scored.jsonl: 1,000 v3-covered, plan-rendered gold, retail_world (40) in. All rows above.")
    print("  HO960   HO1000 minus retail_world: R2 first 59.58, validator 66.46, oracle 70.31 (about -1.0 vs HO1000).")
    print("  HO-train1455 / HO-A1395  training-time greedy evals (train_sq2.py), paired only with the training GATE "
          "(first pick on 1,297 covered dev, uncovered wrong); 5-checkpoint Spearman +0.70 mixes three definitions.")
    print("  ho200 / ho154  ORIGINAL train gold, Mac int8 decode: R2 greedy 53.90 on 154 valid vs dev200 46.94 on 196.")
    print("  v2 decision/reserved (data/sys1/heldout_v2): ORIGINAL gold, all executable questions incl. 16% v3-uncovered, "
          "no retail_world. A System-1 pipeline with no fallback scores 0 on uncovered items, so v2 ~= HO960 x 960/1144 "
          "(= 0.839 x HO960) for such systems: R2 first pick 48.74 on the decision set (scripts/analysis/heldout_v2.py score).")
    print(f"\nReading: on HO1000 the plain offset (dev ~= held-out {off:+.2f}, residual SD {res_off.std(ddof=1):.2f}) is as "
          f"good as the line (slope {b:.2f}); predicting a new system's dev level from held-out carries about +-2 points "
          "(leave-one-system-out RMSE up to 2.1 for the offset, worst on D0, whose dev rows mix GPU and Mac-int8 decodes). "
          "Across systems, held-out deltas map to dev deltas with a median ratio near 1.3 but a range that includes sign "
          "flips; a held-out delta below ~1 point does not predict the sign of the dev delta. Re-fit on v2 once systems "
          "are scored on it.")


if __name__ == "__main__":
    main()
