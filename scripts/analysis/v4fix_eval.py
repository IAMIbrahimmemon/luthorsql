"""v4fix PREREG evaluation (runs/phase1/v4fix/PREREG.md). DECISION only (796 q, ORIGINAL gold); rows of any other id
are skipped while parsing; no BIRD dev file is read. Mac CPU.

  uv run --offline python scripts/analysis/v4fix_eval.py                   # the v4fix arm (after the run)
  uv run --offline python scripts/analysis/v4fix_eval.py --arm v4desc      # dry run on the killed v4desc arm

PRIMARY: the arm's own k=8 list + own validator vs the v3 seed-control mean, per question
    control_q = mean over R1 (runs/r1r2/r1), R3a (runs/phase1/seed), R3b (runs/phase1/seedfull) of hit_q
    delta = 100 * mean_q (hit_arm_q - control_q);  exact DB-cluster sign-flip on the per-DB sums (8 DBs, 256 patterns)
    plus the paired test against each control run alone (luthor.stats: DB sign-flip and McNemar).
SECONDARY (no decision): first pick (greedy), oracle@8, coverage, delta on questions every run covers; k=32 distinct
list oracle@32 vs R1's own (67.84); single-model k=32 pick by (a) the arm's own validator, (b) the R1 wide validator
(R1 baseline 57.29), (c) VALHARD (R1 baseline 59.05) -- read from <arm>/r1/rr/wide/dev_k32_by_{own,r1wide,valhard}.jsonl
when present.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_eval import IDS, db_of  # noqa: E402
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402

CONTROLS = {"R1": "runs/r1r2/r1/rr/dev_scored.jsonl", "R3a": "runs/phase1/seed/r1/rr/dev_scored.jsonl",
            "R3b": "runs/phase1/seedfull/r1/rr/dev_scored.jsonl"}
R1_WIDE = "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl"      # R1 k=32 list (src r1|both) + R1 wide validator
VALHARD = "runs/phase1/valhard/valhard/heldout_by_new.jsonl"    # R1 k=32 pool + VALHARD


def load_ids(p: Path, ids: set) -> dict:
    out = {}
    with open(p) as f:
        for line in f:
            m = re.match(r'\{"id": "([^"]+)"', line)
            if m and m.group(1) not in ids:
                continue
            r = json.loads(line)
            if r["id"] in ids:
                out[r["id"]] = r
    return out


def pick(cands, src=None):
    cs = [c for c in cands if src is None or c.get("src") in src]
    if not cs:
        return False
    live = [c for c in cs if c.get("runs") and c.get("nonnull")] or cs
    return bool(max(live, key=lambda c: c.get("rr") if c.get("rr") is not None else -1e9).get("ok"))


def hits(S, ids, src=None):
    return {i: bool(S.get(i)) and pick(S[i]["cands"], src) for i in ids}


def oracle(S, ids, k=None, src=None):
    return {i: bool(S.get(i)) and any(c.get("ok") for c in S[i]["cands"][:k] if src is None or c.get("src") in src)
            for i in ids}


def frac_signflip(arm: dict, ctl: dict, db: dict, alpha=0.05):
    """Exact DB sign-flip with a fractional control (mean of runs): D_g = sum over DB g of (arm_q - ctl_q)."""
    keys = sorted(set(db[i] for i in arm))
    D = np.array([sum(arm[i] - ctl[i] for i in arm if db[i] == k) for k in keys], float)
    n = np.array([sum(1 for i in arm if db[i] == k) for k in keys], float)
    S = np.array(list(itertools.product((-1.0, 1.0), repeat=len(keys))))
    p = float(np.mean(np.abs(S @ D) >= abs(D.sum()) - 1e-9))
    grid = np.arange(-0.2, 0.2001, 0.0005)
    keep = [g for g in grid if np.mean(np.abs(S @ (D - n * g)) >= abs((D - n * g).sum()) - 1e-9) >= alpha]
    return 100 * D.sum() / n.sum(), p, (100 * min(keep), 100 * max(keep)) if keep else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="v4fix", help="runs/phase1/<arm>")
    a = ap.parse_args()
    ids = list(IDS)
    idset = set(ids)
    DB = {i: d for i, d in db_of().items() if i in idset}
    rr = ROOT / f"runs/phase1/{a.arm}/r1/rr"
    A = load_ids(rr / "dev_scored.jsonl", idset)
    C = {k: load_ids(ROOT / p, idset) for k, p in CONTROLS.items()}
    ha = hits(A, ids)
    hc = {k: hits(S, ids) for k, S in C.items()}
    mean_c = {i: float(np.mean([hc[k][i] for k in C])) for i in ids}
    pct = lambda h: 100 * sum(h.values()) / len(ids)
    print(f"arm {a.arm}: DECISION {pct(ha):.2f} ({sum(ha.values())}/{len(ids)})   controls: "
          + ", ".join(f"{k} {pct(h):.2f}" for k, h in hc.items()) + f"   mean {100 * np.mean(list(mean_c.values())):.2f}")
    d, p, ci = frac_signflip({i: float(ha[i]) for i in ids}, mean_c, DB)
    verdict = "PASS" if d >= 2.0 and p < 0.05 else ("KILL" if d < 1.0 else "NO PASS")
    print(f"PRIMARY delta vs v3 seed mean {d:+.2f} (DB sign-flip p = {p:.3f}, 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]) -> {verdict}")
    for k, h in hc.items():
        c, q = cluster_signflip(h, ha, DB), question_signflip(h, ha)
        print(f"  vs {k}: {c['delta']:+.2f} (DB p = {c['p']:.3f}, CI [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}], McNemar +{q['wins']}/-{q['losses']})")
    g = {i: bool(A.get(i)) and bool(A[i]["cands"]) and bool(A[i]["cands"][0].get("ok")) for i in ids}
    o8 = oracle(A, ids, 8)
    both = [i for i in ids if i in A and all(i in S for S in C.values())]
    print(f"secondary: first pick {pct(g):.2f}, oracle@8 {pct(o8):.2f}, coverage {len(A)} "
          f"(newly covered vs R1: {sum(1 for i in A if i not in C['R1'])}); on the {len(both)} covered by all: arm "
          f"{sum(ha[i] for i in both)} vs control mean {sum(mean_c[i] for i in both):.1f}")
    print("per DB (arm / control mean): " + ", ".join(
        f"{k} {sum(ha[i] for i in ids if DB[i] == k)}/{sum(mean_c[i] for i in ids if DB[i] == k):.1f}" for k in sorted(set(DB.values()))))
    W = rr / "wide/cands_dev_k32.jsonl"
    if W.exists():
        Wa = load_ids(W, idset)
        R1w = load_ids(ROOT / R1_WIDE, idset)
        print(f"k=32 oracle: arm {pct(oracle(Wa, ids, 32)):.2f} vs R1 {pct(oracle(R1w, ids, 32, {'r1', 'both'})):.2f}")
    base = {"r1wide": pct(hits(load_ids(ROOT / R1_WIDE, idset), ids, {"r1", "both"})),
            "valhard": pct(hits(load_ids(ROOT / VALHARD, idset), ids))}
    for name in ("own", "r1wide", "valhard"):
        f = rr / f"wide/dev_k32_by_{name}.jsonl"
        ref = base.get(name)
        if f.exists():
            hv = hits(load_ids(f, idset), ids)
            print(f"k=32 + {name} validator: {pct(hv):.2f}" + (f" vs R1 pool {ref:.2f}" if ref is not None else ""))
        else:
            print(f"k=32 + {name} validator: (no {f.relative_to(ROOT)})" + (f"; R1 pool reference {ref:.2f}" if ref is not None else ""))


if __name__ == "__main__":
    main()
