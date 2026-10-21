"""Task 1: attribute the k=8 "ensemble" gain (+2.4 on DECISION) to pool union vs validator voting.

Reads only DECISION ids (phase1_eval.IDS); rows of other held-out ids are dropped at load.
Arms per pool size (k=8 r1r2 runs, k=32 wide p1w runs):
  own_Rx      validator Rx on its own run's list (rr/dev_scored.jsonl)
  Rx@union    validator Rx on the R1 ∪ R2 union list
  Rx@Ry-list  validator Rx on the union restricted to run Ry's candidates (src in {Ry, both})
  sumz@Ry     sum_z of both validators on run Ry's candidates only (voting without union)
  sumz@union  sum_z on the union (the ENS1 / wide headline)

  uv run python scripts/val2/attribute_ensemble.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import phase1_eval as pe  # noqa: E402

IDS = set(pe.IDS)
assert pe.SPLIT == "decision"


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p)) if r["id"] in IDS}


def hits_single(scored, src=None):
    h, o = {}, {}
    for i in pe.IDS:
        r = scored.get(i)
        c = [] if r is None else [x for x in r["cands"] if src is None or x.get("src") in src]
        h[i] = bool(c) and pe.pick(c, np.array([x.get("rr", -1e9) if x.get("rr") is not None else -1e9 for x in c], float))
        o[i] = any(x.get("ok") for x in c)
    return h, o


def hits_sumz(by1, by2, src=None):
    h, o = {}, {}
    for i in pe.IDS:
        if i not in by1:
            h[i] = o[i] = False
            continue
        keep = [j for j, x in enumerate(by1[i]["cands"]) if src is None or x.get("src") in src]
        c1 = [by1[i]["cands"][j] for j in keep]
        c2 = [by2[i]["cands"][j] for j in keep]
        assert [x["sql"] for x in c1] == [x["sql"] for x in c2]
        if not c1:
            h[i] = o[i] = False
            continue
        s = pe.z(pe.rr_arr(c1)) + pe.z(pe.rr_arr(c2))
        h[i] = pe.pick(c2, s)
        o[i] = any(x.get("ok") for x in c2)
    return h, o


def pct(h):
    return 100 * sum(h.values()) / len(pe.IDS)


def block(name, own1, own2, u1, u2):
    print(f"\n=== {name} (DECISION n={len(pe.IDS)}) ===")
    R1, R2 = {"r1", "both"}, {"r2", "both"}
    arms = {
        "own_R1 (R1 val, R1 list)": hits_single(own1),
        "own_R2 (R2 val, R2 list)": hits_single(own2),
        "R1 val @ R1-part of union": hits_single(u1, R1),
        "R2 val @ R2-part of union": hits_single(u2, R2),
        "R1 val @ R2-part of union": hits_single(u1, R2),
        "R2 val @ R1-part of union": hits_single(u2, R1),
        "sumz @ R1 list only": hits_sumz(u1, u2, R1),
        "sumz @ R2 list only": hits_sumz(u1, u2, R2),
        "R1 val @ union": hits_single(u1),
        "R2 val @ union": hits_single(u2),
        "sumz @ union": hits_sumz(u1, u2),
    }
    for k, (h, o) in arms.items():
        print(f"  {k:30s} {pct(h):6.2f}   oracle {pct(o):6.2f}")
    return arms


def main():
    k8 = block("k=8 pools (r1r2 runs, ENS1 union)",
               load(ROOT / "runs/r1r2/r1/rr/dev_scored.jsonl"), load(ROOT / "runs/r1r2/r2/rr/dev_scored.jsonl"),
               load(ROOT / "data/ens1/out/heldout_by_r1.jsonl"), load(ROOT / "data/ens1/out/heldout_by_r2.jsonl"))
    w = ROOT / "runs/phase1/p1w/p1w"
    k32 = block("k=32 wide pools (p1w runs, wide validators)",
                load(w / "r1/rr/dev_scored.jsonl"), load(w / "r2/rr/dev_scored.jsonl"),
                load(w / "heldout_union_by_r1.jsonl"), load(w / "heldout_union_by_r2.jsonl"))
    print("\n--- paired contrasts (DB sign-flip / McNemar) ---")
    for tag, A in (("k=8", k8), ("k=32", k32)):
        for a, b in (("own_R2 (R2 val, R2 list)", "R2 val @ union"),       # union effect, validator fixed
                     ("own_R1 (R1 val, R1 list)", "R1 val @ union"),
                     ("R2 val @ union", "sumz @ union"),                     # voting effect, pool fixed
                     ("R1 val @ union", "sumz @ union"),
                     ("own_R2 (R2 val, R2 list)", "sumz @ R2 list only"),    # voting without union
                     ("own_R1 (R1 val, R1 list)", "sumz @ R1 list only"),
                     ("own_R2 (R2 val, R2 list)", "sumz @ union")):          # the headline +2.4 path
            print(f"[{tag}] {a}  ->  {b}")
            pe.compare(a, A[a][0], b, A[b][0], A[a][1], A[b][1])


if __name__ == "__main__":
    main()
