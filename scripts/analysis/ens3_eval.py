"""ENS3 (runs/phase1/ens3/PREREG.md): sum_z over three validators vs two on the R1 ∪ R2 ∪ R3a held-out pool.

  python scripts/analysis/ens3_eval.py runs/phase1/ens3/ens3     # pulled /workspace/ens3
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_eval import IDS, ROOT, compare, load, pick, rr_arr, sum_z_hits, z  # noqa: E402


def sum_hits(scored):
    hits, orc = {}, {}
    first = scored[0]
    for i in IDS:
        if i not in first:
            hits[i] = orc[i] = False
            continue
        cs = [s[i]["cands"] for s in scored]
        assert all([c["sql"] for c in x] == [c["sql"] for c in cs[0]] for x in cs)
        hits[i] = pick(cs[0], sum(z(rr_arr(c)) for c in cs))
        orc[i] = any(c.get("ok") for c in cs[0])
    return hits, orc


def main():
    d = Path(sys.argv[1])
    b = {v: load(d / f"heldout_by_{v}.jsonl") for v in ("r1", "r2", "r3a")}
    two, o = sum_hits([b["r1"], b["r2"]])
    three, _ = sum_hits([b["r1"], b["r2"], b["r3a"]])
    print("PRIMARY")
    c = compare("sum_z r1+r2 (union3 pool)", two, "sum_z r1+r2+r3a (union3 pool)", three, o, o)
    print(f"  -> {'PASS' if c['delta'] >= 1.0 and c['p'] < 0.10 else ('KILL' if c['delta'] < 0 else 'NO PASS')}")
    print("secondary (info)")
    w = ROOT / "runs/phase1/p1w/p1w"
    base, ob = sum_z_hits(load(w / "heldout_union_by_r1.jsonl"), load(w / "heldout_union_by_r2.jsonl"))
    compare("W pool sum_z r1+r2 (58.17 baseline)", base, "union3 sum_z r1+r2+r3a", three, ob, o)
    for v in ("r1", "r2", "r3a"):
        s, _ = sum_hits([b[v]])
        print(f"  single {v}: {100 * sum(s.values()) / len(IDS):.2f}")
    _ = np


if __name__ == "__main__":
    main()
