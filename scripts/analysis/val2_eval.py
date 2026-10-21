"""VAL2 (runs/phase1/val2/PREREG.md): new validator alone vs R1's wide validator alone on the R1 ∪ R2 wide pool.

  python scripts/analysis/val2_eval.py runs/phase1/val2/val2     # pulled /workspace/val2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_eval import ROOT, compare, load, single_hits, sum_z_hits  # noqa: E402


def main():
    d = Path(sys.argv[1])
    w = ROOT / "runs/phase1/p1w/p1w"
    r1, r2 = load(w / "heldout_union_by_r1.jsonl"), load(w / "heldout_union_by_r2.jsonl")
    new, on = single_hits(load(d / "heldout_by_new.jsonl"))
    base, ob = single_hits(r1)
    print("PRIMARY")
    c = compare("R1 wide validator alone (wide pool)", base, "VAL2 validator alone (wide pool)", new, ob, on)
    print(f"  -> {'PASS' if c['delta'] >= 1.5 and c['p'] < 0.10 else ('KILL' if c['delta'] < 0.75 else 'NO PASS')}")
    print("secondary (info)")
    sz, osz = sum_z_hits(r1, r2)
    compare("sum_z r1+r2 (58.17)", sz, "VAL2 validator alone", new, osz, on)


if __name__ == "__main__":
    main()
