"""Results-aware validator vs box W's wide R2 validator on the k=32 union pool (runs/phase1/valprev/PREREG.md)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from phase1_eval import compare, load, single_hits, sum_z_hits  # noqa: E402

W = ROOT / "runs/phase1/p1w/p1w"
P = ROOT / "runs/phase1/valprev/valprev/heldout_union_by_prev.jsonl"
by1, by2, prev = load(W / "heldout_union_by_r1.jsonl"), load(W / "heldout_union_by_r2.jsonl"), load(P)
a, oa = single_hits(by2)
b, ob = single_hits(prev)
print("PRIMARY")
c = compare("wide R2 validator (no preview)", a, "preview validator", b, oa, ob)
print(f"  -> {'PASS' if c['delta'] >= 1.0 and c['p'] < 0.10 else ('KILL' if c['delta'] < 0 else 'NO PASS')}")
print("secondary")
s0, o0 = sum_z_hits(by1, by2)
s1, o1 = sum_z_hits(by1, prev)
compare("sum_z(wide R1, wide R2)", s0, "sum_z(wide R1, preview)", s1, o0, o1)
