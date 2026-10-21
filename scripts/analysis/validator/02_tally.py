"""Tally result cues on A (pick) vs B (validator's #2 distinct live result) over DECISION; reads 02_pairs.jsonl."""
import json
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
import sys
P = [json.loads(l) for l in open(ROOT / f"runs/analysis/validator/02_pairs{sys.argv[1] if len(sys.argv) > 1 else ''}.jsonl")]
wb = [p for p in P if "B" in p and "cues" in p]
print("questions", len(P), "with a #2 distinct live result", len(wb))
print("(A_ok, B_ok):", dict(Counter((p["A_ok"], p["B_ok"]) for p in wb)))
tab = Counter()
for p in wb:
    right = "A" if p["A_ok"] and not p["B_ok"] else "B" if p["B_ok"] and not p["A_ok"] else "-"
    for k, v in p["cues"].items():
        tab[(k, v, right)] += 1
print("\ncue -> fires for side X; truth (A right / B right / neither). Flip gain if obeyed = (cue B & B right) - (cue B & A right)")
for k in sorted({k for k, _, _ in tab}):
    for v in "AB":
        row = {r: tab[(k, v, r)] for r in "AB-"}
        if sum(row.values()):
            print(f"  {k:15s} says {v}: A right {row['A']:3d}  B right {row['B']:3d}  neither {row['-']:3d}")
gain = sum(tab[(k, "B", "B")] - tab[(k, "B", "A")] for k in {k for k, _, _ in tab})
print("naive obey-all-B-cues net (questions):", gain)
