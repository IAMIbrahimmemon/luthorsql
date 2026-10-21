"""Keep-or-drop test for a change to the System-1 model: paired comparison of two runs' gate predictions
(gate_predictions.jsonl) on the same questions, 95% range from resampling whole databases. Questions a run
did not cover count as wrong for it. KEEP only when the whole 95% range is above zero.

  python scripts/compare_gate.py runs/sys1/joint/gate_predictions.jsonl runs/sys1/pilot2/gate_predictions.jsonl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sqlw.evalkit import paired  # noqa: E402


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


a, b = load(sys.argv[1]), load(sys.argv[2])
ids = set(a) | set(b)
db_of = {i: (a.get(i) or b.get(i))["db_id"] for i in ids}
sa = {i: int(bool(a.get(i, {}).get("correct"))) for i in ids}
sb = {i: int(bool(b.get(i, {}).get("correct"))) for i in ids}
r = paired(sa, sb, db_of)
print(json.dumps(r))
print("KEEP" if r["lo"] > 0 else "DROP (no significant gain)")
