"""Package check: does the BIRD submission package reproduce the scored BIRD dev run (dev look #1, variant A)?

Compares the box's predict_dev.json (run_bird.sh on BIRD dev with the SQL field removed) with the picks in
runs/phase1/devlook/look_A.json by SQL string only. No gold is read or executed, so this is not a dev look.

  .venv/bin/python scripts/bird_submit/compare_dev.py runs/phase1/bird_check/birdcheck/predict_dev.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(path: str) -> int:
    box = json.load(open(path))
    look = json.load(open(ROOT / "runs/phase1/devlook/look_A.json"))["picks"]
    n = len(look)
    same = diff = 0
    ex = []
    for i in range(n):
        b = box.get(str(i), "").split("\t----- bird -----\t")[0]
        f = (look.get(f"bird_dev_{i}") or "SELECT 1").replace("\t", " ").replace("\n", " ")
        if b == f:
            same += 1
        else:
            diff += 1
            ex.append(i)
    out = {"n": n, "box_predictions": len(box), "same_sql": same, "same_pct": round(100 * same / n, 2),
           "differ": diff, "differ_idx": ex}
    print(json.dumps({k: v for k, v in out.items() if k != "differ_idx"}, indent=1))
    (Path(path).parent / "compare_dev.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
