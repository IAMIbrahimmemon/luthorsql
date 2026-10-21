"""R3b pure seed control (runs/phase1/seedfull/PREREG.md): run-to-run gap on DECISION.

  python scripts/analysis/seed_eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_eval import ROOT, compare, load, single_hits  # noqa: E402


def main():
    r1 = single_hits(load(ROOT / "runs/r1r2/r1/rr/dev_scored.jsonl"))
    r3a = single_hits(load(ROOT / "runs/phase1/seed/r1/rr/dev_scored.jsonl"))
    r3b = single_hits(load(ROOT / "runs/phase1/seedfull/r1/rr/dev_scored.jsonl"))
    print("SEED GAP (only the seed differs)")
    compare("R1 (seed 0)", r1[0], "R3b (seed 1, all DBs)", r3b[0], r1[1], r3b[1])
    print("DATA + ORDER GAP (formula_1 / soccer_1 kept vs dropped; same seed, so batch order also diverges)")
    compare("R3a (seed 1, DBs dropped)", r3a[0], "R3b (seed 1, all DBs)", r3b[0], r3a[1], r3b[1])


if __name__ == "__main__":
    main()
