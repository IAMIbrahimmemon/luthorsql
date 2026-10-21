"""2a data-slope arms (runs/phase1/{teacher,verified}/PREREG.md) vs the R3a seed control on DECISION.

  python scripts/analysis/slope_eval.py teacher|verified
Each run: its own k=8 list scored by its own validator (rr/dev_scored.jsonl), as in the seed control (§7b).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_eval import IDS, ROOT, compare, load, single_hits  # noqa: E402


def first_pick(scored):
    return {i: bool(scored.get(i)) and bool(scored[i]["cands"]) and bool(scored[i]["cands"][0].get("ok")) for i in IDS}


def main(arm):
    ctl = load(ROOT / "runs/phase1/seed/r1/rr/dev_scored.jsonl")
    new = load(ROOT / f"runs/phase1/{arm}/r1/rr/dev_scored.jsonl")
    b, ob = single_hits(ctl)
    n, on = single_hits(new)
    print("PRIMARY")
    c = compare("R3a control (own k=8, own validator)", b, f"{arm} (own k=8, own validator)", n, ob, on)
    print(f"  -> {'PASS' if c['delta'] >= 2.0 and c['p'] < 0.05 else ('KILL' if c['delta'] < 1.0 else 'NO PASS')}")
    print("secondary (info)")
    compare("R3a first pick", first_pick(ctl), f"{arm} first pick", first_pick(new), ob, on)


if __name__ == "__main__":
    main(sys.argv[1])
