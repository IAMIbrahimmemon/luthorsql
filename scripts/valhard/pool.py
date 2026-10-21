"""VALHARD: the held-out pool the box scores = ONE generator's own k=32 candidates (System 1).

  .venv/bin/python scripts/valhard/pool.py --src runs/phase1/p1w/p1w/heldout_union.jsonl --keep-src r1 both \
      --out data/valhard/r1/heldout_pool.jsonl                              # R1: its share of the R1 u R2 wide pool
  .venv/bin/python scripts/valhard/pool.py --src runs/phase1/v4desc/r1/rr/wide/cands_dev_k32.jsonl \
      --out data/valhard/v4desc/heldout_pool.jsonl                          # v4desc: its own WIDE_K=32 list

Held-out DBs only (DECISION + RESERVED, as val2 pushed); RESERVED rows are carried blind (no field is read or printed
here; scripts/analysis/valhard_eval.py drops them at load). Prints only counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--keep-src", nargs="*", default=None, help="keep candidates whose 'src' is one of these")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    held = set(json.load(open(ROOT / "data/sys1/dev_dbs_bird.json")))
    n = nc = 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for line in open(a.src):
            r = json.loads(line)
            assert r["db_id"] in held, r["db_id"]  # never a BIRD dev / training question
            if a.keep_src is not None:
                r["cands"] = [c for c in r["cands"] if c.get("src") in a.keep_src]
            for c in r["cands"]:  # scores of other validators never travel with the pool
                c.pop("rr", None)
            n += 1
            nc += len(r["cands"])
            f.write(json.dumps(r) + "\n")
    print(f"POOL {n} questions, {nc} candidates -> {a.out}")


if __name__ == "__main__":
    main()
