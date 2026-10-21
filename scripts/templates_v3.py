"""Grammar v3: collect the expression and compose templates used by verified training plans, so the
model's template list (a closed set it classifies over) comes from training data only.

  python scripts/templates_v3.py data/plans_v3/train/bird_train/plans.jsonl data/plans_v3/train/spider_train/plans.jsonl \
      --min-count 2 --out data/plans_v3/templates.json

Then measure dev coverage with only those templates:
  python scripts/prepare_plans.py --format bird --root data/bench/dev_20240627 --split dev \
      --out data/plans_v3/dev --allowed-templates data/plans_v3/templates.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sq.plan import COMPOSE, Plan  # noqa: E402


def numbers(plan: Plan) -> list:
    out = [v for c in plan.columns() for v in c.nums] + list(plan.compose_nums)
    for f in plan.filters:
        if f.sub is not None:
            out += numbers(f.sub)
    if plan.other is not None:
        out += numbers(plan.other)
    return out


def templates(plan: Plan) -> list[str]:
    out = [c.fn for c in plan.columns() if c.fn]
    if plan.compose not in COMPOSE:
        out.append(plan.compose)
    for f in plan.filters:
        if f.sub is not None:
            out += templates(f.sub)
    if plan.other is not None:
        out += templates(plan.other)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plans", nargs="+", type=Path)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    n = collections.Counter()
    nums = collections.Counter()
    rows = 0
    for f in a.plans:
        for line in open(f):
            r = json.loads(line)
            if r.get("outcome") != "plan" or r.get("twin_of"):
                continue
            rows += 1
            pl = Plan.from_dict(r["plan"])
            for t in set(templates(pl)):
                n[t] += 1
            for v in set(numbers(pl)):
                nums[v] += 1
    keep = sorted(t for t, k in n.items() if k >= a.min_count)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(keep, indent=0))
    num_file = a.out.with_name(a.out.stem + "_nums.json")
    num_file.write_text(json.dumps(sorted(nums, key=float)))
    print(f"NUMS {len(nums)}: {sorted(nums, key=float)[:60]}")
    print(f"TEMPLATES {len(keep)} kept (count >= {a.min_count}) of {len(n)} seen in {rows} verified plans; "
          f"plans using a template: {sum(n.values())}")
    for t, k in n.most_common(40):
        print(f"{k:6d}  {t}")


if __name__ == "__main__":
    main()
