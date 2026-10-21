"""OmniSQL-7B (greedy) vs the pooled R1 ∪ R2 System 1 on ORIGINAL BIRD dev, split by grammar coverage
(paper: "level where the grammar covers the question, 21 points apart where it does not"; paper/NUMBERS.md:69).
Reads already-scored outputs only (no new dev scoring):
  runs/omnidev/omni7b_dev_greedy.ok3s.json      {id: bool}  OmniSQL-7B greedy, 3 s execution limit
  data/ens1/out/dev_sum_z_scored.jsonl          pooled System 1 records (db_id, covered flag)
  data/ens1/out/dev_sum_z_scored.stack_cons.jsonl  the pooled system's final pick per repair variant (ok flag)
Pooled pick = the V:VAL+DROP variant in stack_cons (sum_z pick + repair = the reported 59.06); covered = the record's
"covered" flag.

  .venv/bin/python scripts/analysis/omni_coverage_split.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luthor.stats import cluster_signflip  # noqa: E402

OMNI = ROOT / "runs/omnidev/omni7b_dev_greedy.ok3s.json"
POOL = ROOT / "data/ens1/out/dev_sum_z_scored.jsonl"               # candidates + "covered" flag + db_id
STACK = ROOT / "data/ens1/out/dev_sum_z_scored.stack_cons.jsonl"    # final pick per repair variant: res[variant] = [ok, sql]
VARIANT = "V:VAL+DROP"  # the reported pooled system (59.06): sum_z pick after the VAL+DROP repair


def main():
    omni = {k: bool(v) for k, v in json.load(open(OMNI)).items()}
    recs = {r["id"]: r for r in map(json.loads, open(POOL))}
    ids = sorted(set(omni) & set(recs))
    db = {i: recs[i]["db_id"] for i in ids}
    stack = {r["id"]: r for r in map(json.loads, open(STACK))}
    sys1 = {i: bool(stack[i]["res"][VARIANT][0]) for i in ids}
    print(f"questions {len(ids)} (omni {len(omni)}, pooled {len(recs)})")
    for name, sel in (("all", ids), ("covered", [i for i in ids if recs[i].get("covered")]),
                      ("uncovered", [i for i in ids if not recs[i].get("covered")])):
        a, b = {i: sys1[i] for i in sel}, {i: omni[i] for i in sel}
        c = cluster_signflip(a, b, {i: db[i] for i in sel})
        print(f"{name:9s} n={len(sel):4d}  pooled System 1 {100 * sum(a.values()) / len(sel):6.2f}  "
              f"OmniSQL-7B {100 * sum(b.values()) / len(sel):6.2f}  omni - pooled {c['delta']:+.2f}  DB p={c['p']:.3f}")


if __name__ == "__main__":
    main()
