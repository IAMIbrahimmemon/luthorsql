"""Cascade cost saving with System 1 run on hardware the user already owns (its marginal cost set to zero).

The registered scorers add System 1's rented-GPU cost (H100 at $2.73/h) to every question of the cascade and of B1.
Run locally, System 1's marginal cost is electricity, which we treat as zero. This recomputes every test's saving
from the same logged agent usage with that one change: cost(arm) = the arm's agent cost only. Database bootstrap
(10,000 resamples, seed 20261023) for the intervals.

  .venv/bin/python experiments/cost_confirmation/local_s1.py <session dir with scratchpad/ and tasks/>
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external_cascade"))
import score  # noqa: E402
import score_ext  # noqa: E402

S = Path(sys.argv[1])
TESTS = {"reserved_run1": ["s2res"], "reserved_run2": ["s2res_r2"], "reserved_run3": ["s2res_r3"],
         "reserved_pooled": ["s2res", "s2res_r2", "s2res_r3"], "external": ["s2ext"], "fresh": ["s2fresh"]}


def per_db(boxes):
    pat = re.compile(r"scratchpad/(s2\w+)/chunks/(B0|B1|C)/([A-Za-z0-9_]+_\d+)\.jsonl")
    cost = defaultdict(lambda: defaultdict(float))
    for f in (S / "tasks").glob("*.output"):
        m = pat.search(open(f).read(20000))
        if not m or m.group(1) not in boxes:
            continue
        cost[m.group(2)][m.group(3).rsplit("_", 1)[0]] += score.dollars(score_ext._one(str(f)))
    return cost


def saving(cost, base, dbs):
    return 1 - sum(cost["C"][d] for d in dbs) / sum(cost[base][d] for d in dbs)


def main():
    out = {}
    for test, boxes in TESTS.items():
        cost = per_db(boxes)
        dbs = sorted(cost["B0"])
        rng = random.Random(20261023)
        row = {"databases": len(dbs)}
        for base in ("B0", "B1"):
            if not cost.get(base):
                continue
            boot = sorted(saving(cost, base, [rng.choice(dbs) for _ in dbs]) for _ in range(10000))
            row[f"saving_vs_{base}"] = {"point": round(saving(cost, base, dbs), 4),
                                        "ci95": [round(boot[249], 4), round(boot[9749], 4)]}
        out[test] = row
    print(json.dumps(out, indent=1))
    Path(__file__).with_name("local_s1.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
