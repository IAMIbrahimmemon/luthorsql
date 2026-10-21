"""Blame each wrong System-1 BIRD-dev answer on the first plan slot that differs from gold.

  python scripts/analysis/sys1_errors.py runs/sys1/pilot2/gate_predictions.jsonl data/plans_v3/dev/bird_dev
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from error_breakdown import first_diff  # noqa: E402
from luthor.sq.convert import convert  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402


def fns(p):
    cs = [x["col"] for x in p["projections"]] + [f["col"] for f in p.get("filters") or []]
    return sorted((c or {}).get("fn") or "" for c in cs)


def main():
    preds, root = sys.argv[1], Path(sys.argv[2])
    gold = {r["id"]: r for r in map(json.loads, open(root / "plans.jsonl")) if r.get("plan")}
    graphs, tally, by_db, ex = {}, collections.Counter(), collections.defaultdict(collections.Counter), collections.defaultdict(list)
    for r in map(json.loads, open(preds)):
        if r["correct"] or r["id"] not in gold:
            continue
        db = r["db_id"]
        if db not in graphs:
            graphs[db] = SchemaCard.load(root / "cards" / f"{db}.json").graph()
        try:
            pp = convert(r["pred_sql"], graphs[db]).plan.to_dict()
        except Exception as e:  # noqa: BLE001
            k = f"pred does not convert ({type(e).__name__})"
        else:
            g = gold[r["id"]]["plan"]
            k = first_diff(g, pp)
            if k.startswith("same plan") and fns(g) != fns(pp):
                k = "expression template"
        tally[k] += 1
        by_db[db][k] += 1
        ex[k].append((r["id"], r["question"], gold[r["id"]].get("context", ""), r["gold_sql"], r["pred_sql"]))
    n = sum(tally.values())
    print(f"{n} wrong covered answers")
    for k, v in tally.most_common():
        print(f"  {v:4d} {v / n:5.1%}  {k}")
    print()
    for db, c in sorted(by_db.items(), key=lambda x: -sum(x[1].values())):
        print(f"{db:26s} " + ", ".join(f"{k} {v}" for k, v in c.most_common(3)))
    out = Path(preds).with_name("errors_by_slot.json")
    json.dump(ex, open(out, "w"), indent=1)
    print("examples ->", out)


if __name__ == "__main__":
    main()
