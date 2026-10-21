"""Task 5: BIRD-train DBs (minus held-out and contaminated) with no question in the wide pools (box W R1/R2
cands_train.jsonl). Counts per DB: train.json questions, plans.jsonl rows, verified plans (outcome == plan) and the
gen-eligible ones (plan, not a twin) — what `rerank_sq2.py gen --split train` would decode.

  uv run python scripts/val2/uncovered_dbs.py   # -> data/val2/uncovered_dbs.json
"""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    held = set(json.load(open(ROOT / "data/sys1/dev_dbs_bird.json")))
    cont = set(json.load(open(ROOT / "data/sys1/contaminated_dbs.json"))["dbs"])
    covered = collections.Counter()
    for role in ("r1", "r2"):
        for ln in open(ROOT / f"runs/phase1/p1w/p1w/{role}/cands_train.jsonl"):
            covered[json.loads(ln)["db_id"]] += role == "r1"
    q = collections.Counter(t["db_id"] for t in tr)
    plans, ok, elig = collections.Counter(), collections.Counter(), collections.Counter()
    for ln in open(ROOT / "data/sys1/bird_train/plans.jsonl"):
        r = json.loads(ln)
        if r.get("source", "bird") != "bird":
            continue
        plans[r["db_id"]] += 1
        if r.get("outcome") == "plan":
            ok[r["db_id"]] += 1
            if not r.get("twin_of"):
                elig[r["db_id"]] += 1
    dbs = sorted(set(q) - held - cont)
    unc = [d for d in dbs if d not in covered]
    out = {"n_train_dbs": len(set(q)), "excluded_heldout": sorted(held & set(q)), "excluded_contaminated": sorted(cont & set(q)),
           "n_candidate_dbs": len(dbs), "n_covered": len(dbs) - len(unc), "n_uncovered": len(unc),
           "uncovered": [{"db_id": d, "train_questions": q[d], "plan_rows": plans[d], "verified_plans": ok[d],
                          "gen_eligible": elig[d]} for d in sorted(unc, key=lambda d: -q[d])],
           "totals_uncovered": {"train_questions": sum(q[d] for d in unc), "verified_plans": sum(ok[d] for d in unc),
                                "gen_eligible": sum(elig[d] for d in unc)},
           "covered": {d: {"pool_questions": covered[d], "train_questions": q[d], "gen_eligible": elig[d]}
                       for d in dbs if d in covered}}
    (ROOT / "data/val2/uncovered_dbs.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "covered"}, indent=1))


if __name__ == "__main__":
    main()
