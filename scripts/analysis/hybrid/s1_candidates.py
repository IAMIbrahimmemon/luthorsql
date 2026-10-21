"""System 1's candidate block for System-1 + LLM runs (DECISION only, label-free): the frozen System 1 (R1 k=32 +
VALHARD) live candidates grouped by execution result, best first by VALHARD, one representative per group (top 8),
with a short result preview and n_of_32. No ok / gold field is written.

  .venv/bin/python scripts/analysis/hybrid/s1_candidates.py --out runs/analysis/hybrid/s1_cands_decision.jsonl
"""
import argparse
import json
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    ids = json.load(open(ROOT / "data/sys1/heldout_v2/decision.json"))["ids"]
    idset = set(ids)
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    have = {}
    for ln in open(SRC):
        r = json.loads(ln)
        if r["id"] not in idset:
            continue
        c = [x for x in r["cands"] if x.get("sql") and x.get("src") in ("r1", "both", None)]
        lv = [x for x in c if x.get("runs") and x.get("nonnull")] or c
        groups = {}
        for x in sorted(lv, key=lambda x: -x.get("rr", -1e9)):
            s = x.get("sig") or x["sql"]
            groups.setdefault(s, []).append(x)
        reps = [{"label": string.ascii_uppercase[k], "sql": g[0]["sql"], "result": g[0].get("res", ""),
                 "n_of_32": len(g)} for k, g in enumerate(list(groups.values())[:a.top])]
        have[r["id"]] = reps
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for i in ids:
            t = tr[int(i.rsplit("_", 1)[1])]
            f.write(json.dumps({"question_id": i, "db_id": t["db_id"], "question": t["question"],
                                "evidence": t.get("evidence") or "", "candidates": have.get(i, [])}) + "\n")
    print(f"wrote {a.out}: {len(ids)} questions, {sum(1 for i in ids if have.get(i))} with candidates")


if __name__ == "__main__":
    main()
