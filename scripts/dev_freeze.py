"""Freeze the held-out dev list (round 5): every human question on the 20 held-out training databases.

Unlike the round-3/4 dev files, nothing is dropped for being outside the slot grammar, and the gold is the
original human SQL (not the plan-compiled copy). Generated value-swapped twins are left out, as before.
Each gold query is run once with a long time limit; questions whose gold fails stay in the list with a
status, so every report can state its denominator.

  python scripts/dev_freeze.py            # -> data/eval/dev_frozen.jsonl + data/eval/dev_frozen_summary.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sqlw.evalkit import db_paths, run_gold  # noqa: E402

PLANS = ["data/plans/spider_train/plans.jsonl", "data/plans/bird_train/plans.jsonl"]
BIRD_FILTERED = Path("data/raw/bird_filtered/data/train-00000-of-00001.jsonl")  # birdsql/bird23-train-filtered


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-dbs", type=Path, default=Path("data/mix2_dev_dbs.json"))
    ap.add_argument("--db-roots", nargs="+", default=["data/raw/spider_data/database", "data/raw/bird_train/train_databases"])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=Path("data/eval/dev_frozen.jsonl"))
    a = ap.parse_args()
    dev = set(json.loads(a.dev_dbs.read_text()))
    paths = db_paths(a.db_roots)
    rows = []
    for f in PLANS:
        for line in open(f):
            r = json.loads(line)
            if r["db_id"] in dev and not r.get("twin_of"):
                rows.append({"id": r["id"], "source": r["source"], "db_id": r["db_id"], "question": r["question"],
                             "context": r.get("context", ""), "gold_sql": r["gold_sql"],
                             "in_slots": r.get("outcome") == "plan", "slot_reason": r.get("reason", "")})
    missing = sorted({r["db_id"] for r in rows if r["db_id"] not in paths})
    if missing:
        sys.exit(f"databases not found: {missing}")
    by_db = collections.defaultdict(list)
    for r in rows:
        by_db[r["db_id"]].append(r)
    tasks = [(paths[db], [r["gold_sql"] for r in rs], a.seconds) for db, rs in by_db.items()]
    with Pool(8) as pool:
        for rs, res in zip(by_db.values(), pool.map(run_gold, tasks)):
            for r, (status, n_rows, empty, sig) in zip(rs, res):
                r.update(gold_status=status, gold_rows=n_rows, gold_empty=empty, gold_sig=sig)
    # BIRD's official quality filter (6,601 of 9,428 train questions kept): BIRD questions it removed are
    # flagged gold_clean=False. Spider questions have no such audit and stay True.
    clean = {(r["db_id"], _norm(r["question"])) for r in map(json.loads, open(BIRD_FILTERED))}
    for r in rows:
        r["gold_clean"] = r["source"] != "bird" or (r["db_id"], _norm(r["question"])) in clean
    rows.sort(key=lambda r: r["id"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summ = {"questions": len(rows), "databases": len(by_db),
            "by_source": dict(collections.Counter(r["source"] for r in rows)),
            "in_slots": sum(r["in_slots"] for r in rows),
            "gold_status": dict(collections.Counter(r["gold_status"] for r in rows)),
            "gold_clean": sum(r["gold_clean"] for r in rows),
            "gold_empty": sum(r["gold_empty"] for r in rows if r["gold_status"] == "ok"),
            "gold_seconds": a.seconds}
    (a.out.parent / "dev_frozen_summary.json").write_text(json.dumps(summ, indent=1))
    print("FROZEN", json.dumps(summ))


if __name__ == "__main__":
    main()
