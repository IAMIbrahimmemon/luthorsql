"""Training questions for session A (round 5): human Spider and BIRD train questions, never on the held-out
dev databases or any sealed database, no sealed-overlap questions. The writers sample answers to these;
the samples give selector/reviser training items and each question's right-count (0/8 hard cases).

  python scripts/train_questions.py --n 12000 --bird-list data/bird_train_list.txt
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mix3 import EVAL  # noqa: E402
from prepare_plans import dedup_against  # noqa: E402

PLANS = {"spider_train": Path("data/plans/spider_train/plans.jsonl"), "bird_train": Path("data/plans/bird_train/plans.jsonl")}
BIRD_FILTERED = Path("data/raw/bird_filtered/data/train-00000-of-00001.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--dev-dbs", type=Path, default=Path("data/mix2_dev_dbs.json"))
    ap.add_argument("--bird-list", type=Path, help="BIRD train databases present on the pod (one per line)")
    ap.add_argument("--out", type=Path, default=Path("data/sel/train_questions.jsonl"))
    a = ap.parse_args()
    rnd = random.Random(0)
    dev = set(json.loads(a.dev_dbs.read_text()))
    eval_q, eval_dbs = [], set()
    for d in EVAL.iterdir():
        for l in open(d / "plans.jsonl"):
            r = json.loads(l)
            eval_q.append(r["question"])
            eval_dbs.add(r["db_id"])
    bird_ok = set(a.bird_list.read_text().split()) if a.bird_list else None
    norm = lambda s: " ".join(s.lower().split())
    clean = {(r["db_id"], norm(r["question"])) for r in map(json.loads, open(BIRD_FILTERED))}
    rows = []
    for src, f in PLANS.items():
        for l in open(f):
            r = json.loads(l)
            if r.get("twin_of") or r["db_id"] in dev or r["db_id"] in eval_dbs:
                continue
            if src == "bird_train" and bird_ok is not None and r["db_id"] not in bird_ok:
                continue
            rows.append({"id": r["id"], "db_id": r["db_id"], "source": src, "question": r["question"],
                         "context": r.get("context", ""), "gold_sql": r["gold_sql"],
                         "in_slots": r.get("outcome") == "plan",
                         "gold_clean": src != "bird_train" or (r["db_id"], norm(r["question"])) in clean})
    n0 = len(rows)
    rows = dedup_against(rows, eval_q)
    rnd.shuffle(rows)
    rows = rows[:a.n]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"TRAIN_Q {len(rows)} (of {n0}, sealed-overlap dropped {n0 - len(rows) if len(rows) < a.n else 'some'}); "
          f"{dict(collections.Counter(r['source'] for r in rows))}; clean gold {sum(r['gold_clean'] for r in rows)}; "
          f"databases {len({r['db_id'] for r in rows})}", flush=True)


if __name__ == "__main__":
    main()
