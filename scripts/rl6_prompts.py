"""Round-6 training-question pool (RL prompts and self-sampled SFT): BIRD train (BIRD's official clean list)
and Spider train, with Arctic-Text2SQL-R1's filters: drop questions whose gold query fails, returns no rows
(or only NULLs), or takes more than 5 s. Questions that overlap any sealed set (word Jaccard >= 0.5) are
dropped too. Gold is run here once and its result signature stored for the execution reward.

  .venv/bin/python scripts/rl6_prompts.py   # -> data/rl6/prompts.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from luthor.sqlw.evalkit import db_paths  # noqa: E402
from luthor.sqlw.execute import connect, signature  # noqa: E402
from prepare_plans import dedup_against  # noqa: E402

BIRD_CLEAN = Path("data/raw/bird_filtered/data/train-00000-of-00001.jsonl")
SPIDER = [Path("data/raw/spider_data/train_spider.json"), Path("data/raw/spider_data/train_others.json")]
ROOTS = ["data/raw/bird_train/train_databases", "data/raw/spider_data/database"]


def timed_gold(task):
    path, items, limit = task
    run = connect(path, limit + 1.0)
    out = []
    for it in items:
        t = time.time()
        try:
            rows = run(it["gold_sql"])
        except Exception as e:
            out.append({**it, "gold_status": "timeout" if "interrupt" in str(e).lower() else "error"})
            continue
        dt = time.time() - t
        empty = not any(v is not None for r in rows for v in r)
        status = "slow" if dt > limit else "empty" if empty else "ok"
        out.append({**it, "gold_status": status, "gold_secs": round(dt, 3), "gold_rows": len(rows),
                    "gold_sig": signature(rows)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=5.0, help="seconds; slower gold is dropped")
    ap.add_argument("--out", type=Path, default=Path("data/rl6/prompts.jsonl"))
    a = ap.parse_args()
    rows = []
    for i, r in enumerate(map(json.loads, open(BIRD_CLEAN))):
        rows.append({"id": f"bird_train_{i}", "source": "bird_train", "db_id": r["db_id"], "question": r["question"].strip(),
                     "context": (r.get("evidence") or "").strip(), "gold_sql": r["SQL"]})
    for f in SPIDER:
        for i, r in enumerate(json.load(open(f))):
            rows.append({"id": f"spider_train_{f.stem}_{i}", "source": "spider_train", "db_id": r["db_id"],
                         "question": r["question"].strip(), "context": "", "gold_sql": r["query"]})
    n0 = len(rows)
    sealed = [json.loads(l)["question"] for p in Path("data/eval").glob("sealed_*.jsonl") for l in open(p)]
    rows = dedup_against(rows, sealed)
    n_dedup = n0 - len(rows)
    paths = db_paths(ROOTS)
    by_db = collections.defaultdict(list)
    for r in rows:
        if r["db_id"] in paths:
            by_db[r["db_id"]].append(r)
    # big databases first so the pool stays busy
    tasks = sorted(((paths[db], rs, a.limit) for db, rs in by_db.items()), key=lambda t: -len(t[1]))
    with Pool(8) as pool:
        done = [x for part in pool.imap_unordered(timed_gold, tasks) for x in part]
    status = collections.Counter((r["source"], r["gold_status"]) for r in done)
    keep = [r for r in done if r["gold_status"] == "ok"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in sorted(keep, key=lambda r: r["id"])))
    print("RL6_PROMPTS", json.dumps({"read": n0, "sealed_overlap_dropped": n_dedup, "no_db": len(rows) - len(done),
                                     "status": {f"{s}:{k}": v for (s, k), v in sorted(status.items())},
                                     "kept": len(keep), "kept_by_source": dict(collections.Counter(r["source"] for r in keep)),
                                     "databases": len({r["db_id"] for r in keep})}, indent=1))


if __name__ == "__main__":
    main()
