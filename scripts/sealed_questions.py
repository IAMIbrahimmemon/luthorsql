"""Sealed question files for the one-time final scoring (round 5): every question of the official files of Spider dev, Spider test, Spider-Syn dev and BIRD dev, with the original gold SQL and evidence.
Gold is run once (60 s limit) and its result signature stored, as for the frozen dev list. Scored once per
final system; record each scoring in docs/EVAL_LEDGER.md. Never used for training or tuning.

  python scripts/sealed_questions.py   # -> data/eval/sealed_<set>.jsonl
"""
from __future__ import annotations

import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sqlw.evalkit import db_paths, run_gold  # noqa: E402

SETS = {  # official files: (path, database root, question key, gold key, evidence key)
    "spider_dev": ("data/raw/spider_data/dev.json", "data/raw/spider_data/database", "question", "query", None),
    "spider_test": ("data/raw/spider_data/test.json", "data/raw/spider_data/test_database", "question", "query", None),
    "spider_syn_dev": ("data/raw/spider_syn/dev.json", "data/raw/spider_data/database", "SpiderSynQuestion", "query", None),
    "bird_dev": ("data/bench/dev_20240627/dev.json", "data/bench/dev_20240627/dev_databases", "question", "SQL", "evidence"),
}


def main():
    for name, (f, root, qk, gk, ek) in SETS.items():
        paths = db_paths([root])
        rows = [{"id": f"{name}_{i}", "set": name, "source": name, "db_id": r["db_id"], "question": r[qk].strip(),
                 "context": (r.get(ek) or "") if ek else "", "gold_sql": r[gk]} for i, r in enumerate(json.load(open(f)))]
        missing = sorted({r["db_id"] for r in rows if r["db_id"] not in paths})
        by_db = collections.defaultdict(list)
        for r in rows:
            if r["db_id"] in paths:
                by_db[r["db_id"]].append(r)
        tasks = [(paths[db], [r["gold_sql"] for r in rs], 60.0) for db, rs in by_db.items()]
        with Pool(8) as pool:
            for rs, res in zip(by_db.values(), pool.map(run_gold, tasks)):
                for r, (status, n, empty, sig) in zip(rs, res):
                    r.update(gold_status=status, gold_rows=n, gold_empty=empty, gold_sig=sig)
        Path(f"data/eval/sealed_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(name, len(rows), "missing dbs", missing, dict(collections.Counter(r.get("gold_status", "no_db") for r in rows)), flush=True)


if __name__ == "__main__":
    main()
