"""Round-6 dev list: BIRD dev (1,534 questions, 11 databases), now a development set (owner, 2026-10-04;
see docs/EVAL_LEDGER.md). Starts from the round-5 sealed file (gold already run with 60 s), re-runs any gold
that failed with a longer limit, and adds BIRD's difficulty label.

  python scripts/dev6_bird.py   # -> data/eval/dev6_bird.jsonl (original 2024-06-27 gold)
  python scripts/dev6_bird.py --corrected   # -> data/eval/dev6c_bird.jsonl (BIRD's 2025-11-06 corrected dev:
                                            #    450 SQL, 378 evidence and 182 questions revised; same databases)
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sqlw.evalkit import db_paths, run_gold  # noqa: E402

SRC = Path("data/eval/sealed_bird_dev.jsonl")
OFFICIAL = Path("data/bench/dev_20240627/dev.json")
ROOT = "data/bench/dev_20240627/dev_databases"
OUT = Path("data/eval/dev6_bird.jsonl")
CORRECTED = Path("data/raw/bird_dev_20251106/data/dev_20251106-00000-of-00001.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrected", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(SRC)]
    official = json.load(open(OFFICIAL))
    if a.corrected:
        new = {r["question_id"]: r for r in json.load(open(CORRECTED))}
        for r, o in zip(rows, official):
            n = new[o["question_id"]]
            r.update(question=n["question"].strip(), context=(n.get("evidence") or "").strip(), gold_sql=n["SQL"],
                     changed={"question": n["question"].strip() != o["question"].strip(),
                              "evidence": (n.get("evidence") or "").strip() != (o.get("evidence") or "").strip(),
                              "sql": " ".join(n["SQL"].split()) != " ".join(o["SQL"].split())})
            if r["changed"]["sql"]:
                r["gold_status"] = "rerun"
        official = [{"question": r["question"], "difficulty": o.get("difficulty", "")} for r, o in zip(rows, official)]
        global OUT
        OUT = Path("data/eval/dev6c_bird.jsonl")
    assert len(official) == len(rows)
    paths = db_paths([ROOT])
    for r, o in zip(rows, official):
        assert r["question"] == o["question"].strip()
        r.update(set="bird_dev", source="bird_dev", difficulty=o.get("difficulty", ""))
        if r["gold_status"] != "ok":
            (status, n, empty, sig), = run_gold((paths[r["db_id"]], [r["gold_sql"]], 900.0))
            print("rerun", r["id"], r["gold_status"], "->", status, n, flush=True)
            r.update(gold_status=status, gold_rows=n, gold_empty=empty, gold_sig=sig)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("DEV6", len(rows), dict(collections.Counter(r["gold_status"] for r in rows)),
          dict(collections.Counter(r["difficulty"] for r in rows)), "empty gold", sum(r["gold_empty"] for r in rows))


if __name__ == "__main__":
    main()
