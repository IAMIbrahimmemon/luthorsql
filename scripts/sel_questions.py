"""Selector training questions from BIRD-Verified (ReViSQL; expert-corrected BIRD train): corrected question,
evidence and gold SQL, gold result signature from the BIRD train databases. Their own split is kept
(train / val); val is the held-out set for selector choices. BIRD dev and the sealed sets are not touched.

  python scripts/sel_questions.py --verified data/raw/bird_verified --db-root data/raw/bird_train/train_databases \
      --out data/eval/sel
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sqlw.execute import connect, signature  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", type=Path, required=True)
    ap.add_argument("--db-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        rows, bad = [], 0
        for v in json.load(open(a.verified / f"bird-verified-{split}.json")):
            db = v["db_id"]
            try:
                res = connect(str(a.db_root / db / f"{db}.sqlite"), 30)(v["SQL"])
            except Exception:
                bad += 1
                continue
            rows.append({"id": f"bv_{split}_{v['question_id']}", "set": f"bird_verified_{split}", "db_id": db,
                         "question": v["question"], "context": v.get("evidence") or "", "gold_sql": v["SQL"],
                         "gold_sig": signature(res), "gold_empty": not any(x is not None for r in res for x in r)})
        (a.out / f"verified_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(split, len(rows), "kept;", bad, "gold failed to run", "| dbs", len({r['db_id'] for r in rows}))


if __name__ == "__main__":
    main()
