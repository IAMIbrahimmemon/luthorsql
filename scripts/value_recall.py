"""Do value hints supply the string values gold SQL needs? (round 5, step 1; gold used for diagnosis only)

For each string literal in a frozen dev question's gold SQL (LIKE patterns stripped of %):
  in_text  - appears with the same spelling in the question or evidence
  in_hints - appears in the value hints the writer prompt would show
  missing  - neither

  python scripts/value_recall.py
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sq.candidates import ValueIndex  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sqlw.evalkit import db_paths  # noqa: E402
from luthor.sqlw.prompt import value_hints  # noqa: E402

ROOTS = ["data/raw/spider_data/database", "data/raw/bird_train/train_databases"]
CARD_DIRS = [Path("data/plans/spider_train/cards"), Path("data/plans/bird_train/cards")]


def literals(sql: str) -> list[str]:
    out = []
    for m in re.finditer(r"'((?:[^']|'')*)'", sql):
        v = m.group(1).replace("''", "'").strip("%")
        if v and not re.fullmatch(r"[\d.\-/: ]+", v):  # numbers and dates are not looked up
            out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", type=Path, default=Path("data/eval/dev_frozen.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/value_recall.json"))
    ap.add_argument("--roots", nargs="+", default=ROOTS)
    ap.add_argument("--cards", nargs="+", type=Path, default=CARD_DIRS)
    a = ap.parse_args()
    cards = {}
    for d in a.cards:
        for p in d.glob("*.json"):
            cards.setdefault(p.stem, p)
    paths = db_paths(a.roots)
    rows = [json.loads(l) for l in open(a.frozen)]
    by_db = collections.defaultdict(list)
    for r in rows:
        by_db[r["db_id"]].append(r)
    cnt = collections.defaultdict(collections.Counter)
    n_hints, per_q = [], {}
    for db, rs in by_db.items():
        card = SchemaCard.load(cards[db])
        con = sqlite3.connect(f"file:{paths[db]}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode("utf-8", "replace")
        vi = ValueIndex.build(con, card, skip_card_values=False)
        con.close()
        for r in rs:
            hints = value_hints(vi, r["question"], r["context"])
            n_hints.append(len(hints))
            text = r["question"] + " " + r["context"]
            hint_vals = {h.split(" = ", 1)[1][1:-1].replace("''", "'") for h in hints}
            kinds = []
            for v in literals(r["gold_sql"]):
                k = "in_text" if v in text else "in_hints" if v in hint_vals else "missing"
                kinds.append(k)
                for g in ("all", r["source"]):
                    cnt[g][k] += 1
            per_q[r["id"]] = {"kinds": kinds, "hints": hints, "missing": [v for v, k in zip(literals(r["gold_sql"]), kinds) if k == "missing"]}
    report = {g: {**dict(c), "total": sum(c.values()),
                  "covered_pct": round(100 * (c["in_text"] + c["in_hints"]) / max(sum(c.values()), 1), 1),
                  "hints_add_pct": round(100 * c["in_hints"] / max(sum(c.values()), 1), 1)} for g, c in cnt.items()}
    report["mean_hints_per_question"] = round(sum(n_hints) / len(n_hints), 2)
    report["questions_with_hints"] = sum(n > 0 for n in n_hints)
    a.out.write_text(json.dumps(report, indent=1))
    (a.out.parent / (a.out.stem + "_per_q.json")).write_text(json.dumps(per_q))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
