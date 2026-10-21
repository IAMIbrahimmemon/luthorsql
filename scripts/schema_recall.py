"""How often does the writer's prompt keep every table the gold SQL needs? (round 5, step 1)

Gold SQL is used for diagnosis only. Three prompt settings on the frozen dev list:
  overlap   - tables ranked by word overlap with the question and evidence (serve-time default)
  with_slot - plus the tables of the slot model's top-3 candidates as `must` (round-4 dev setting)
  oracle    - must = gold tables (upper bound: always 100%; reported as a sanity check)
Also tries larger character budgets to see how much budget alone recovers.

  python scripts/schema_recall.py
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sqlw.prompt import BUDGET, schema_ddl, used_tables  # noqa: E402

CARD_DIRS = [Path("data/plans/spider_train/cards"), Path("data/plans/bird_train/cards")]


def kept_tables(card: SchemaCard, ddl: str) -> set[str]:
    return {t.name for t in card.tables if f"CREATE TABLE {t.name} (" in ddl or f'CREATE TABLE "{t.name}" (' in ddl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", type=Path, default=Path("data/eval/dev_frozen.jsonl"))
    ap.add_argument("--slot-cands", type=Path, default=Path("runs/sq3_pilot/cands_dev32.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/schema_recall.json"))
    a = ap.parse_args()
    cards = {}
    for d in CARD_DIRS:
        for p in d.glob("*.json"):
            cards.setdefault(p.stem, p)
    rows = [json.loads(l) for l in open(a.frozen)]
    slot_must = {}
    if a.slot_cands.exists():
        for l in open(a.slot_cands):
            r = json.loads(l)
            sqls = [c["sql"] for c in r["cands"][:3] if c.get("sql")]
            slot_must[r["id"]] = sqls
    cache, res = {}, collections.defaultdict(lambda: collections.Counter())
    misses = []
    for r in rows:
        if r["db_id"] not in cache:
            cache[r["db_id"]] = SchemaCard.load(cards[r["db_id"]])
        card = cache[r["db_id"]]
        gold = set(used_tables(card, r["gold_sql"]))
        fits = len(schema_ddl(card, r["question"], r["context"], budget=10**9)) <= BUDGET
        for setting, budget in [("overlap", BUDGET), ("overlap_12k", 12000), ("overlap_16k", 16000),
                                ("with_slot", BUDGET)]:
            must = None
            if setting == "with_slot":
                must = sorted({t for s in slot_must.get(r["id"], []) for t in used_tables(card, s)})
            ddl = schema_ddl(card, r["question"], r["context"], must=must, budget=budget)
            ok = gold <= kept_tables(card, ddl)
            for g in ("all", r["source"], "fits" if fits else "cut"):
                res[(setting, g)]["n"] += 1
                res[(setting, g)]["ok"] += ok
            if setting == "overlap" and not ok:
                misses.append({"id": r["id"], "db_id": r["db_id"], "missing": sorted(gold - kept_tables(card, ddl))})
    report = {f"{s}/{g}": {"n": c["n"], "recall": round(100 * c["ok"] / c["n"], 1)} for (s, g), c in sorted(res.items())}
    report["missing_by_db"] = dict(collections.Counter(m["db_id"] for m in misses).most_common())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    for k, v in report.items():
        print(k, v)


if __name__ == "__main__":
    main()
