"""BIRD-train column descriptions: how much schema text they add, which ones say more than the column name, and how
many training rows a capped variant would push over the size that fit in R3a (the OOM-skip cause found in
v4desc_regression.py section H). Mac CPU, no model. Training data only (BIRD-train v4 plans; held-out DBs skipped).

  uv run --offline python scripts/analysis/desc_budget.py

Variants (luthor.sq.bird_desc.apply modes + sq2 serialize.schema_text, the renderer the generator sees):
  none       no BIRD descriptions (R3a's cards)
  full120    v4desc: every empty column description filled, cap 120 characters (bird_desc.DESC_CAP)
  info60     only descriptions that add a word beyond the column name / expanded name (restatements such as
             "unique order number" for orderNumber dropped), cap 60
  cryptic60  info60, and only for columns whose name is cryptic (no dictionary-like word of >= 4 letters, or <= 4
             characters, or mostly digits / abbreviations), cap 60
Per training row the shown schema is what train_tables shows at --budget-chars 9000: the whole schema when it fits,
else the gold tables (plus random fillers up to the budget; the gold tables alone when they exceed it).
"Over" counts rows whose shown text is longer than the longest shown text of any no-description row on a database
R3a trained without single-question OOM skips (works_cycles / movie_3 excluded) -- a size proxy for OOM risk.
Writes runs/analysis/grammar_v4_fix/desc_budget.txt.
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

os.environ["LUTHOR_GRAMMAR"] = "v4"
os.environ.pop("LUTHOR_BIRD_DESC", None)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luthor.sq import bird_desc  # noqa: E402
from luthor.sq.plan import Plan  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sq2.serialize import SCHEMA_HEAD, schema_text  # noqa: E402

PLANS = ROOT / "data/plans_v4/train/bird_train"
OUT = ROOT / "runs/analysis/grammar_v4_fix"
BUDGET = 9000
HELD = set(json.load(open(ROOT / "data/sys1/dev_dbs_bird.json")))
LOG: list[str] = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def card_variant(db: str, variant: str, D) -> SchemaCard:
    c = SchemaCard.load(PLANS / "cards" / f"{db}.json")
    if variant != "none":
        mode = {"full120": "full", "info60": "info", "cryptic60": "cryptic"}[variant]
        bird_desc.apply(c, D, cap=120 if variant == "full120" else 60, mode=mode)
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = bird_desc.load_dir(bird_desc.DEFAULT_DIR)
    rows = collections.defaultdict(list)
    for line in open(PLANS / "plans.jsonl"):
        r = json.loads(line)
        if r.get("outcome") == "plan" and r["db_id"] not in HELD:
            rows[r["db_id"]].append(set(Plan.from_dict(r["plan"]).tables()))
    variants = ("none", "full120", "info60", "cryptic60")
    shown = {v: {} for v in variants}
    stats = {v: collections.Counter() for v in variants}
    for db in sorted(rows):
        for v in variants:
            c = card_variant(db, v, D)
            full = len(schema_text(c).text)
            stats[v]["chars"] += full
            stats[v]["described"] += sum(bool(x.description) for t in c.tables for x in t.columns)
            stats[v]["cols"] += sum(len(t.columns) for t in c.tables)
            per_t = {t.name: len(schema_text(c, [t.name]).text) - len(SCHEMA_HEAD) for t in c.tables}
            sizes = []
            for gold in rows[db]:
                if full <= BUDGET:
                    sizes.append(full)
                else:
                    g = len(SCHEMA_HEAD) + sum(per_t.get(t, 0) for t in gold)
                    sizes.append(max(min(full, BUDGET), g))
            shown[v][db] = sizes
    ok_dbs = [d for d in rows if d not in ("works_cycles", "movie_3")]
    T = max(max(shown["none"][d]) for d in ok_dbs)
    say(f"BIRD-train training DBs: {len(rows)} (held-out DBs excluded); rows {sum(len(v) for v in rows.values())}")
    say(f"size threshold T = longest no-description shown schema on DBs R3a trained without OOM skips: {T} chars")
    say("variant    schema chars (sum over DBs)  described cols / cols   rows over T   (top DBs over T)")
    for v in variants:
        over = {d: sum(s > T for s in shown[v][d]) for d in rows}
        top = sorted(((n, d) for d, n in over.items() if n), reverse=True)[:6]
        say(f"{v:10s} {stats[v]['chars']:12d} ({100 * stats[v]['chars'] / stats['none']['chars'] - 100:+5.1f}%)"
            f"      {stats[v]['described']:5d} / {stats[v]['cols']}   {sum(over.values()):9d}   "
            + ", ".join(f"{d} {n}" for n, d in top))
    say("\nper OOM-skipped DB (v4desc skips: legislator 440, professional_basketball 416, movie_3 224 (R3a 16), "
        "college_completion 152): rows, max shown chars per variant")
    for d in ("legislator", "professional_basketball", "movie_3", "college_completion", "works_cycles"):
        if d in rows:
            say(f"  {d:24s} rows {len(rows[d]):4d}  " + "  ".join(f"{v} max {max(shown[v][d])} over {sum(s > T for s in shown[v][d])}"
                                                           for v in variants))
    (OUT / "desc_budget.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
