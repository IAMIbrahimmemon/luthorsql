"""07 backbone_scale: token lengths of validator inputs (schema prefix + candidate rows) for memory/time estimates.

Reads the VALHARD r1 train lists (BIRD-train DBs, no held-out rows) and the DECISION ids of the held-out pool, tokenises
with the local OmniSQL-7B tokenizer (Qwen2.5 vocabulary, same for Qwen2.5-Coder 7B/14B/32B and OmniSQL 14B/32B).
Prints only aggregate token statistics (no scores, no labels).

  .venv/bin/python scripts/analysis/validator/07_token_stats.py > runs/analysis/validator/07_token_stats.out
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tokenizers import Tokenizer  # noqa: E402

from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sq2.rerank import candidate_text  # noqa: E402
from luthor.sq2.serialize import schema_text  # noqa: E402

BUDGET = 9000  # rerank_sq2.py --budget-chars default
TOK = Tokenizer.from_file(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--seeklhy--OmniSQL-7B/snapshots/*/tokenizer.json"))[0])
CARDS = {p.stem: p for p in (ROOT / "data/sys1/bird_train/cards").glob("*.json")}


def tables_for(card, sqls):  # copy of rerank_sq2.tables_for (no import of the GPU script)
    from luthor.sq2.serialize import SCHEMA_HEAD
    if len(schema_text(card).text) <= BUDGET:
        return None
    used = [t.name for t in card.tables if any(f'"{t.name}"' in s for s in sqls)]
    size = {t.name: len(schema_text(card, [t.name]).text) - len(SCHEMA_HEAD) for t in card.tables}
    keep, total = list(used), len(SCHEMA_HEAD) + sum(size[t] for t in used)
    for t in card.tables:
        if t.name not in keep and total + size[t.name] <= BUDGET:
            keep.append(t.name)
            total += size[t.name]
    names = [t.name for t in card.tables]
    return sorted(keep, key=names.index)


def ntok(s: str) -> int:
    return len(TOK.encode(s, add_special_tokens=False).ids)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def stats(name, recs, with_result):
    cache, pre, row, ncand, tot = {}, [], [], [], []
    for r in recs:
        db = r["db_id"]
        if db not in CARDS:
            continue
        card = cache.setdefault(db, SchemaCard.load(CARDS[db]))
        cands = [c for c in r["cands"] if c.get("sql")]
        tabs = tables_for(card, [c["sql"] for c in cands])
        p = ntok(schema_text(card, tabs).text)
        rows = [min(384, ntok(candidate_text(r["question"], r.get("context", ""), c["sql"],
                                            c.get("res") if with_result else None))) for c in cands]
        pre.append(p)
        row += rows
        ncand.append(len(cands))
        tot.append(p + sum(rows))
    print(f"{name}: lists {len(pre)}, candidates {len(row)} ({st.mean(ncand):.1f}/list)")
    print(f"  schema prefix tokens: mean {st.mean(pre):.0f} p50 {pct(pre, .5)} p90 {pct(pre, .9)} max {max(pre)}")
    print(f"  candidate row tokens: mean {st.mean(row):.0f} p50 {pct(row, .5)} p90 {pct(row, .9)} max {max(row)}")
    print(f"  tokens per list (prefix once + rows): mean {st.mean(tot):.0f} p90 {pct(tot, .9)} max {max(tot)}")
    print(f"  total tokens with prefix sharing: {sum(tot) / 1e6:.2f}M; "
          f"without sharing (prefix per row): {sum(p * n for p, n in zip(pre, ncand)) / 1e6 + sum(row) / 1e6:.2f}M")


def main():
    tr = [json.loads(l) for l in open(ROOT / "data/valhard/r1/train_lists.jsonl")]
    stats("VALHARD r1 train lists", tr, with_result=False)
    dec = set(json.load(open(ROOT / "data/sys1/heldout_v2/decision.json"))["ids"])
    held = [r for r in (json.loads(l) for l in open(ROOT / "data/valhard/r1/heldout_pool.jsonl")) if r["id"] in dec]
    stats("DECISION pool (R1 k=32)", held, with_result=False)


if __name__ == "__main__":
    main()
