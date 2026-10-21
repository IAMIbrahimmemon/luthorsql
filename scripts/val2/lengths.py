"""Task 4: validator input length per training list (data/val2/train_lists.jsonl), to size token-budget chunks.

  uv run python scripts/val2/lengths.py [--lists data/val2/train_lists.jsonl] [--approx]

What rerank_sq2.py train feeds the validator for one question (Reranker.scores): one schema pass
(schema_text(card, tables_for(card, sqls, budget_chars=9000))) and then one row per candidate,
candidate_text(question, context, sql) truncated to its last 384 tokens, right-padded to the longest row.
Per question: schema tokens S, n rows, longest row T, sum of row tokens, padded n*T, total S + n*T.
Also the same numbers for the old k=32 training list (1 right + 31 wrong unique running candidates) and for
the full schema (no 9,000-char cut). Tokenizer: seeklhy/OmniSQL-7B from the local HF cache (tokenizer only),
else chars / 3.5. Writes data/val2/lengths.json. The skipped-for-memory counts per DB come from
runs/phase1/p1w/logs/rrtrain_{r1,r2}.log ("OOM on <db>: one question skipped").
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sq2.serialize import SCHEMA_HEAD, question_text, schema_text  # noqa: E402

SQL_MAX = 700   # luthor.sq2.rerank.SQL_MAX (that module imports torch)
MAX_TOK = 384   # Reranker.scores max_tok
BUDGET = 9000   # rerank_sq2.py --budget-chars default (box W did not override it)


def candidate_text(question, context, sql):  # = luthor.sq2.rerank.candidate_text without a result preview
    q = question_text(question, context).text.removesuffix("\nPlan:")
    return f"{q}\nCandidate SQL: {sql[:SQL_MAX]}\nDoes this SQL answer the question exactly?"


def tables_for(card, sqls, budget):  # = scripts/rerank_sq2.py tables_for
    if len(schema_text(card).text) <= budget:
        return None
    used = [t.name for t in card.tables if any(f'"{t.name}"' in s for s in sqls)]
    size = {t.name: len(schema_text(card, [t.name]).text) - len(SCHEMA_HEAD) for t in card.tables}
    keep, total = list(used), len(SCHEMA_HEAD) + sum(size[t] for t in used)
    for t in card.tables:
        if t.name not in keep and total + size[t.name] <= budget:
            keep.append(t.name)
            total += size[t.name]
    names = [t.name for t in card.tables]
    return sorted(keep, key=names.index)


def tokenizer(approx: bool):
    if not approx:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("seeklhy/OmniSQL-7B", local_files_only=True)
            return lambda s: len(tok(s, add_special_tokens=False)["input_ids"]), "seeklhy/OmniSQL-7B tokenizer"
        except Exception as e:
            print(f"tokenizer unavailable ({e}); chars/3.5", flush=True)
    return lambda s: int(np.ceil(len(s) / 3.5)), "chars/3.5"


def dist(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "mean": round(float(x.mean()), 1), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", type=Path, default=ROOT / "data/val2/train_lists.jsonl")
    ap.add_argument("--approx", action="store_true")
    a = ap.parse_args()
    ntok, how = tokenizer(a.approx)
    cards = {p.stem: p for p in (ROOT / "data/sys1/bird_train/cards").glob("*.json")}
    cache = {}
    rows = []
    pools = {}
    for role in ("r1", "r2"):  # old k=32 training lists (what box W trained on), for comparison
        for ln in open(ROOT / f"runs/phase1/p1w/p1w/{role}/cands_train.jsonl"):
            r = json.loads(ln)
            pools.setdefault(r["id"], {})[role] = r
    full_schema = {}
    for ln in open(a.lists):
        r = json.loads(ln)
        db = r["db_id"]
        if db not in cache:
            cache[db] = SchemaCard.load(cards[db])
            full_schema[db] = ntok(schema_text(cache[db]).text)
        card = cache[db]
        sqls = [c["sql"] for c in r["cands"]]
        S = ntok(schema_text(card, tables_for(card, sqls, BUDGET)).text)
        L = [min(ntok(candidate_text(r["question"], r["context"], s)), MAX_TOK) for s in sqls]
        old = {}
        for role, p in pools.get(r["id"], {}).items():
            u, seen = [], set()
            for c in p["cands"]:
                if c.get("sql") and c["sql"] not in seen and c.get("runs"):
                    seen.add(c["sql"])
                    u.append(c)
            pos = [c for c in u if c["ok"]][:1]
            neg = [c for c in u if not c["ok"]][:31]
            if pos and neg:
                osq = [c["sql"] for c in pos + neg]
                oS = ntok(schema_text(card, tables_for(card, osq, BUDGET)).text)
                oL = [min(ntok(candidate_text(p["question"], p["context"], s)), MAX_TOK) for s in osq]
                old[role] = {"n": len(oL), "total": oS + len(oL) * max(oL)}
        rows.append({"id": r["id"], "db_id": db, "schema": S, "schema_full": full_schema[db], "n": len(L),
                     "row_max": max(L), "row_sum": sum(L), "padded": len(L) * max(L), "total": S + len(L) * max(L),
                     "old_n": max((v["n"] for v in old.values()), default=0),
                     "old_total": max((v["total"] for v in old.values()), default=0)})
    skips = {}
    for role in ("r1", "r2"):
        log = ROOT / f"runs/phase1/p1w/logs/rrtrain_{role}.log"
        skips[role] = dict(collections.Counter(re.findall(r"^OOM on (\S+): one question skipped", log.read_text(), re.M)))
    by_db = collections.defaultdict(list)
    for x in rows:
        by_db[x["db_id"]].append(x)
    dbs = sorted(({"db_id": d, "lists": len(v), "schema_full_tokens": v[0]["schema_full"],
                   "schema_tokens_max": max(x["schema"] for x in v), "total_p50": float(np.median([x["total"] for x in v])),
                   "total_max": max(x["total"] for x in v), "old_total_max": max(x["old_total"] for x in v),
                   "skipped_r1": skips["r1"].get(d, 0), "skipped_r2": skips["r2"].get(d, 0)} for d, v in by_db.items()),
                 key=lambda d: -d["total_max"])
    tot = np.array([x["total"] for x in rows])
    out = {"tokenizer": how, "lists": str(a.lists.relative_to(ROOT)), "budget_chars": BUDGET, "row_max_tok": MAX_TOK,
           "per_question": {k: dist([x[k] for x in rows]) for k in ("schema", "schema_full", "n", "row_max", "row_sum",
                                                                    "padded", "total")},
           "old_k32_per_question": {k: dist([x[k] for x in rows if x["old_n"]]) for k in ("old_n", "old_total")},
           "share_total_over": {str(t): round(float((tot > t).mean()), 4) for t in (2048, 4096, 6144, 8192, 12288)},
           "approx_q_per_pass_4_total": dist([sum(x["total"] - x["schema"] for x in v[i:i + 4]) + max(x["schema"] for x in v[i:i + 4])
                                       for v in by_db.values() for i in range(0, len(v), 4)]),
           "skipped_for_memory": {k: {"total": sum(v.values()), "by_db": dict(sorted(v.items(), key=lambda t: -t[1]))}
                                  for k, v in skips.items()},
           "by_db": dbs,
           "longest_questions": sorted(rows, key=lambda x: -x["total"])[:15]}
    (ROOT / "data/val2/lengths.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("tokenizer", "per_question", "old_k32_per_question", "share_total_over",
                                          "approx_q_per_pass_4_total")}, indent=1))
    for d in dbs[:15]:
        print(d)


if __name__ == "__main__":
    main()
