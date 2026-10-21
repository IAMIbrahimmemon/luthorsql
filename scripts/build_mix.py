"""Build the training mix from the converted plan sets.

  python scripts/build_mix.py --out data/mix

- train sets: bird_train, spider_train, spider_syn_train, wikisql_train, synsql_train
- eval sets (sealed, never mixed in): bird_dev, spider_dev, spider_test, spider_syn_dev, wikisql_dev, wikisql_test
- no eval database may appear in training (checked by db_id; Spider-Syn shares Spider's databases)
- a training question with >= 50% word overlap (Jaccard) with any eval question is dropped
- human-written sets are repeated (--real-repeat); synthetic / single-table sets are capped
- empty-result plans are dropped; unsupported rows are subsampled (--unsup-keep)
Output: data/mix/plans.jsonl + data/mix/cards/ (one card per training database) + stats.json.
Everything in data/mix is public-dataset derived; it is what gets uploaded to the pod.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_plans import dedup_against  # noqa: E402

TRAIN = {"bird_train": "real", "spider_train": "real", "spider_syn_train": "real",
         "wikisql_train": "wiki", "synsql_train": "synth"}
EVAL = ["bird_dev", "spider_dev", "spider_test", "spider_syn_dev", "wikisql_dev", "wikisql_test"]


def rows(path: Path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", type=Path, default=Path("data/plans"))
    ap.add_argument("--out", type=Path, default=Path("data/mix"))
    ap.add_argument("--real-repeat", type=int, default=2)
    ap.add_argument("--wiki-cap", type=int, default=25000, help="WikiSQL originals kept (twins follow their original)")
    ap.add_argument("--synth-cap", type=int, default=120000)
    ap.add_argument("--unsup-keep", type=float, default=0.25)
    a = ap.parse_args()
    rnd = random.Random(0)

    eval_q, eval_dbs = [], set()
    for e in EVAL:
        p = a.plans / e / "plans.jsonl"
        if p.exists():
            for r in rows(p):
                eval_q.append(r["question"])
                eval_dbs.add(r["db_id"])
    print(f"eval: {len(eval_q)} questions, {len(eval_dbs)} databases (sealed)")

    out_rows, stats = [], collections.Counter()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "cards").mkdir(exist_ok=True)
    for name, kind in TRAIN.items():
        d = a.plans / name
        if not (d / "plans.jsonl").exists():
            print(f"missing {name}, skipped")
            continue
        rs = [r for r in rows(d / "plans.jsonl") if not r.get("empty_result")]
        leaked = [r for r in rs if r["db_id"] in eval_dbs]
        if leaked:
            stats[f"{name}:dropped_eval_db"] += len(leaked)
            rs = [r for r in rs if r["db_id"] not in eval_dbs]
        rs = [r for r in rs if r["outcome"] == "plan" or rnd.random() < a.unsup_keep]
        before = len(rs)
        rs = dedup_against(rs, eval_q)
        stats[f"{name}:dedup_removed"] += before - len(rs)
        # caps apply to originals; twins follow their original
        cap = {"wiki": a.wiki_cap, "synth": a.synth_cap}.get(kind)
        if cap:
            origs = [r for r in rs if not r.get("twin_of")]
            rnd.shuffle(origs)
            keep = {r["id"] for r in origs[:cap]}
            rs = [r for r in rs if (r.get("twin_of") or r["id"]) in keep]
        reps = a.real_repeat if kind == "real" else 1
        for k in range(reps):
            for r in rs:
                rr = dict(r)
                if k:
                    rr["id"] = f"{r['id']}~{k}"
                    if rr.get("twin_of"):
                        rr["twin_of"] = f"{r['twin_of']}~{k}"
                out_rows.append(rr)
        dbs = {r["db_id"] for r in rs}
        for db in dbs:
            src = d / "cards" / f"{db}.json"
            if src.exists():
                shutil.copy(src, a.out / "cards" / f"{db}.json")
        n_plan = sum(r["outcome"] == "plan" for r in rs)
        stats[f"{name}:rows"] = len(rs)
        stats[f"{name}:plans"] = n_plan
        stats[f"{name}:twins"] = sum(bool(r.get("twin_of")) for r in rs)
        stats[f"{name}:unsupported"] = len(rs) - n_plan
        stats[f"{name}:databases"] = len(dbs)
        print(f"{name:18s} rows {len(rs):>7} x{reps}  plans {n_plan:>7}  dbs {len(dbs):>6}", flush=True)
    rnd.shuffle(out_rows)
    with open(a.out / "plans.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    stats["total_rows"] = len(out_rows)
    (a.out / "stats.json").write_text(json.dumps(dict(stats), indent=1))
    print(f"total {len(out_rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
