"""Selector candidate files (Mac): pool autoregressive samples (gen6 outputs) and slot-model candidates per
question, execute every distinct SQL once, and write rerank_sq2.py records (runs / nonnull / ok / sig / res).

  python scripts/build_sel.py --questions data/eval/sel/verified_train.jsonl --gen runs/sel/sel_a.jsonl ... \
      --db-roots data/raw/bird_train/train_databases --out data/eval/sel/train_cands.jsonl
  python scripts/build_sel.py --questions data/eval/dev6c_bird.jsonl --gen runs/r6/s1_omni.jsonl runs/arc/arc_bird_dev.jsonl \
      --slot runs/sys1/r6c/sys1_full/rr/bird_dev_scored.jsonl --db-roots data/bench/dev_20240627/dev_databases \
      --out data/eval/sel/score_bird_dev.jsonl

Candidate order per question: each source's greedy answer first, then samples (so "greedy" = first source's
greedy). A gen6 candidate's one-shot fix (--fix) is added as its own candidate when it differs.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sq2.rerank import result_preview  # noqa: E402
from luthor.sqlw.evalkit import db_paths  # noqa: E402
from luthor.sqlw.execute import connect, signature  # noqa: E402

SECONDS = 10.0


def run_db(task):
    path, sqls = task
    run = connect(path, SECONDS)
    out = {}
    for s in sqls:
        try:
            rows = run(s)
            out[s] = {"runs": True, "nonnull": any(v is not None for r in rows for v in r), "sig": signature(rows),
                      "res": result_preview(rows)}
        except Exception as e:
            out[s] = {"runs": False, "nonnull": False, "sig": "", "res": "error: " + str(e)[:80]}
    return path, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--gen", nargs="*", type=Path, default=[], help="gen6 outputs (cands with src, sql)")
    ap.add_argument("--slot", type=Path, help="slot-model candidates (rerank_sq2 gen/scored file)")
    ap.add_argument("--slot-k", type=int, default=8)
    ap.add_argument("--db-roots", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    qs = {r["id"]: r for r in map(json.loads, open(a.questions))}
    cands = collections.defaultdict(list)
    for f in a.gen:
        tag = f.stem
        for r in map(json.loads, open(f)):
            if r["id"] not in qs:
                continue
            cs = sorted(r["cands"], key=lambda c: not c["src"].endswith("_g"))
            for c in cs:
                cands[r["id"]].append({"sql": c["sql"], "src": f"{tag}:{c['src']}"})
                fx = c.get("fixed")
                if fx and fx.get("sql") and fx["sql"] != c["sql"]:
                    cands[r["id"]].append({"sql": fx["sql"], "src": f"{tag}:{c['src']}_fix"})
    in_slots = {}
    if a.slot:
        for r in map(json.loads, open(a.slot)):
            if r["id"] in qs:
                in_slots[r["id"]] = bool(r.get("in_slots"))
                for c in r["cands"][:a.slot_k]:
                    if c.get("sql"):
                        cands[r["id"]].append({"sql": c["sql"], "src": "slot", "cost": c.get("cost", 0.0)})
    paths = db_paths(a.db_roots)
    by_db = collections.defaultdict(set)
    for q, cs in cands.items():
        by_db[qs[q]["db_id"]].update(c["sql"] for c in cs if c["sql"])
    tasks = []
    for db, sqls in by_db.items():
        sqls = sorted(sqls)
        tasks += [(paths[db], sqls[i:i + 40]) for i in range(0, len(sqls), 40)]
    res = collections.defaultdict(dict)
    with Pool(a.workers) as p:
        for path, out in p.imap_unordered(run_db, tasks):
            res[path].update(out)
    recs, n_ok, n_any = [], 0, 0
    for q, r in qs.items():
        cs = []
        for c in cands.get(q, []):
            if not c["sql"]:
                continue
            x = dict(res[paths[r["db_id"]]][c["sql"]])
            x["ok"] = bool(x["runs"] and x["sig"] == r["gold_sig"])
            cs.append({**c, **x, "cost": c.get("cost", 0.0)})
        n_ok += bool(cs and cs[0]["ok"])
        n_any += any(c["ok"] for c in cs)
        recs.append({"id": q, "db_id": r["db_id"], "question": r["question"], "context": r.get("context", ""),
                     "gold_sql": r.get("gold_sql", ""), "in_slots": in_slots.get(q, False), "cands": cs})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    n = len(recs)
    print(f"{a.out.name}: {n} questions, mean candidates {sum(len(r['cands']) for r in recs) / n:.1f}, "
          f"first right {n_ok / n:.1%}, any right {n_any / n:.1%}")


if __name__ == "__main__":
    main()
