"""VAL4 list helpers (Mac, CPU, stdlib only), called by scripts/val4/build_lists.sh around VALHARD's unchanged prep steps.

  python3 scripts/val4/lists.py pool  --fold 1=F1 --fold 2=F2 --out data/val4/work/cands_folds.jsonl [--max-greedy 68]
  python3 scripts/val4/lists.py merge --work data/val4/work --old data/valhard/r1/train_lists.jsonl --out data/val4

pool   checks every fold list (each question on a DB of its fold's dev list data/val4/fold<F>_dev.json, never a
       held-out DB, every candidate labelled), writes the union of both folds (one generator per question: the folds are
       disjoint) as ONE pool for VALHARD's prep (scripts/val2/build_lists.py, scripts/valhard/mine.py and mix.py, all
       with their defaults), and prints the raw greedy-right rate (rank-0 candidate ok, gen-time labels) per fold and
       overall. PREREG data rule: overall greedy right must be <= --max-greedy (68%; DECISION 58.2%, VALHARD's
       in-sample lists 74.8%), else UNSEEN_CHECK STOP (exit 3: lists not DB-unseen-like, the validator is not trained).
merge  train = the fold lists on non-val DBs (work/train_lists.jsonl) + VALHARD's train lists (--old), val = the fold
       lists on the 4 pinned val DBs (work/val_lists.jsonl; NOT in train). Checks: val only on the pinned DBs, no
       pinned DB in train, no held-out DB anywhere. Writes <out>/train_lists.jsonl, <out>/val_lists.jsonl and
       <out>/lists_stats.json (counts, list-level greedy right = 1 - share of weight-3 lists, per source).
Only BIRD-train data is read. Prints counts and BIRD-train rates only.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELDOUT = ROOT / "data/sys1/dev_dbs_bird.json"
FOLDS = ROOT / "data/val4"
PINNED = {"citeseer", "disney", "professional_basketball", "trains"}
MAX_GREEDY = 68.0


def greedy_right(r: dict) -> bool:
    cs = sorted((c for c in r.get("cands", []) if c.get("sql")), key=lambda c: c.get("rank", 0))
    return bool(cs and cs[0].get("ok"))


def rate(rows) -> float:
    rows = list(rows)
    return round(100 * sum(map(greedy_right, rows)) / max(len(rows), 1), 2)


def check_fold(rows: list[dict], fold_dbs: set[str], held: set[str], name: str) -> list[str]:
    bad = []
    for r in rows:
        if r["db_id"] in held:
            bad.append(f"{name}: held-out DB {r['db_id']} ({r['id']})")
        elif r["db_id"] not in fold_dbs:
            bad.append(f"{name}: {r['db_id']} is not a DB of this fold ({r['id']})")
        if any("ok" not in c for c in r.get("cands", [])):
            bad.append(f"{name}: unlabelled candidate in {r['id']}")
    return bad


def pool(folds: dict[str, Path], out: Path, max_greedy: float = MAX_GREEDY, fold_dir: Path = FOLDS,
         held_path: Path = HELDOUT) -> dict:
    held = set(json.load(open(held_path)))
    rows, per, problems, seen = [], {}, [], set()
    for f, p in sorted(folds.items()):
        dbs = set(json.load(open(fold_dir / f"fold{f}_dev.json")))
        rs = [json.loads(l) for l in open(p)]
        problems += check_fold(rs, dbs, held, f"fold{f}")
        dup = {r["id"] for r in rs} & seen
        if dup:
            problems.append(f"fold{f}: {len(dup)} ids already in another fold")
        seen |= {r["id"] for r in rs}
        per[f] = {"questions": len(rs), "dbs": len({r["db_id"] for r in rs}), "greedy_right": rate(rs),
                  "oracle_right": round(100 * sum(any(c.get("ok") for c in r["cands"]) for r in rs) / max(len(rs), 1), 2)}
        rows += rs
    st = {"folds": per, "questions": len(rows), "greedy_right": rate(rows), "max_greedy": max_greedy,
          "problems": problems[:20]}
    if problems:
        return {**st, "status": "FAIL"}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, out)
    st["status"] = "PASS" if st["greedy_right"] <= max_greedy else "STOP"
    return st


def list_stats(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {"lists": len(rows), "dbs": len({r["db_id"] for r in rows}),
            "entries": sum(len(r["cands"]) for r in rows),
            "mean_entries": round(sum(len(r["cands"]) for r in rows) / n, 2),
            "weight3_share": round(sum(float(r.get("weight", 1.0)) == 3.0 for r in rows) / n, 4),
            "list_greedy_right": round(100 * (1 - sum(float(r.get("weight", 1.0)) == 3.0 for r in rows) / n), 2)}


def merge(work: Path, old: Path, out: Path, held_path: Path = HELDOUT) -> dict:
    held = set(json.load(open(held_path)))
    new_tr = [json.loads(l) for l in open(work / "train_lists.jsonl")]
    new_va = [json.loads(l) for l in open(work / "val_lists.jsonl")]
    old_tr = [json.loads(l) for l in open(old)]
    problems = []
    for name, rows in (("fold train", new_tr), ("valhard train", old_tr)):
        problems += [f"{name}: pinned val DB {r['db_id']}" for r in rows if r["db_id"] in PINNED][:5]
    problems += [f"val: {r['db_id']} is not a pinned val DB" for r in new_va if r["db_id"] not in PINNED][:5]
    problems += [f"held-out DB {r['db_id']}" for r in new_tr + new_va + old_tr if r["db_id"] in held][:5]
    tr_ids = {r["id"] for r in new_tr + old_tr}
    problems += [f"val id {r['id']} also in train" for r in new_va if r["id"] in tr_ids][:5]
    if problems:
        return {"status": "FAIL", "problems": problems}
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train_lists.jsonl", new_tr + old_tr), ("val_lists.jsonl", new_va)):
        tmp = out / (name + ".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp, out / name)
    st = {"status": "OK", "train": list_stats(new_tr + old_tr), "val": list_stats(new_va),
          "train_fold": list_stats(new_tr), "train_valhard": list_stats(old_tr),
          "ids_in_both_sources": len({r["id"] for r in new_tr} & {r["id"] for r in old_tr}),
          "sources": {"fold_work": str(work), "valhard": str(old)}}
    (out / "lists_stats.json").write_text(json.dumps(st, indent=1) + "\n")
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pool", "merge"])
    ap.add_argument("--fold", action="append", default=[], metavar="F=PATH")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-greedy", type=float, default=MAX_GREEDY)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--old", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "pool":
        folds = dict(s.split("=", 1) for s in a.fold)
        if sorted(folds) != ["1", "2"]:
            print("need --fold 1=PATH --fold 2=PATH")
            return 2
        st = pool({k: Path(v) for k, v in folds.items()}, a.out, a.max_greedy)
        for p in st["problems"]:
            print("VAL4_POOL_PROBLEM", p)
        for f, s in st["folds"].items():
            print(f"fold {f}: {s['questions']} questions on {s['dbs']} DBs, greedy right {s['greedy_right']}%, "
                  f"oracle {s['oracle_right']}%")
        print(f"UNSEEN_CHECK {st['status']}: greedy right {st['greedy_right']}% over {st['questions']} fold questions "
              f"(rule <= {a.max_greedy}%; DECISION 58.2, VALHARD in-sample 74.8)")
        return {"PASS": 0, "STOP": 3}.get(st["status"], 1)
    if not (a.work and a.old):
        print("merge needs --work and --old")
        return 2
    st = merge(a.work, a.old, a.out)
    print("VAL4_LISTS " + json.dumps(st))
    return 0 if st["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
