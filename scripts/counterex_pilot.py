"""Counterexample pilot (round 5, step 1): how many writer answers that match the gold on the real
database stop matching on mutated copies? Diagnosis only; official scoring is unchanged.

  python scripts/counterex_pilot.py --rescored data/eval/rescored_rl.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sqlw.counterex import MUTATIONS, make_variants  # noqa: E402
from luthor.sqlw.evalkit import db_paths  # noqa: E402
from luthor.sqlw.execute import connect  # noqa: E402

ROOTS = ["data/raw/spider_data/database", "data/raw/bird_train/train_databases"]


def run_db(task):
    """(variant paths, [(id, gold, cand)]) -> [(id, {mutation: 'same'|'diff'|'gold_fail'|'cand_fail'})]."""
    variants, items = task
    runs = {m: connect(str(p), 10.0) for m, p in variants.items()}
    out = []
    for qid, gold, cand in items:
        res = {}
        for m, run in runs.items():
            try:
                g = set(map(tuple, run(gold)))
            except Exception:
                res[m] = "gold_fail"
                continue
            try:
                c = set(map(tuple, run(cand)))
            except Exception:
                res[m] = "cand_fail"
                continue
            res[m] = "same" if g == c else "diff"
        out.append((qid, res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", type=Path, default=Path("data/eval/dev_frozen.jsonl"))
    ap.add_argument("--rescored", type=Path, default=Path("data/eval/rescored_rl.json"))
    ap.add_argument("--work", type=Path, default=Path("data/eval/cx"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/counterex_pilot.json"))
    a = ap.parse_args()
    frozen = {json.loads(l)["id"]: json.loads(l) for l in open(a.frozen)}
    cands = json.loads(a.rescored.read_text())
    paths = db_paths(ROOTS)
    items = collections.defaultdict(list)
    for qid, cs in cands.items():
        w = sorted([c for c in cs if c["src"] == "w"], key=lambda c: c["rank"])
        if w and w[0].get("ok") and w[0]["sql"].strip().lower() != frozen[qid]["gold_sql"].strip().lower():
            items[frozen[qid]["db_id"]].append((qid, frozen[qid]["gold_sql"], w[0]["sql"]))
    tasks = []
    for db, its in items.items():
        vdir = a.work / db
        vp = {m: vdir / f"{m}.sqlite" for m in MUTATIONS}
        if not all(p.exists() for p in vp.values()):
            make_variants(paths[db], vdir, seed=0)
        tasks.append((vp, its))
    with Pool(8) as pool:
        res = [x for part in pool.map(run_db, tasks) for x in part]
    n = len(res)
    any_diff = [(q, r) for q, r in res if "diff" in r.values()]
    by_m = {m: collections.Counter(r[m] for _, r in res) for m in MUTATIONS}
    has_limit = lambda q: bool(re.search(r"\blimit\b", frozen[q]["gold_sql"], re.I))
    report = {
        "greedy_right_and_not_identical_to_gold": n,
        "differs_on_some_variant": len(any_diff),
        "differs_share": round(100 * len(any_diff) / max(n, 1), 1),
        "differs_without_limit": sum(not has_limit(q) for q, _ in any_diff),
        "by_mutation": {m: dict(c) for m, c in by_m.items()},
        "by_source": dict(collections.Counter(frozen[q]["source"] for q, _ in any_diff)),
        "examples": [{"id": q, "question": frozen[q]["question"], "gold": frozen[q]["gold_sql"],
                      "cand": next(c["sql"] for c in cands[q] if c["src"] == "w" and c["rank"] == 0),
                      "diff_on": [m for m, v in r.items() if v == "diff"]} for q, r in any_diff[:40]],
    }
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
