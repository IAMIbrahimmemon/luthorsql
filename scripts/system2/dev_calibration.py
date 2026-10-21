"""Descriptive dev calibration table for the System 2 dev run (PREREG.md: diagnostic only, never used to retune).
Same bands as docs/paper_notes/cascade_threshold_2026-10-17.md; per band: n, System 1 / A1 / A0 EX on original gold.

  .venv/bin/python scripts/system2/dev_calibration.py --box <sandbox> --out runs/phase2/system2_dev/calibration.json
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from score_dev import DEV, POOL, load_answers, margin, run

BANDS = [(-math.inf, 0.0, "no margin"), (0.0, 0.52, "0 - 0.52"), (0.52, 1.38, "0.52 - 1.38"), (1.38, 2.48, "1.38 - 2.48"),
         (2.48, 3.53, "2.48 - 3.53"), (3.53, 4.12, "3.53 - 4.12"), (4.12, 4.62, "4.12 - 4.62"),
         (4.62, 5.39, "4.62 - 5.39"), (5.39, math.inf, ">= 5.39")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    q = json.load(open(DEV / "dev.json"))
    ids = [f"bird_dev_{i}" for i in range(len(q))]
    db = {i: q[k]["db_id"] for k, i in enumerate(ids)}
    gold = {i: q[k]["SQL"] for k, i in enumerate(ids)}
    chunks = {}
    for f in (a.box / "chunks/A1").glob("*.jsonl"):
        for ln in open(f):
            r = json.loads(ln)
            chunks[r["question_id"]] = r
    s1, m = {}, {}
    for ln in open(POOL):
        r = json.loads(ln)
        live = [c for c in r["cands"] if c.get("sql") and c.get("runs") and c.get("nonnull")] or \
               [c for c in r["cands"] if c.get("sql")]
        s1[r["id"]] = max(live, key=lambda c: c.get("rr", -1e9))["sql"] if live else None
        m[r["id"]] = margin(r["cands"])[0]
    preds = {"S1": s1, "A1": load_answers(a.box, "A1", chunks), "A0": load_answers(a.box, "A0", chunks)}
    tasks = {(db[i], s) for i in ids for s in [gold[i]] + [p.get(i) for p in preds.values()] if s}
    with ProcessPoolExecutor(8) as ex:
        res = dict(ex.map(run, sorted(tasks, key=str), chunksize=16))

    def ok(i, sql):
        pr, gr = res.get((db[i], sql)) if sql else None, res.get((db[i], gold[i]))
        return pr is not None and gr is not None and pr == gr

    rows = []
    for lo, hi, name in BANDS:
        sel = [i for i in ids if (m[i] == -math.inf if name == "no margin" else lo <= m[i] < hi)]
        row = {"band": name, "n": len(sel)}
        for arm in preds:
            row[arm] = round(100 * sum(ok(i, preds[arm].get(i)) for i in sel) / max(1, len(sel)), 1)
        rows.append(row)
    a.out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
