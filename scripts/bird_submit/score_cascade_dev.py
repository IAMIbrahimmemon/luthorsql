"""Score a BIRD dev prediction file (cascade or System 1 alone) against the original dev gold, BIRD's way.

Not part of the submission package (it reads gold). EX = set of result rows equal (row order and duplicates ignored),
read-only SQLite, 30 s per query as in BIRD's evaluation script, no row cap. Reports EX overall and by difficulty and,
with --routing, separately on the questions the cascade kept and routed. This is the third and last BIRD dev look.

  .venv/bin/python scripts/bird_submit/score_cascade_dev.py --pred <predict_dev.json> [--base <stage-1 predict_dev.json>] \
      [--routing <cascade_work/routing.json>] --out <result.json>
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "data/bench/dev_20240627"


def run(task):
    db, sql = task
    con = sqlite3.connect(f"file:{DEV}/dev_databases/{db}/{db}.sqlite?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="replace")
    t = threading.Timer(30, con.interrupt)
    t.start()
    try:
        return frozenset(map(tuple, con.execute(sql).fetchall()))
    except Exception:
        return None
    finally:
        t.cancel()
        con.close()


def score(pred_path, gold, golds):
    p = json.load(open(pred_path))
    sql = [p[str(i)].split("\t----- bird -----\t")[0] for i in range(len(gold))]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run, [(g["db_id"], s) for g, s in zip(gold, sql)], chunksize=8))
    return [r is not None and g is not None and r == g for r, g in zip(res, golds)]


def ex(ok, idx):
    idx = list(idx)
    return round(100 * sum(ok[i] for i in idx) / max(1, len(idx)), 2), len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--base", type=Path)
    ap.add_argument("--routing", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    gold = json.load(open(DEV / "dev.json"))
    with ProcessPoolExecutor() as exe:
        golds = list(exe.map(run, [(g["db_id"], g["SQL"]) for g in gold], chunksize=8))
    out = {"gold": "dev_20240627 original", "n": len(gold), "gold_failed_to_run": sum(g is None for g in golds)}
    arms = {"pred": a.pred} | ({"base": a.base} if a.base else {})
    routed = set()
    if a.routing:
        routed = {int(q[1:]) for q in json.load(open(a.routing))["routed_ids"]}
    for name, path in arms.items():
        ok = score(path, gold, golds)
        r = {"ex": ex(ok, range(len(gold)))[0]}
        for d in ("simple", "moderate", "challenging"):
            r[d] = ex(ok, (i for i, g in enumerate(gold) if g.get("difficulty") == d))
        if routed:
            r["kept"] = ex(ok, (i for i in range(len(gold)) if i not in routed))
            r["routed"] = ex(ok, sorted(routed))
        out[name] = r
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
