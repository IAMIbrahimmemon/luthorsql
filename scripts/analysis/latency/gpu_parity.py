"""GPU / environment parity of the frozen System 1 (R1 k=32 + VALHARD) on the latency sample (200 DECISION questions).

Compares the box's batched k=32 picks (`picks_k32.jsonl`, written by sys1_latency.py run; label-free on the box) with
the frozen picks for the same questions (VALHARD's held-out file, same pick rule: first max-rr live candidate), and
scores both by strict EX against the original BIRD-train gold here on the Mac.

  .venv/bin/python scripts/analysis/latency/gpu_parity.py runs/phase1/bird_a100/latency --json runs/phase1/bird_a100/parity.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "data/raw/bird_train/train_databases"
FROZEN = ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl"


def run(db, sql, t=30):
    if not sql:
        return None
    con = sqlite3.connect(f"file:{DB}/{db}/{db}.sqlite?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="replace")
    tm = threading.Timer(t, con.interrupt)
    tm.start()
    try:
        return frozenset(con.execute(sql).fetchall())
    except Exception:
        return None
    finally:
        tm.cancel()
        con.close()


def job(a):
    qid, db, gold, sqls = a
    g = run(db, gold)
    return qid, [g is not None and run(db, s) == g for s in sqls]


def frozen_picks(ids: set[str]) -> dict[str, str | None]:
    out = {}
    for ln in open(FROZEN):
        r = json.loads(ln)
        if r["id"] not in ids:
            continue
        c = [x for x in r["cands"] if x.get("sql") and x.get("src") in ("r1", "both", None)]
        lv = [x for x in c if x.get("runs") and x.get("nonnull")] or c
        out[r["id"]] = max(lv, key=lambda x: x.get("rr", -1e9))["sql"] if lv else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("box_dir", type=Path, help="dir holding picks_k32.jsonl (and latency.json)")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    box = {r["id"]: r["sql"] for r in map(json.loads, open(a.box_dir / "picks_k32.jsonl"))}
    ids = sorted(box)
    fr = frozen_picks(set(ids))
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    row = {i: tr[int(i.rsplit("_", 1)[1])] for i in ids}
    jobs = [(i, row[i]["db_id"], row[i]["SQL"], [box[i], fr.get(i)]) for i in ids]
    with ProcessPoolExecutor(6) as ex:
        res = dict(ex.map(job, jobs, chunksize=4))
    n = len(ids)
    same = sum(box[i] == fr.get(i) for i in ids)
    out = {"n": n, "frozen_found": sum(i in fr for i in ids), "same_sql": same, "same_sql_pct": round(100 * same / n, 1),
           "ex_box": round(100 * sum(res[i][0] for i in ids) / n, 2),
           "ex_frozen": round(100 * sum(res[i][1] for i in ids) / n, 2),
           "box_only_right": sum(res[i][0] and not res[i][1] for i in ids),
           "frozen_only_right": sum(res[i][1] and not res[i][0] for i in ids),
           "differ": [i for i in ids if box[i] != fr.get(i)]}
    c = [i for i in ids if i in fr]   # questions the frozen file holds (it lacks some DECISION ids)
    out["comparable"] = {"n": len(c), "same_sql": sum(box[i] == fr[i] for i in c),
                         "ex_box": round(100 * sum(res[i][0] for i in c) / max(1, len(c)), 2),
                         "ex_frozen": round(100 * sum(res[i][1] for i in c) / max(1, len(c)), 2),
                         "box_only_right": sum(res[i][0] and not res[i][1] for i in c),
                         "frozen_only_right": sum(res[i][1] and not res[i][0] for i in c)}
    lj = a.box_dir / "latency.json"
    if lj.exists():
        L = json.load(open(lj))
        out["env"] = L.get("env") or L.get("versions")
    print(json.dumps({k: v for k, v in out.items() if k != "differ"}, indent=1))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
