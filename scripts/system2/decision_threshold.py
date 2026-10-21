"""Cascade threshold on all 796 DECISION questions (runs/phase2/decision_rest/PREREG.md).
A1 (Opus + System 1 handoff) = the 2026-10-12 400-sample answers (runs/analysis/hybrid/opus_s1.json, plus_s1) + the
396 new answers in <box>/answers/A1, scored here by strict EX vs the original BIRD-train gold (30 s). System 1
correctness and margin come from the frozen pool via scripts/analysis/cascade_threshold.py:load / margin.
Primary rule (frozen): t on the grid 2.0..6.0 step 0.25 maximising V*EX - cost with V = $1; ties -> lower t.

  .venv/bin/python scripts/system2/decision_threshold.py --box <sandbox> --cost-opus 0.024 --out runs/phase2/decision_rest/result.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from cascade_threshold import load  # noqa: E402
from luthor.stats import cluster_signflip, percentile_bootstrap  # noqa: E402

DB = ROOT / "data/raw/bird_train/train_databases"
TRAIN = ROOT / "data/raw/bird_train/train.json"
GRID = [2.0 + 0.25 * k for k in range(17)]
COST_S1 = 0.00064


def run(task):
    db, sql = task
    con = sqlite3.connect(f"file:{DB}/{db}/{db}.sqlite?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="replace")
    t = threading.Timer(30, con.interrupt)
    t.start()
    try:
        return task, frozenset(con.execute(sql).fetchall())
    except Exception:
        return task, None
    finally:
        t.cancel()
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--cost-opus", type=float, required=True, help="A1 $ per question from this run's usage logs")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    s1, opus = load()
    tr = json.load(open(TRAIN))
    chunks, new = {}, {}
    for f in sorted((a.box / "chunks/A1").glob("*.jsonl")):
        for ln in open(f):
            r = json.loads(ln)
            chunks[r["question_id"]] = r
    for f in sorted((a.box / "answers/A1").glob("*.jsonl")):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            q = r.get("question_id")
            if q in new or q not in chunks:
                continue
            pick = str(r.get("pick", "")).strip()
            cand = {c["label"]: c["sql"] for c in chunks[q]["candidates"]}
            new[q] = (r.get("sql") or None) if pick.upper() == "NONE" else cand.get(pick)
    missing = sorted(set(chunks) - set(new))
    gold = {q: tr[int(q.rsplit("_", 1)[1])]["SQL"] for q in chunks}
    tasks = {(chunks[q]["db_id"], s) for q in chunks for s in (gold[q], new.get(q)) if s}
    with ProcessPoolExecutor(8) as ex:
        res = dict(ex.map(run, sorted(tasks), chunksize=8))
    plus = {q: bool(v["plus_s1"]) for q, v in opus.items()}
    for q in chunks:
        g, p = res.get((chunks[q]["db_id"], gold[q])), res.get((chunks[q]["db_id"], new.get(q))) if new.get(q) else None
        plus[q] = g is not None and p is not None and p == g
    ids = sorted(s1)
    assert set(plus) == set(ids), (len(plus), len(ids))
    db = {i: s1[i]["db"] for i in ids}
    n = len(ids)
    always = {i: plus[i] for i in ids}

    def cascade(t):
        return {i: (s1[i]["ok"] if s1[i]["m"] >= t else plus[i]) for i in ids}

    rows = []
    for t in GRID:
        h = cascade(t)
        share = sum(s1[i]["m"] < t for i in ids) / n
        ex = 100 * sum(h.values()) / n
        rows.append({"t": t, "ex": round(ex, 2), "opus_share": round(share, 4),
                     "cost": round(share * a.cost_opus + COST_S1, 5), "loss_vs_always": round(ex - 100 * sum(always.values()) / n, 2)})

    def choose(value, sel=None):
        sel = sel or ids
        best = None
        for t in GRID:
            h = cascade(t)
            share = sum(s1[i]["m"] < t for i in sel) / len(sel)
            u = value * sum(h[i] for i in sel) / len(sel) - (share * a.cost_opus + COST_S1)
            if best is None or u > best[1] + 1e-12:
                best = (t, u)
        return best[0]

    t_star = choose(1.0)
    h = cascade(t_star)
    lo, hi = percentile_bootstrap(always, h, db)
    out = {"n": n, "new_answered": len(new), "new_missing": missing, "cost_opus": a.cost_opus,
           "always_handoff_ex": round(100 * sum(always.values()) / n, 2),
           "system1_ex": round(100 * sum(s1[i]["ok"] for i in ids) / n, 2),
           "new396_handoff_ex": round(100 * sum(plus[q] for q in chunks) / len(chunks), 2),
           "curve": rows, "primary_t": t_star,
           "primary": {**next(r for r in rows if r["t"] == t_star), "db_bootstrap_95": [lo, hi],
                       "signflip": cluster_signflip(always, h, db)},
           "sensitivity": {"V=0.25": choose(0.25), "V=4": choose(4.0)},
           "lodo": {d: choose(1.0, [i for i in ids if db[i] != d]) for d in sorted(set(db.values()))}}
    # band table on all 796: System 1 vs handoff, and the 400 sample vs the 396 new questions
    edges = [(-math.inf, -math.inf, "no margin"), (0, 0.52, "0-0.52"), (0.52, 1.38, "0.52-1.38"), (1.38, 2.48, "1.38-2.48"),
             (2.48, 3.53, "2.48-3.53"), (3.53, 4.12, "3.53-4.12"), (4.12, 4.62, "4.12-4.62"), (4.62, 5.39, "4.62-5.39"),
             (5.39, math.inf, ">=5.39")]
    bands = []
    for lo_, hi_, name in edges:
        sel = [i for i in ids if (s1[i]["m"] == -math.inf if name == "no margin" else lo_ <= s1[i]["m"] < hi_)]
        row = {"band": name, "n": len(sel)}
        for part, keep in (("all", lambda i: True), ("sample400", lambda i: i in opus), ("new396", lambda i: i in chunks)):
            g = [i for i in sel if keep(i)]
            row[part] = {"n": len(g), "s1": round(100 * sum(s1[i]["ok"] for i in g) / max(1, len(g)), 1),
                         "handoff": round(100 * sum(plus[i] for i in g) / max(1, len(g)), 1)}
        bands.append(row)
    out["bands"] = bands
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("n", "new_answered", "always_handoff_ex", "system1_ex", "new396_handoff_ex",
                                          "primary_t", "sensitivity", "lodo")}, indent=1))
    print("PRIMARY", json.dumps({k: v for k, v in out["primary"].items() if k != "signflip"}),
          "signflip ci", out["primary"]["signflip"]["ci"], "p", out["primary"]["signflip"]["p"])
    for r in rows:
        print(r)
    for b in bands:
        print(b)


if __name__ == "__main__":
    main()
