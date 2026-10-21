"""Score the GLM 5.3 Flash / DeepSeek 4.1 Flash System 2 runs on all 796 DECISION questions (PREREG.md here).
Each model separately: A0 (alone), A1 (+ System 1 handoff), cascade curve, frozen threshold rule with Opus's cost,
per-band gains; compared with Opus (runs/phase2/decision_rest/result.json).

  .venv/bin/python experiments/ollama_crosscheck/score.py --runs <dir with <model>/A0.jsonl, A1.jsonl> \
      --chunks <dir with A1/cNN.jsonl> --out experiments/ollama_crosscheck/result.json
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from cascade_threshold import load  # noqa: E402
from luthor.stats import cluster_signflip  # noqa: E402

import os
# data/raw is a link to an external drive; LUTHOR_TRAIN_DBS / LUTHOR_TRAIN_JSON point at identical copies when it is unmounted
DB = Path(os.environ.get("LUTHOR_TRAIN_DBS", ROOT / "data/raw/bird_train/train_databases"))
TRAIN = Path(os.environ.get("LUTHOR_TRAIN_JSON", ROOT / "data/raw/bird_train/train.json"))
OPUS = ROOT / "runs/phase2/decision_rest/result.json"
GRID = [2.0 + 0.25 * k for k in range(17)]
COST_OPUS, COST_S1 = 0.0278, 0.00064
BANDS = [("no margin", None, None), ("0-0.52", 0, 0.52), ("0.52-1.38", 0.52, 1.38), ("1.38-2.48", 1.38, 2.48),
         ("2.48-3.53", 2.48, 3.53), ("3.53-4.12", 3.53, 4.12), ("4.12-4.62", 4.12, 4.62), ("4.62-5.39", 4.62, 5.39),
         (">=5.39", 5.39, math.inf)]


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
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    s1, _ = load()
    ids = sorted(s1)
    tr = json.load(open(TRAIN))
    cq = {json.loads(l)["question_id"]: json.loads(l) for f in sorted((a.chunks / "A1").glob("*.jsonl")) for l in open(f)}
    gold = {i: tr[int(i.rsplit("_", 1)[1])]["SQL"] for i in ids}
    dbid = {i: cq[i]["db_id"] for i in ids}
    preds, usage = {}, {}
    for mdir in sorted(p for p in a.runs.iterdir() if p.is_dir()):
        for arm in ("A0", "A1"):
            f = mdir / f"{arm}.jsonl"
            ans, u = {}, {"n": 0, "tok_in": 0, "tok_out": 0, "sql_executions": 0, "unsubmitted": 0}
            for ln in open(f):
                r = json.loads(ln)
                q = r["question_id"]
                if q in ans:
                    continue
                u["n"] += 1
                for k in ("tok_in", "tok_out", "sql_executions"):
                    u[k] += r.get(k, 0)
                u["unsubmitted"] += not r.get("submitted")
                if arm == "A1" and str(r.get("pick", "")).upper() != "NONE":
                    ans[q] = {c["label"]: c["sql"] for c in cq[q]["candidates"]}.get(str(r["pick"]).strip())
                else:
                    ans[q] = r.get("sql") or None
            preds[(mdir.name, arm)], usage[(mdir.name, arm)] = ans, u
    tasks = {(dbid[i], s) for i in ids for s in [gold[i]] + [p.get(i) for p in preds.values()] if s}
    with ProcessPoolExecutor(8) as ex:
        res = dict(ex.map(run, sorted(tasks), chunksize=16))

    def ok(i, s):
        g, p = res.get((dbid[i], gold[i])), res.get((dbid[i], s)) if s else None
        return g is not None and p is not None and p == g

    S1 = {i: s1[i]["ok"] for i in ids}
    dbs = {i: s1[i]["db"] for i in ids}
    n = len(ids)
    opus = json.load(open(OPUS))
    out = {"system1_ex": round(100 * sum(S1.values()) / n, 2), "opus": {"always_handoff": opus["always_handoff_ex"],
           "t_star": opus["primary_t"], "curve": opus["curve"]}, "models": {}}
    for model in sorted({m for m, _ in preds}):
        H0 = {i: ok(i, preds[(model, "A0")].get(i)) for i in ids}
        H1 = {i: ok(i, preds[(model, "A1")].get(i)) for i in ids}
        ex1 = 100 * sum(H1.values()) / n
        curve, best = [], None
        for t in GRID:
            h = {i: (S1[i] if s1[i]["m"] >= t else H1[i]) for i in ids}
            share = sum(s1[i]["m"] < t for i in ids) / n
            ex = 100 * sum(h.values()) / n
            curve.append({"t": t, "ex": round(ex, 2), "opus_share": round(share, 4), "loss": round(ex - ex1, 2)})
            u = ex / 100 - (share * COST_OPUS + COST_S1)
            if best is None or u > best[1] + 1e-12:
                best = (t, u)
        within = next((c["t"] for c in curve if c["loss"] >= -0.5), None)
        bands = []
        for name, lo, hi in BANDS:
            sel = [i for i in ids if (s1[i]["m"] == -math.inf if lo is None else lo <= s1[i]["m"] < hi)]
            bands.append({"band": name, "n": len(sel), "s1": round(100 * sum(S1[i] for i in sel) / max(1, len(sel)), 1),
                          "handoff": round(100 * sum(H1[i] for i in sel) / max(1, len(sel)), 1)})
        h3 = {i: (S1[i] if s1[i]["m"] >= 3.0 else H1[i]) for i in ids}
        out["models"][model] = {
            "A0": round(100 * sum(H0.values()) / n, 2), "A1": round(ex1, 2),
            "A1_vs_A0": cluster_signflip(H0, H1, dbs), "A1_vs_S1": cluster_signflip(S1, H1, dbs),
            "t_star_opus_cost": best[0], "smallest_t_within_0.5": within,
            "t3": {**next(c for c in curve if c["t"] == 3.0), "signflip": cluster_signflip(H1, h3, dbs)},
            "curve": curve, "bands": bands,
            "usage": {arm: usage[(model, arm)] for arm in ("A0", "A1")}}
    a.out.write_text(json.dumps(out, indent=1, default=str))
    print("System 1", out["system1_ex"], "| Opus always-handoff", opus["always_handoff_ex"], "t*", opus["primary_t"])
    for m, r in out["models"].items():
        print(f"{m}: A0 {r['A0']} A1 {r['A1']} | t* {r['t_star_opus_cost']} | within0.5 {r['smallest_t_within_0.5']} | "
              f"t=3.0 {r['t3']['ex']} loss {r['t3']['loss']} share {r['t3']['opus_share']} | usage {r['usage']}")
        for b in r["bands"]:
            print("   ", b)


if __name__ == "__main__":
    main()
