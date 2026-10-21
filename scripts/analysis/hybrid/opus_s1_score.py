"""Opus 5.5 alone vs Opus 5.5 + System 1 candidates on the same 400 DECISION questions (strict EX by execution vs the
original BIRD-train gold), plus per-arm agent usage (tokens, tool calls, wall time) from the subagent notices.

  .venv/bin/python scripts/analysis/hybrid/opus_s1_score.py <hyb_opus dir> --json runs/analysis/hybrid/opus_s1.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sqlite3
import statistics
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "data/raw/bird_train/train_databases"


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
    qid, db, gold, preds = a
    g = run(db, gold)
    return qid, [g is not None and run(db, p) == g for p in preds]


def answers(d: Path):
    out = {}
    for f in sorted(glob.glob(str(d / "answers/*.jsonl"))):
        for ln in open(f):
            try:
                r = json.loads(ln)
                out[str(r["question_id"])] = r
            except Exception:
                pass
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hyb", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    ids = json.load(open(a.hyb / "sample_ids.json"))
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    cands = {}
    for f in glob.glob(str(a.hyb / "s1/chunks/*.jsonl")):
        for ln in open(f):
            r = json.loads(ln)
            cands[r["question_id"]] = {c["label"]: c["sql"] for c in r["candidates"]}
    al, s1 = answers(a.hyb / "alone"), answers(a.hyb / "s1")
    s1_sql, picked = {}, {}
    for i in ids:
        r = s1.get(i, {})
        p = str(r.get("pick", "NONE"))
        if p != "NONE" and p in cands.get(i, {}):
            s1_sql[i], picked[i] = cands[i][p], True
        else:
            s1_sql[i], picked[i] = r.get("sql"), False
    first = {i: (cands[i].get("A") if cands.get(i) else None) for i in ids}   # System 1's own pick (top class)
    jobs = [(i, tr[int(i.rsplit("_", 1)[1])]["db_id"], tr[int(i.rsplit("_", 1)[1])]["SQL"],
             [al.get(i, {}).get("sql"), s1_sql[i], first[i]]) for i in ids]
    with ProcessPoolExecutor(6) as ex:
        res = dict(ex.map(job, jobs, chunksize=4))
    n = len(ids)
    acc = lambda k: 100 * sum(res[i][k] for i in ids) / n  # noqa: E731
    b = sum(1 for i in ids if res[i][1] and not res[i][0])
    c = sum(1 for i in ids if res[i][0] and not res[i][1])
    use = [json.loads(l) for l in open(a.hyb / "usage.jsonl")]
    U = {}
    for arm in ("alone", "s1"):
        u = [x for x in use if x["arm"] == arm]
        U[arm] = {"chunks": len(u), "tokens": sum(x["tokens"] for x in u), "tool_uses": sum(x["tool_uses"] for x in u),
                  "agent_seconds": sum(x["ms"] for x in u) / 1000,
                  "median_chunk_seconds": statistics.median(x["ms"] for x in u) / 1000 if u else None}
    out = {"n": n, "answered": {"alone": sum(i in al for i in ids), "s1": sum(i in s1 for i in ids)},
           "ex": {"opus_alone": round(acc(0), 2), "opus_plus_s1": round(acc(1), 2), "system1_alone": round(acc(2), 2)},
           "plus_s1_vs_alone": {"wins": b, "losses": c, "mcnemar_p": mcnemar(b, c)},
           "picked_candidate_share": round(sum(picked.values()) / n, 3),
           "acc_when_picked": round(100 * sum(res[i][1] for i in ids if picked[i]) / max(1, sum(picked.values())), 1),
           "acc_when_own": round(100 * sum(res[i][1] for i in ids if not picked[i]) / max(1, n - sum(picked.values())), 1),
           "usage": U}
    # System 1 skips the LLM when confident: VALHARD top-2 result-class margin (DECISION rows only), official pick
    idset, s1ok, marg = set(ids), {}, {}
    for ln in open(ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl"):
        r = json.loads(ln)
        if r["id"] not in idset:
            continue
        c = [x for x in r["cands"] if x.get("sql") and x.get("src") in ("r1", "both", None)]
        lv = [x for x in c if x.get("runs") and x.get("nonnull")] or c
        if not lv:
            continue
        s1ok[r["id"]] = bool(max(lv, key=lambda x: x.get("rr", -1e9)).get("ok"))
        best = {}
        for x in lv:
            k = x.get("sig") or x["sql"]
            best[k] = max(best.get(k, -1e9), x["rr"])
        t = sorted(best.values(), reverse=True)
        marg[r["id"]] = t[0] - t[1] if len(t) > 1 else 1e6
    order = sorted(ids, key=lambda i: -marg.get(i, -1e9))   # most confident first; no candidate = never skipped
    out["skip_curve"] = []
    for f in (0.0, 0.25, 0.5, 0.75):
        skip = set(order[:round(f * n)])
        out["skip_curve"].append({
            "skip_share": f, "llm_share": 1 - f,
            "ex_plus_s1": round(100 * sum(s1ok.get(i, False) if i in skip else res[i][1] for i in ids) / n, 2),
            "ex_alone_rest": round(100 * sum(s1ok.get(i, False) if i in skip else res[i][0] for i in ids) / n, 2),
            "s1_acc_on_skipped": round(100 * sum(s1ok.get(i, False) for i in skip) / max(1, len(skip)), 1)})
    out["per_question"] = {i: {"alone": res[i][0], "plus_s1": res[i][1], "s1": s1ok.get(i, False),
                               "picked": picked[i]} for i in ids}
    print(json.dumps({k: v for k, v in out.items() if k != "per_question"}, indent=1))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
