"""Score the System 2 BIRD dev run (runs/phase2/system2_dev/PREREG.md). Strict EX (result sets equal, BIRD's rule).
Arms: S1 = System 1 alone (look #1's frozen pick: max VALHARD score over live candidates), A0 = Opus alone, A1 = Opus +
System 1 handoff on all, A2(t) = System 1 when its margin >= t else A1, A3(t) = System 1 when margin >= t else A0.
Golds: ORIGINAL dev (headline) and the 2025-11 corrected gold on the same question ids (original question text).
Also: the pre-declared memorisation test, exact gold-text match rates, Opus share, and the primary non-inferiority test.

  .venv/bin/python scripts/system2/score_dev.py --box <sandbox> --out runs/phase2/system2_dev/result.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from cascade_threshold import margin  # noqa: E402
from luthor.stats import cluster_signflip, percentile_bootstrap, question_signflip  # noqa: E402

DEV = ROOT / "data/bench/dev_20240627"
DEV_NEW = ROOT / "data/eval/dev6c_bird.jsonl"
POOL = ROOT / "runs/phase1/devlook/devlook/poolA_by_valhard.jsonl"
GRID = [2.0, 2.5, 3.0, 3.5]
PRIMARY_T, NONINF = 2.5, -1.5


def run(task):
    db, sql = task
    if not sql:
        return task, None
    con = sqlite3.connect(f"file:{DEV}/dev_databases/{db}/{db}.sqlite?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="replace")
    t = threading.Timer(60, con.interrupt)
    t.start()
    try:
        return task, frozenset(map(tuple, con.execute(sql).fetchall()))
    except Exception:
        return task, None
    finally:
        t.cancel()
        con.close()


def norm(s):
    return re.sub(r"\s+", " ", (s or "").replace("`", '"').strip().rstrip(";")).lower()


def load_answers(box, arm, chunks):
    ans = {}
    for f in sorted((box / "answers" / arm).glob("*.jsonl")):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            q = str(r.get("question_id"))
            if q in ans:
                continue  # first line per question counts
            if arm == "A0":
                ans[q] = r.get("sql") or None
            else:
                pick = str(r.get("pick", "")).strip()
                if pick.upper() == "NONE":
                    ans[q] = r.get("sql") or None
                else:
                    cand = {c["label"]: c["sql"] for c in chunks[q]["candidates"]}
                    ans[q] = cand.get(pick)  # a label not in the list -> None -> wrong
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="*", default=["A0", "A1"])
    a = ap.parse_args()
    q = json.load(open(DEV / "dev.json"))
    ids = [f"bird_dev_{i}" for i in range(len(q))]
    db = {i: q[k]["db_id"] for k, i in enumerate(ids)}
    gold = {i: q[k]["SQL"] for k, i in enumerate(ids)}
    new = {r["id"]: r for r in map(json.loads, open(DEV_NEW))}
    gold_new = {i: new[i]["gold_sql"] for i in ids}
    same_text = {i for k, i in enumerate(ids) if new[i]["question"].strip() == q[k]["question"].strip()
                 and (new[i].get("context") or "").strip() == (q[k].get("evidence") or "").strip()}
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
    preds = {"S1": s1}
    for arm in a.arms:
        preds[arm] = load_answers(a.box, arm, chunks)
    tasks = {(db[i], s) for i in ids for s in [gold[i], gold_new[i]] + [p.get(i) for p in preds.values()] if s}
    with ProcessPoolExecutor(8) as ex:
        res = dict(ex.map(run, sorted(tasks, key=str), chunksize=16))
    def ok(i, sql, g):
        gr = res.get((db[i], g))
        pr = res.get((db[i], sql)) if sql else None
        return pr is not None and gr is not None and pr == gr
    out = {"n": len(ids), "answered": {k: sum(1 for i in ids if v.get(i)) for k, v in preds.items()}}
    hits = {}
    for gname, G in (("original", gold), ("corrected", gold_new)):
        H = {arm: {i: ok(i, p.get(i), G[i]) for i in ids} for arm, p in preds.items()}
        for t in GRID:
            if "A1" in H:
                H[f"A2@{t}"] = {i: (H["S1"][i] if m[i] >= t else H["A1"][i]) for i in ids}
            if "A0" in H:
                H[f"A3@{t}"] = {i: (H["S1"][i] if m[i] >= t else H["A0"][i]) for i in ids}
        hits[gname] = H
        out[gname] = {k: round(100 * sum(v.values()) / len(ids), 2) for k, v in H.items()}
    out["opus_share"] = {f"{t}": round(sum(m[i] < t for i in ids) / len(ids), 4) for t in GRID}
    H = hits["original"]
    if "A1" in H:
        A2, A1 = H[f"A2@{PRIMARY_T}"], H["A1"]
        lo, hi = percentile_bootstrap(A1, A2, db)
        sf = cluster_signflip(A1, A2, db)  # exact DB sign-flip with its inverted CI (the repo's preferred interval)
        out["primary"] = {"comparison": f"A2@{PRIMARY_T} vs A1 (original gold)", "delta": round(out["original"][f"A2@{PRIMARY_T}"] - out["original"]["A1"], 2),
                          "db_bootstrap_95": [lo, hi], "signflip": sf, "mcnemar": question_signflip(A1, A2),
                          # non-inferior only if BOTH lower bounds clear the bar (bootstrap is anti-conservative at 11 DBs)
                          "noninferior": bool(lo >= NONINF and sf["ci"][0] >= NONINF)}
        for x, y in (("A0", "A1"), ("S1", "A1")):
            if x in H:
                out[f"{y}_vs_{x}"] = {"signflip": cluster_signflip(H[x], H[y], db), "mcnemar": question_signflip(H[x], H[y])}
    # memorisation test: text unchanged, gold result changed -> does the answer match the old or the new gold?
    M = [i for i in sorted(same_text) if res.get((db[i], gold[i])) != res.get((db[i], gold_new[i]))]
    mem = {"n_questions": len(M)}
    for arm, p in preds.items():
        mem[arm] = {"match_old_gold": sum(ok(i, p.get(i), gold[i]) for i in M), "match_new_gold": sum(ok(i, p.get(i), gold_new[i]) for i in M)}
    out["memorisation_test"] = mem
    out["exact_gold_text_match"] = {arm: round(100 * sum(norm(p.get(i)) == norm(gold[i]) for i in ids if p.get(i)) / max(1, sum(1 for i in ids if p.get(i))), 2)
                                    for arm, p in preds.items()}
    for arm in ("A0", "A1"):
        if arm in preds:
            nm = [i for i in ids if preds[arm].get(i) and norm(preds[arm][i]) != norm(gold[i])]
            out.setdefault("ex_on_non_matching_text", {})[arm] = {"n": len(nm), "ex": round(100 * sum(H[arm][i] for i in nm) / max(1, len(nm)), 2)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else str(x)))
    print(json.dumps({k: out[k] for k in ("answered", "original", "corrected", "opus_share") if k in out}, indent=1))
    if "primary" in out:
        print("PRIMARY", json.dumps({k: v for k, v in out["primary"].items() if k in ("comparison", "delta", "db_bootstrap_95", "noninferior")}))
    print("MEMORISATION", json.dumps(mem))


if __name__ == "__main__":
    main()
