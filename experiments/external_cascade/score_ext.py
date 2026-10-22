"""Score the external cascade test (experiments/external_cascade/PREREG.md): KaggleDBQA test + Spider-Realistic.

Same accuracy, cost and time accounting as experiments/cost_confirmation/score.py, on the sealed gold. Primary
endpoints are pooled over both sets (27 database clusters); each set and KaggleDBQA's clean 130 are reported too.
Above 16 clusters, luthor.stats.cluster_signflip uses 200,000 random sign flips (seed 0) instead of exact enumeration.

  LUTHOR_TRAIN_DBS=data/sys1/sealed/dbs .venv/bin/python experiments/external_cascade/score_ext.py --box <sandbox> \
      --tasks <dir> --s1 runs/phase2/external_s1/s1_external.jsonl --s1-cost-per-q <$> --s1-sec-per-q <s> --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("LUTHOR_TRAIN_DBS", str(ROOT / "data/sys1/sealed/dbs"))
sys.path.insert(0, str(ROOT / "experiments/cost_confirmation"))
sys.path.insert(0, str(ROOT / "src"))
import score  # noqa: E402
from luthor.stats import cluster_signflip  # noqa: E402

ARMS = ("B0", "B1", "C")
SETS = ("kaggledbqa_test", "spider_realistic")
DIRTY = {"WhatCDHipHop", "TheHistoryofBaseball"}  # schema overlap with BIRD/Spider train (EVAL_LEDGER.md)
T = 3.0


def usage(tasks_dir: Path, box: Path):
    pat = re.compile(re.escape(str(box)) + r"/chunks/(B0|B1|C)/([A-Za-z0-9_]+_\d+)\.jsonl")
    out = defaultdict(dict)
    for f in glob.glob(str(tasks_dir / "*.output")):
        m = pat.search(open(f).read(20000))
        if not m:
            continue
        arm, chunk = m.groups()
        if chunk in out[arm]:
            raise SystemExit(f"two transcripts for {arm}/{chunk}")
        out[arm][chunk] = _one(f)
    return out


def _one(f):
    from datetime import datetime
    tok, chars, seen, stamps = defaultdict(int), 0, set(), []
    for ln in open(f):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("timestamp"):
            stamps.append(r["timestamp"])
        msg = r.get("message") or {}
        if r.get("type") != "assistant" or not isinstance(msg, dict):
            continue
        for b in msg.get("content") or []:
            if isinstance(b, dict):
                chars += len(b.get("text", "")) + len(b.get("thinking", ""))
                if b.get("type") == "tool_use":
                    chars += len(json.dumps(b.get("input")))
        if msg.get("id") in seen:
            continue
        seen.add(msg.get("id"))
        for k in list(score.PRICE) + ["output_tokens"]:
            tok[k] += (msg.get("usage") or {}).get(k, 0) or 0
    tok["output_est"] = chars // 4
    if stamps:
        ts = sorted(datetime.fromisoformat(x.replace("Z", "+00:00")) for x in stamps)
        tok["seconds"] = (ts[-1] - ts[0]).total_seconds()
    return dict(tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--s1", type=Path, required=True)
    ap.add_argument("--s1-cost-per-q", type=float, required=True)
    ap.add_argument("--s1-sec-per-q", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    q = {json.loads(l)["id"]: json.loads(l) for s in SETS for l in open(ROOT / f"data/eval/sealed_{s}.jsonl")}
    ids = sorted(q)
    gold = {i: q[i]["gold_sql"] for i in ids}
    dbid = {i: q[i]["db_id"] for i in ids}
    sset = {i: q[i]["set"] for i in ids}
    s1 = {json.loads(l)["id"]: json.loads(l) for l in open(a.s1)}
    assert set(s1) == set(ids)
    routed = {i for i in ids if s1[i]["margin"] is None or s1[i]["margin"] < T}
    cands = {i: {c["label"]: c["sql"] for c in s1[i].get("candidates", [])} for i in ids}
    preds = {}
    for arm in ARMS:
        ans = {}
        for f in sorted(glob.glob(str(a.box / f"answers/{arm}/*.jsonl"))):
            for ln in open(f):
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                qi = r.get("question_id")
                if qi in ans or qi not in dbid:
                    continue
                p = str(r.get("pick", "NONE")).strip()
                ans[qi] = cands[qi].get(p) if arm != "B0" and p.upper() != "NONE" else (r.get("sql") or None)
        if arm == "C":
            assert not (set(ans) - routed), "cascade answered a non-routed question"
            for i in ids:
                if i not in routed:
                    ans[i] = s1[i]["sql"]
        preds[arm] = ans
    tasks = {(dbid[i], s) for i in ids for s in [gold[i], s1[i]["sql"]] + [p.get(i) for p in preds.values()]
             + [c["sql"] for c in s1[i].get("candidates", [])] if s}
    res = dict(map(score.run, sorted(tasks)))

    def ok(i, s):
        g = res.get((dbid[i], gold[i]))
        return bool(s) and g is not None and res.get((dbid[i], s)) == g

    R = {arm: {i: ok(i, preds[arm].get(i)) for i in ids} for arm in ARMS}
    R["S1"] = {i: ok(i, s1[i]["sql"]) for i in ids}
    U = usage(a.tasks, a.box)
    cost_db = {arm: defaultdict(float) for arm in ARMS}
    time_db = {arm: defaultdict(float) for arm in ARMS}
    for arm in ARMS:
        for chunk, t in U.get(arm, {}).items():
            d = chunk.rsplit("_", 1)[0]
            cost_db[arm][d] += score.dollars(t)
            time_db[arm][d] += t.get("seconds", 0.0)
    dbs_all = sorted(set(dbid.values()))
    for d in dbs_all:
        n = sum(dbid[i] == d for i in ids)
        for arm in ("B1", "C"):
            cost_db[arm][d] += a.s1_cost_per_q * n
            time_db[arm][d] += a.s1_sec_per_q * n
    rng = random.Random(20261021)

    def block(sel_ids):
        dbs = sorted({dbid[i] for i in sel_ids})
        n = len(sel_ids)
        sub = lambda arm: {i: R[arm][i] for i in sel_ids}
        o = {"n": n, "dbs": len(dbs), "routed": sum(i in routed for i in sel_ids),
             "ex": {k: round(100 * sum(R[k][i] for i in sel_ids) / n, 2) for k in R},
             "cost_per_q": {arm: round(sum(cost_db[arm][d] for d in dbs) / n, 5) for arm in ARMS},
             "time_per_q": {arm: round(sum(time_db[arm][d] for d in dbs) / n, 2) for arm in ARMS}}
        o["agent_share"] = round(o["routed"] / n, 4)
        for base in ("B1", "B0"):
            for kind, tab in (("saving", cost_db), ("time_saving", time_db)):
                f = lambda smp, b=base, t_=tab: 1 - sum(t_["C"][d] for d in smp) / sum(t_[b][d] for d in smp)
                boot = sorted(f([rng.choice(dbs) for _ in dbs]) for _ in range(10000))
                o[f"{kind}_vs_{base}"] = {"point": round(f(dbs), 4), "ci95": [round(boot[249], 4), round(boot[9749], 4)]}
            o[f"C_minus_{base}"] = cluster_signflip(sub(base), sub("C"), dbid)
            o[f"C_vs_{base}_wins_losses"] = [sum(R["C"][i] and not R[base][i] for i in sel_ids), sum(R[base][i] and not R["C"][i] for i in sel_ids)]
        o["B1_minus_B0"] = cluster_signflip(sub("B0"), sub("B1"), dbid)
        return o

    out = {"pooled": block(ids)}
    for s in SETS:
        out[s] = block([i for i in ids if sset[i] == s])
    out["kaggledbqa_clean130"] = block([i for i in ids if sset[i] == "kaggledbqa_test" and dbid[i] not in DIRTY])
    out["chunks"] = {arm: len(U.get(arm, {})) for arm in ARMS}
    out["missing"] = {arm: len(set(ids) - set(preds[arm])) for arm in ARMS}
    P = out["pooled"]
    out["pass"] = {"cost": bool(P["saving_vs_B1"]["point"] >= 0.30 and P["saving_vs_B1"]["ci95"][0] >= 0.20),
                   "noninferior": bool(P["C_minus_B1"]["ci"][0] >= -2.5),
                   "time": bool(P["time_saving_vs_B1"]["point"] >= 0.25 and P["time_saving_vs_B1"]["ci95"][0] >= 0.15)}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    for k in ("pooled",) + SETS + ("kaggledbqa_clean130",):
        b = out[k]
        print(k, json.dumps({x: b[x] for x in ("n", "agent_share", "ex", "cost_per_q", "saving_vs_B1", "saving_vs_B0", "time_saving_vs_B1", "time_saving_vs_B0")}, default=float))
        print("   C-B1", round(b["C_minus_B1"]["delta"], 2), [round(x, 2) for x in b["C_minus_B1"]["ci"]], "C-B0", round(b["C_minus_B0"]["delta"], 2), [round(x, 2) for x in b["C_minus_B0"]["ci"]])
    print("missing", out["missing"], "chunks", out["chunks"], "pass", out["pass"])


if __name__ == "__main__":
    main()
