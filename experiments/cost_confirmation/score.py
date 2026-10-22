"""Score the pre-registered cost-confirmation runs on RESERVED (experiments/cost_confirmation/PREREG.md).

Accuracy: strict EX vs the original BIRD-train gold (read-only SQLite, 30 s, set of rows), per arm.
Cost: each arm's Claude Code subagent usage, summed per database (chunks never mix databases), at Opus 5.5 list prices,
plus System 1's measured GPU cost on every question of the cascade arm.

  .venv/bin/python experiments/cost_confirmation/score.py --box <sandbox> --tasks <subagent transcript dir> \
      --s1 runs/phase2/cost_confirmation/s1_reserved.jsonl --s1-cost-per-q <dollars> --out experiments/cost_confirmation/result.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sqlite3
import sys
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luthor.stats import cluster_signflip  # noqa: E402

DB = Path(os.environ.get("LUTHOR_TRAIN_DBS", ROOT / "data/raw/bird_train/train_databases"))
TRAIN = Path(os.environ.get("LUTHOR_TRAIN_JSON", ROOT / "data/raw/bird_train/train.json"))
PRICE = {"input_tokens": 4.0, "cache_creation_input_tokens": 5.0, "cache_read_input_tokens": 0.20}  # $/M, Opus 5.5
PRICE_OUT = 20.0
ARMS = ("B0", "B1", "B2", "C")


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


def usage(tasks_dir: Path, box: Path):
    """Per arm, per chunk: summed token usage of the subagent whose prompt names that chunk file."""
    pat = re.compile(re.escape(str(box)) + r"/chunks/(B0|B1|B2|C)/([a-z_]+_\d+)\.jsonl")
    out = defaultdict(dict)
    for f in glob.glob(str(tasks_dir / "*.output")):
        head = open(f).read(20000)
        m = pat.search(head)
        if not m:
            continue
        arm, chunk = m.groups()
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
            for k in list(PRICE) + ["output_tokens"]:
                tok[k] += (msg.get("usage") or {}).get(k, 0) or 0
        tok["output_est"] = chars // 4
        if stamps:
            from datetime import datetime
            ts = sorted(datetime.fromisoformat(x.replace("Z", "+00:00")) for x in stamps)
            tok["seconds"] = (ts[-1] - ts[0]).total_seconds()
        if chunk in out[arm]:
            raise SystemExit(f"two transcripts for {arm}/{chunk}: resolve before scoring")
        out[arm][chunk] = dict(tok)
    return out


def dollars(t):
    return sum(t.get(k, 0) * v for k, v in PRICE.items()) / 1e6 + max(t.get("output_tokens", 0), t.get("output_est", 0)) * PRICE_OUT / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--s1", type=Path, required=True, help="System 1 rows on RESERVED: id, pick sql, margin, candidates")
    ap.add_argument("--s1-cost-per-q", type=float, required=True)
    ap.add_argument("--s1-sec-per-q", type=float, required=True, help="System 1 seconds per question (batched GPU run)")
    ap.add_argument("--t", type=float, default=3.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tr = json.load(open(TRAIN))
    ids = json.load(open(ROOT / "data/sys1/heldout_v2/reserved.json"))["ids"]
    gold = {i: tr[int(i.rsplit("_", 1)[1])]["SQL"] for i in ids}
    dbid = {i: tr[int(i.rsplit("_", 1)[1])]["db_id"] for i in ids}
    s1 = {json.loads(l)["id"]: json.loads(l) for l in open(a.s1)}
    assert set(s1) == set(ids), "System 1 rows must cover all RESERVED ids"
    routed = {i for i in ids if s1[i]["margin"] is None or s1[i]["margin"] < a.t}
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
                q = r.get("question_id")
                if q in ans or q not in dbid:
                    continue
                p = str(r.get("pick", "NONE")).strip()
                ans[q] = cands[q].get(p) if arm in ("B1", "C") and p.upper() != "NONE" else (r.get("sql") or None)
        if arm == "C":
            extra = set(ans) - routed
            assert not extra, f"cascade arm answered non-routed questions: {sorted(extra)[:3]}"
            for i in ids:
                if i not in routed:
                    ans[i] = s1[i]["sql"]
        preds[arm] = ans
    missing = {arm: sorted(set(ids) - set(p)) for arm, p in preds.items()}
    tasks = {(dbid[i], s) for i in ids for s in [gold[i]] + [p.get(i) for p in preds.values()] + [s1[i]["sql"]]
             + [c["sql"] for c in s1[i].get("candidates", [])] if s}
    with ProcessPoolExecutor(8) as ex:
        res = dict(ex.map(run, sorted(tasks), chunksize=16))

    def ok(i, s):
        g, p = res.get((dbid[i], gold[i])), res.get((dbid[i], s)) if s else None
        return g is not None and p is not None and p == g

    R = {arm: {i: ok(i, preds[arm].get(i)) for i in ids} for arm in ARMS}
    R["S1"] = {i: ok(i, s1[i]["sql"]) for i in ids}
    U = usage(a.tasks, a.box)
    dbs = sorted(set(dbid.values()))
    cost_db = {arm: defaultdict(float) for arm in ARMS}
    for arm in ARMS:
        for chunk, t in U.get(arm, {}).items():
            cost_db[arm][chunk.rsplit("_", 1)[0]] += dollars(t)
    for d in dbs:  # System 1 runs on every question in B1 (to make the candidates) and in C (to route); not in B0
        for arm in ("B1", "C"):
            cost_db[arm][d] += a.s1_cost_per_q * sum(1 for i in ids if dbid[i] == d)
    nq = {d: sum(1 for i in ids if dbid[i] == d) for d in dbs}

    def saving(base, arm, sample):
        cb, ca = sum(cost_db[base][d] for d in sample), sum(cost_db[arm][d] for d in sample)
        return 1 - ca / cb if cb else float("nan")

    rng = random.Random(20261020)
    out = {"n": len(ids), "routed": len(routed), "agent_share": round(len(routed) / len(ids), 4),
           "missing": {k: len(v) for k, v in missing.items()},
           "ex": {k: round(100 * sum(v.values()) / len(ids), 2) for k, v in R.items()},
           "cost_per_q": {arm: round(sum(cost_db[arm].values()) / len(ids), 5) for arm in ARMS},
           "tokens": {arm: {k: sum(t.get(k, 0) for t in U.get(arm, {}).values()) for k in list(PRICE) + ["output_tokens", "output_est"]} for arm in ARMS},
           "chunks": {arm: len(U.get(arm, {})) for arm in ARMS}}
    for base in ("B1", "B0"):
        point = saving(base, "C", dbs)
        boot = sorted(saving(base, "C", [rng.choice(dbs) for _ in dbs]) for _ in range(10000))
        out[f"saving_vs_{base}"] = {"point": round(point, 4), "ci95": [round(boot[249], 4), round(boot[9749], 4)]}
        out[f"C_minus_{base}"] = cluster_signflip(R[base], R["C"], dbid)
    out["B1_minus_B0"] = cluster_signflip(R["B0"], R["B1"], dbid)
    out["B1_minus_B2"] = cluster_signflip(R["B2"], R["B1"], dbid)
    out["B2_minus_B0"] = cluster_signflip(R["B0"], R["B2"], dbid)
    # end-to-end time: summed agent seconds per database + System 1 seconds where it runs (B1, C)
    time_db = {arm: defaultdict(float) for arm in ARMS}
    for arm in ARMS:
        for chunk, t in U.get(arm, {}).items():
            time_db[arm][chunk.rsplit("_", 1)[0]] += t.get("seconds", 0.0)
    for d in dbs:
        for arm in ("B1", "C"):
            time_db[arm][d] += a.s1_sec_per_q * nq[d]
    out["time_per_q"] = {arm: round(sum(time_db[arm].values()) / len(ids), 2) for arm in ARMS}
    for base in ("B1", "B0"):
        f = lambda smp, b=base: 1 - sum(time_db["C"][d] for d in smp) / sum(time_db[b][d] for d in smp)
        boot = sorted(f([rng.choice(dbs) for _ in dbs]) for _ in range(10000))
        out[f"time_saving_vs_{base}"] = {"point": round(f(dbs), 4), "ci95": [round(boot[249], 4), round(boot[9749], 4)]}
    # candidate effect split by whether any shown candidate is right
    right = {i: any(ok(i, c["sql"]) for c in s1[i].get("candidates", [])) for i in ids}
    out["split_by_right_candidate"] = {}
    for name, sel in (("right_candidate_shown", [i for i in ids if right[i]]), ("no_right_candidate", [i for i in ids if not right[i]])):
        e = {arm: round(100 * sum(R[arm][i] for i in sel) / max(1, len(sel)), 2) for arm in ("B0", "B1", "B2")}
        w = lambda x, y: (sum(R[y][i] and not R[x][i] for i in sel), sum(R[x][i] and not R[y][i] for i in sel))
        out["split_by_right_candidate"][name] = {"n": len(sel), "ex": e, "B1_vs_B0_wins_losses": w("B0", "B1"), "B1_vs_B2_wins_losses": w("B2", "B1")}
    out["cost_per_db"] = {arm: {d: round(cost_db[arm][d], 4) for d in dbs} for arm in ARMS}
    out["questions_per_db"] = nq
    out["pass"] = {"cost": out["saving_vs_B1"]["point"] >= 0.30 and out["saving_vs_B1"]["ci95"][0] >= 0.20,
                   "noninferior": out["C_minus_B1"]["ci"][0] >= -2.5,
                   "time": out["time_saving_vs_B1"]["point"] >= 0.25 and out["time_saving_vs_B1"]["ci95"][0] >= 0.15}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("agent_share", "missing", "ex", "cost_per_q", "saving_vs_B1", "saving_vs_B0", "time_per_q", "time_saving_vs_B1", "time_saving_vs_B0", "pass")}, indent=1, default=float))


if __name__ == "__main__":
    main()
