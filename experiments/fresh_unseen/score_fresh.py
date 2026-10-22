"""Score the fresh unseen set (experiments/fresh_unseen/PREREG.md).

EX (read-only SQLite, 30 s, set of rows) of S1, B0 and C on the 227 verified items. C's routed half (margin < 3) vs
its kept half. Projection for BIRD test. C - B0 with an exact DB sign-flip (8 clusters). Cost of C vs B0 with the
cost tests' accounting (Opus 5.5 list prices, chars/4 floor on output, System 1 GPU cost on every question of C).

  .venv/bin/python experiments/fresh_unseen/score_fresh.py --box <s2fresh> --tasks <dir> --out experiments/fresh_unseen/result.json
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cost_confirmation"))
sys.path.insert(0, str(ROOT / "experiments/external_cascade"))
sys.path.insert(0, str(ROOT / "experiments/renamed_routed"))
sys.path.insert(0, str(ROOT / "src"))
import build  # noqa: E402  (execute)
import score  # noqa: E402  (dollars)
import score_ext  # noqa: E402  (_one: usage of one transcript)
from luthor.stats import cluster_signflip  # noqa: E402

DBS = ROOT / "data/sys1/fresh/dbs"
T = 3.0
S1_SEC_PER_Q = 313 / 227             # PREDICT_DONE: 227 predictions in 313 s on one H100
S1_COST_PER_Q = S1_SEC_PER_Q / 3600 * 2.73


def answers(g):
    out = {}
    for f in sorted(glob.glob(g)):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            out.setdefault(r.get("question_id"), r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    items = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/sys1/fresh/items_final.jsonl")}
    s1 = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "runs/phase2/fresh_s1/s1_fresh.jsonl")}
    assert set(s1) == set(items)
    ids = sorted(items)
    dbid = {i: items[i]["db_id"] for i in ids}
    routed = {i for i in ids if s1[i]["margin"] is None or s1[i]["margin"] < T}
    path = lambda i: DBS / dbid[i] / f"{dbid[i]}.sqlite"
    gold = {i: build.execute(path(i), items[i]["sql"]) for i in ids}
    ok = lambda i, sql: bool(sql) and gold[i] is not None and build.execute(path(i), sql) == gold[i]

    b0a = answers(str(a.box / "answers/B0/*.jsonl"))
    ca = answers(str(a.box / "answers/C/*.jsonl"))
    R = {"S1": {i: ok(i, s1[i]["sql"]) for i in ids}, "B0": {i: ok(i, (b0a.get(i) or {}).get("sql")) for i in ids}}
    agent_routed = {}
    for i in routed:
        r = ca.get(i) or {}
        p = str(r.get("pick", "NONE")).strip()
        cands = {c["label"]: c["sql"] for c in s1[i].get("candidates", [])}
        agent_routed[i] = ok(i, cands.get(p) if p.upper() != "NONE" else r.get("sql"))
    R["C"] = {i: agent_routed[i] if i in routed else R["S1"][i] for i in ids}
    missing = {"B0": sum(i not in b0a for i in ids), "C": sum(i not in ca for i in routed)}

    n = len(ids)
    ex = lambda arm, sel=ids: round(100 * sum(R[arm][i] for i in sel) / max(1, len(sel)), 2)
    kept = [i for i in ids if i not in routed]
    rt = sorted(routed)
    dbs = sorted(set(dbid.values()))
    rng = random.Random(20261021)

    def boot(f):
        v = []
        for _ in range(10000):
            smp = [rng.choice(dbs) for _ in dbs]
            v.append(f([i for d in smp for i in ids if dbid[i] == d]))
        v.sort()
        return [round(v[249], 2), round(v[9749], 2)]

    routed_ex = lambda sel: 100 * sum(R["C"][i] for i in sel if i in routed) / max(1, sum(i in routed for i in sel))
    share = len(rt) / n
    proj = lambda kept_acc, routed_acc: round((1 - share) * kept_acc + share * routed_acc, 2)
    # cost
    pat = re.compile(re.escape(str(a.box)) + r"/chunks/(B0|C)/([A-Za-z0-9_]+_\d+)\.jsonl")
    U = defaultdict(dict)
    for f in glob.glob(str(a.tasks / "*.output")):
        m = pat.search(open(f).read(20000))
        if m:
            U[m.group(1)][m.group(2)] = score_ext._one(f)
    cost_db = {arm: defaultdict(float) for arm in ("B0", "C")}
    for arm in ("B0", "C"):
        for ch, t in U[arm].items():
            cost_db[arm][ch.rsplit("_", 1)[0]] += score.dollars(t)
    for d in dbs:
        cost_db["C"][d] += S1_COST_PER_Q * sum(dbid[i] == d for i in ids)
    sav = lambda smp: 1 - sum(cost_db["C"][d] for d in smp) / sum(cost_db["B0"][d] for d in smp)
    sboot = sorted(sav([rng.choice(dbs) for _ in dbs]) for _ in range(10000))
    out = {
        "n": n, "routed": len(rt), "agent_share": round(share, 4), "missing": missing,
        "ex": {k: ex(k) for k in ("S1", "B0", "C")},
        "ex_ci95_dbboot": {k: boot(lambda sel, k=k: 100 * sum(R[k][i] for i in sel) / len(sel)) for k in ("S1", "B0", "C")},
        "routed_half": {"n": len(rt), "C_agent_ex": ex("C", rt), "ci95_dbboot": boot(routed_ex),
                        "S1_ex_on_routed": ex("S1", rt), "B0_ex_on_routed": ex("B0", rt),
                        "compare": {"RESERVED_run1": 72.32, "renamed": 72.88}},
        "kept_half": {"n": len(kept), "S1_ex": ex("S1", kept), "B0_ex_on_kept": ex("B0", kept)},
        "C_minus_B0": cluster_signflip(R["B0"], R["C"], dbid),
        "per_db": {d: {"n": sum(dbid[i] == d for i in ids), "routed": sum(dbid[i] == d for i in rt),
                       **{k: sum(R[k][i] for i in ids if dbid[i] == d) for k in ("S1", "B0", "C")}} for d in dbs},
        "projection_bird_test": {
            "fresh_measured": proj(ex("S1", kept), ex("C", rt)),
            "fresh_routed_with_reserved_kept": round(0.49 * 90.6 + 0.51 * ex("C", rt), 2),
            "reserved_based": 81.6,
            "note": "first: kept share/accuracy and routed accuracy all from this set; second: RESERVED's kept half "
                    "(share 0.49, accuracy 90.6) with this set's routed accuracy"},
        "cost": {"per_q": {arm: round(sum(cost_db[arm].values()) / n, 5) for arm in ("B0", "C")},
                 "saving_C_vs_B0": {"point": round(sav(dbs), 4), "ci95": [round(sboot[249], 4), round(sboot[9749], 4)]},
                 "chunks": {arm: len(U[arm]) for arm in ("B0", "C")}, "s1_cost_per_q": round(S1_COST_PER_Q, 6)},
    }
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
