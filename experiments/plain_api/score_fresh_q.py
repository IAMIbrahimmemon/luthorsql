"""Score the question-only plain-API cascade on the fresh set (experiments/plain_api/PREREG_fresh_question_only.md).

  python3 experiments/plain_api/score_fresh_q.py --run runs/phase2/plain_api/fresh_q --out experiments/plain_api/result_fresh_q.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for d in ("experiments/cost_confirmation", "experiments/external_cascade", "experiments/renamed_routed", "src"):
    sys.path.insert(0, str(ROOT / d))
import build  # noqa: E402  (the fresh set's executor, as in experiments/fresh_unseen/score_fresh.py)
from luthor.stats import cluster_signflip  # noqa: E402

import os  # noqa: E402


def _first(*ps):  # development layout first, then the public release layout
    return next((p for p in ps if p.exists()), ps[-1])


DBS = Path(os.environ.get("LUTHOR_FRESH_DBS", ROOT / "data/sys1/fresh/dbs"))  # or the HF dataset's databases/
ITEMS = _first(ROOT / "data/sys1/fresh/items_final.jsonl", ROOT / "splits/fresh_unseen/items.jsonl")
S1 = _first(ROOT / "runs/phase2/fresh_s1/s1_fresh.jsonl", ROOT / "experiments/fresh_unseen/artifacts/s1_fresh.jsonl")
T = 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    items = {json.loads(l)["id"]: json.loads(l) for l in open(ITEMS)}
    s1 = {json.loads(l)["id"]: json.loads(l) for l in open(S1)}
    assert set(s1) == set(items)
    ids = sorted(items)
    dbid = {i: items[i]["db_id"] for i in ids}
    routed = {i for i in ids if s1[i]["margin"] is None or s1[i]["margin"] < T}
    path = lambda i: DBS / dbid[i] / f"{dbid[i]}.sqlite"
    gold = {i: build.execute(path(i), items[i]["sql"]) for i in ids}
    ok = lambda i, sql: bool(sql) and gold[i] is not None and build.execute(path(i), sql) == gold[i]
    A0 = {}
    for l in open(a.run / "answers_A0.jsonl"):
        r = json.loads(l)
        A0.setdefault(r["question_id"], r)
    R = {"A0": {i: ok(i, (A0.get(i) or {}).get("sql")) for i in ids}, "S1": {i: ok(i, s1[i]["sql"]) for i in ids}}
    R["Q"] = {i: (R["A0"][i] if i in routed else R["S1"][i]) for i in ids}
    dbs = sorted(set(dbid.values()))
    cost = {k: defaultdict(float) for k in ("A0", "Q")}
    for i in ids:
        c = (A0.get(i) or {}).get("dollars", 0.0)
        cost["A0"][dbid[i]] += c
        cost["Q"][dbid[i]] += c if i in routed else 0.0
    sv = lambda smp: 1 - sum(cost["Q"][d] for d in smp) / sum(cost["A0"][d] for d in smp)
    rng = random.Random(20261026)
    boot = sorted(sv([rng.choice(dbs) for _ in dbs]) for _ in range(10000))
    n, kept = len(ids), [i for i in ids if i not in routed]
    tok = defaultdict(int)
    for r in A0.values():
        for k, v in r["usage"].items():
            tok[k] += v
    out = {"prereg": "experiments/plain_api/PREREG_fresh_question_only.md", "n": n, "routed": len(routed),
           "missing": len(set(ids) - set(A0)), "not_submitted": sum(not r["submitted"] for r in A0.values()),
           "ex": {k: round(100 * sum(v.values()) / n, 2) for k, v in R.items()},
           "ex_routed": {k: round(100 * sum(R[k][i] for i in routed) / len(routed), 2) for k in ("A0", "S1")},
           "ex_kept": {k: round(100 * sum(R[k][i] for i in kept) / len(kept), 2) for k in ("A0", "S1")},
           "cost_per_q": {k: round(sum(v.values()) / n, 5) for k, v in cost.items()},
           "saving_Q_vs_A0": {"point": round(sv(dbs), 4), "ci95": [round(boot[249], 4), round(boot[9749], 4)]},
           "Q_minus_A0": cluster_signflip(R["A0"], R["Q"], dbid),
           "tokens": dict(tok), "dollars_total": round(sum(r["dollars"] for r in A0.values()), 4)}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("ex", "ex_kept", "cost_per_q", "saving_Q_vs_A0", "missing", "dollars_total")}))
    print("Q-A0", {k: out["Q_minus_A0"][k] for k in ("delta", "p", "ci")})


if __name__ == "__main__":
    main()
