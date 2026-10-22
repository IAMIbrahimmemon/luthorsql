"""Pre-registered secondaries for the external test: routing baselines at the cascade's share (scored with B1's answers)
and price sensitivity. Same definitions as experiments/cost_confirmation/secondary.py.

  .venv/bin/python experiments/external_cascade/secondary_ext.py --box <sandbox> --tasks <dir> --s1 <s1_external.jsonl> --out <json>
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cost_confirmation"))
import score_ext as se  # noqa: E402
import score  # noqa: E402
from secondary import PRICES, cost  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("--box", "--tasks", "--s1", "--out"):
    ap.add_argument(k, type=Path, required=True)
ap.add_argument("--s1-cost-per-q", type=float, default=0.000664)
a = ap.parse_args()
q = {json.loads(l)["id"]: json.loads(l) for s in se.SETS for l in open(se.ROOT / f"data/eval/sealed_{s}.jsonl")}
ids = sorted(q)
s1 = {json.loads(l)["id"]: json.loads(l) for l in open(a.s1)}
cands = {i: {c["label"]: c["sql"] for c in s1[i]["candidates"]} for i in ids}
b1 = {}
for f in glob.glob(str(a.box / "answers/B1/*.jsonl")):
    for ln in open(f):
        r = json.loads(ln)
        if r["question_id"] not in b1:
            p = str(r.get("pick", "NONE")).strip()
            b1[r["question_id"]] = cands[r["question_id"]].get(p) if p.upper() != "NONE" else r.get("sql")
db = {i: q[i]["db_id"] for i in ids}
res = dict(map(score.run, sorted({(db[i], s) for i in ids for s in (q[i]["gold_sql"], b1.get(i), s1[i]["sql"]) if s})))
ok = lambda i, s: bool(s) and res.get((db[i], q[i]["gold_sql"])) is not None and res.get((db[i], s)) == res[(db[i], q[i]["gold_sql"])]
A = {i: ok(i, b1.get(i)) for i in ids}
S = {i: ok(i, s1[i]["sql"]) for i in ids}
out = {}
for name, sel in (("pooled", ids),) + tuple((s, [i for i in ids if q[i]["set"] == s]) for s in se.SETS):
    routed = {i for i in sel if s1[i]["margin"] is None or s1[i]["margin"] < se.T}
    k, n = len(routed), len(sel)
    ex = lambda r: round(100 * sum(A[i] if i in r else S[i] for i in sel) / n, 2)
    agree = {i: s1[i]["candidates"][0]["n_of_32"] / max(1, sum(c["n_of_32"] for c in s1[i]["candidates"])) if s1[i]["candidates"] else 0 for i in sel}
    out[name] = {"share": round(k / n, 4), "margin": ex(routed),
                 "vote_agreement": ex(set(sorted(sel, key=lambda i: (agree[i], i))[:k])),
                 "question_length": ex(set(sorted(sel, key=lambda i: (-len(q[i]["question"]), i))[:k])),
                 "random_expected": round(100 * (sum(S[i] for i in sel) + k / n * sum(A[i] - S[i] for i in sel)) / n, 2),
                 "all_B1": ex(set(sel)), "system1": ex(set())}
U = se.usage(a.tasks, a.box)
out["price"] = {}
for nm, p in PRICES.items():
    t = {arm: sum(cost(x, p) for x in U[arm].values()) for arm in ("B0", "B1", "C")}
    s1c = a.s1_cost_per_q * len(ids)
    out["price"][nm] = {"saving_vs_B1": round(1 - (t["C"] + s1c) / (t["B1"] + s1c), 4), "saving_vs_B0": round(1 - (t["C"] + s1c) / t["B0"], 4)}
a.out.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
