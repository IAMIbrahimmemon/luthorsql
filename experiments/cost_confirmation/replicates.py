"""Amendment 2 (PREREG.md): three runs of every RESERVED arm. Reporting only; the run-1 verdict stands.

  .venv/bin/python experiments/cost_confirmation/replicates.py --boxes <r1> <r2> <r3> --tasks <dir> --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402
from luthor.stats import _scan, _signs  # noqa: E402
import numpy as np  # noqa: E402


def signflip_frac(a: dict, b: dict, db: dict, alpha=0.05, step=0.0001):
    """luthor.stats.cluster_signflip for per-question scores in [0, 1] (the mean over runs); it casts to bool."""
    keys = sorted(set(db[i] for i in a))
    D = np.array([sum(b[i] - a[i] for i in a if db[i] == k) for k in keys])
    n = np.array([sum(1 for i in a if db[i] == k) for k in keys], float)
    S = _signs(len(D))
    obs = D.sum()
    p = float(np.mean(np.abs(S @ D) >= abs(obs) - 1e-9))
    pval = lambda d0: np.mean(np.abs(S @ (D - n * d0)) >= abs((D - n * d0).sum()) - 1e-9)
    est = obs / n.sum()
    inside = _scan(np.arange(-1.0, 1.0 + step / 2, step), est, pval, alpha)
    return {"delta": 100 * est, "p": p, "ci": (100 * min(inside), 100 * max(inside)), "G": len(D)}

ARMS = ("B0", "B1", "B2", "C")
S1_COST, T = 0.000849, 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", type=Path, nargs=3, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--s1", type=Path, default=score.ROOT / "runs/phase2/cost_confirmation/s1_reserved.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tr = json.load(open(score.TRAIN))
    ids = json.load(open(score.ROOT / "data/sys1/heldout_v2/reserved.json"))["ids"]
    gold = {i: tr[int(i.rsplit("_", 1)[1])]["SQL"] for i in ids}
    dbid = {i: tr[int(i.rsplit("_", 1)[1])]["db_id"] for i in ids}
    s1 = {json.loads(l)["id"]: json.loads(l) for l in open(a.s1)}
    routed = {i for i in ids if s1[i]["margin"] is None or s1[i]["margin"] < T}
    cands = {i: {c["label"]: c["sql"] for c in s1[i].get("candidates", [])} for i in ids}
    runs = []
    for box in a.boxes:
        P = {}
        for arm in ARMS:
            ans = {}
            for f in sorted(glob.glob(str(box / f"answers/{arm}/*.jsonl"))):
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
                for i in ids:
                    if i not in routed:
                        ans[i] = s1[i]["sql"]
            P[arm] = ans
        U = score.usage(a.tasks, box)
        cost = {arm: defaultdict(float) for arm in ARMS}
        for arm in ARMS:
            for chunk, t in U[arm].items():
                cost[arm][chunk.rsplit("_", 1)[0]] += score.dollars(t)
        for d in set(dbid.values()):
            for arm in ("B1", "C"):
                cost[arm][d] += S1_COST * sum(dbid[i] == d for i in ids)
        runs.append((P, cost, {arm: len(U[arm]) for arm in ARMS}))
    tasks = {(dbid[i], s) for i in ids for P, _, _ in runs for s in [gold[i]] + [P[x].get(i) for x in ARMS] if s}
    res = dict(map(score.run, sorted(tasks)))
    ok = lambda i, s: bool(s) and res.get((dbid[i], gold[i])) is not None and res.get((dbid[i], s)) == res[(dbid[i], gold[i])]
    R = [{arm: {i: ok(i, P[arm].get(i)) for i in ids} for arm in ARMS} for P, _, _ in runs]
    dbs = sorted(set(dbid.values()))
    n = len(ids)
    out = {"chunks": [c for _, _, c in runs], "per_run": []}
    for k, (Rk, (P, cost, _)) in enumerate(zip(R, runs)):
        sv = lambda b: round(1 - sum(cost["C"].values()) / sum(cost[b].values()), 4)
        out["per_run"].append({"run": k + 1, "ex": {arm: round(100 * sum(Rk[arm].values()) / n, 2) for arm in ARMS},
                               "cost_per_q": {arm: round(sum(cost[arm].values()) / n, 5) for arm in ARMS},
                               "saving_vs_B1": sv("B1"), "saving_vs_B0": sv("B0"),
                               "C_minus_B1": round(100 * (sum(Rk["C"].values()) - sum(Rk["B1"].values())) / n, 2),
                               "C_minus_B0": round(100 * (sum(Rk["C"].values()) - sum(Rk["B0"].values())) / n, 2),
                               "B1_minus_B2": round(100 * (sum(Rk["B1"].values()) - sum(Rk["B2"].values())) / n, 2)})
    summ = {}
    for key in ("ex", "cost_per_q"):
        summ[key] = {arm: {"mean": round(sum(r[key][arm] for r in out["per_run"]) / 3, 5 if key == "cost_per_q" else 2),
                           "range": [min(r[key][arm] for r in out["per_run"]), max(r[key][arm] for r in out["per_run"])]} for arm in ARMS}
    for key in ("saving_vs_B1", "saving_vs_B0", "C_minus_B1", "C_minus_B0", "B1_minus_B2"):
        v = [r[key] for r in out["per_run"]]
        summ[key] = {"mean": round(sum(v) / 3, 4), "range": [min(v), max(v)]}
    rng = random.Random(20261021)
    for b in ("B1", "B0"):
        f = lambda smp, b=b: 1 - sum(c["C"][d] for _, c, _ in runs for d in smp) / sum(c[b][d] for _, c, _ in runs for d in smp)
        boot = sorted(f([rng.choice(dbs) for _ in dbs]) for _ in range(10000))
        summ[f"pooled_saving_vs_{b}"] = {"point": round(f(dbs), 4), "ci95": [round(boot[249], 4), round(boot[9749], 4)]}
        # accuracy pooled over runs: per-question mean of the 3 runs, DB sign-flip
        mean_b = {i: sum(Rk[b][i] for Rk in R) / 3 for i in ids}
        mean_c = {i: sum(Rk["C"][i] for Rk in R) / 3 for i in ids}
        summ[f"pooled_C_minus_{b}"] = signflip_frac(mean_b, mean_c, dbid)
    summ["pooled_B1_minus_B2"] = signflip_frac({i: sum(Rk["B2"][i] for Rk in R) / 3 for i in ids},
                                                  {i: sum(Rk["B1"][i] for Rk in R) / 3 for i in ids}, dbid)
    summ["agreement"] = {arm: {"all3": sum(all(Rk[arm][i] for Rk in R) for i in ids), "none": sum(not any(Rk[arm][i] for Rk in R) for i in ids),
                               "some": sum(0 < sum(Rk[arm][i] for Rk in R) < 3 for i in ids)} for arm in ARMS}
    # incident sensitivity (INCIDENTS.md): drop r2's B0 cs_semester_2 questions from B0 comparisons in run 2
    drop = {json.loads(l)["question_id"] for l in open(a.boxes[1] / "chunks/B0/cs_semester_2.jsonl")}
    keep = [i for i in ids if i not in drop]
    R2 = R[1]
    summ["r2_without_B0_cs_semester_2"] = {"n": len(keep), "B0_ex": round(100 * sum(R2["B0"][i] for i in keep) / len(keep), 2),
                                           "C_minus_B0": round(100 * sum(R2["C"][i] - R2["B0"][i] for i in keep) / len(keep), 2),
                                           "B0_ex_on_dropped": round(100 * sum(R2["B0"][i] for i in drop) / len(drop), 2)}
    out["summary"] = summ
    a.out.write_text(json.dumps(out, indent=1, default=float))
    for r in out["per_run"]:
        print(json.dumps(r))
    print(json.dumps({k: v for k, v in summ.items() if not k.startswith("pooled_C") and k != "pooled_B1_minus_B2"}, default=float))
    for k in ("pooled_C_minus_B1", "pooled_C_minus_B0", "pooled_B1_minus_B2"):
        print(k, round(summ[k]["delta"], 2), [round(x, 2) for x in summ[k]["ci"]], "p", summ[k]["p"])


if __name__ == "__main__":
    main()
