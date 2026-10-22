"""Score the renamed-schema test of the routed half (experiments/renamed_routed/PREREG.md).

Renamed arm: each agent answer (pick -> that candidate's rewritten SQL, else its own SQL) executed on the renamed copy
against the rewritten gold. Original arm: the cascade's routed answers from the RESERVED runs 1-3 on the original
databases (same scoring as experiments/cost_confirmation/score.py). Paired, DB-clustered.

  .venv/bin/python experiments/renamed_routed/score_ren.py --box <s2ren> --orig <s2res> <s2res_r2> <s2res_r3> \
      --tasks <dir> --out experiments/renamed_routed/result.json
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cost_confirmation"))
sys.path.insert(0, str(ROOT / "experiments/renamed_routed"))
sys.path.insert(0, str(ROOT / "src"))
import build  # noqa: E402
import score  # noqa: E402
from luthor.stats import cluster_signflip  # noqa: E402
from replicates import signflip_frac  # noqa: E402


def answers(path_glob):
    out = {}
    for f in sorted(glob.glob(path_glob)):
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
    ap.add_argument("--orig", type=Path, nargs=3, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(ROOT / "experiments/renamed_routed/build/items.jsonl")]
    items = [x for x in items if "excluded" not in x]
    tr = json.load(open(score.TRAIN))
    s1 = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "runs/phase2/cost_confirmation/s1_reserved.jsonl")}
    dbid = {x["id"]: x["orig_db_id"] for x in items}

    ren_ans = answers(str(a.box / "answers/C/*.jsonl"))
    ren, own, missing = {}, 0, 0
    for x in items:
        qid = f"q{x['id'].rsplit('_', 1)[1]}"
        r = ren_ans.get(qid)
        dbp = a.box / "dbs" / x["db_id"] / f"{x['db_id']}.sqlite"
        if r is None:
            missing += 1
            ren[x["id"]] = False
            continue
        p = str(r.get("pick", "NONE")).strip()
        cands = {c["label"]: c["sql"] for c in x["candidates"]}
        if p.upper() == "NONE" or p not in cands:
            own += 1
            sql = r.get("sql") or ""
        else:
            sql = cands[p]
        g = build.execute(dbp, x["gold"])
        res = build.execute(dbp, sql) if sql else None
        ren[x["id"]] = g is not None and res is not None and res == g

    def orig_run(box):
        ans = answers(str(box / "answers/C/*.jsonl"))
        out = {}
        for x in items:
            i = x["id"]
            r = ans.get(i)
            if r is None:
                out[i] = False
                continue
            cands = {c["label"]: c["sql"] for c in s1[i].get("candidates", [])}
            p = str(r.get("pick", "NONE")).strip()
            sql = cands.get(p) if p.upper() != "NONE" else (r.get("sql") or None)
            d = dbid[i]
            g = score.run((d, tr[int(i.rsplit("_", 1)[1])]["SQL"]))[1]
            res = score.run((d, sql))[1] if sql else None
            out[i] = g is not None and res is not None and res == g
        return out

    runs = [orig_run(b) for b in a.orig]
    n = len(items)
    ex = lambda d: round(100 * sum(d.values()) / n, 2)
    mean_orig = {i: sum(r[i] for r in runs) / 3 for i in ren}
    dbs = sorted(set(dbid.values()))
    rng = random.Random(20261021)
    boot = []
    for _ in range(10000):
        smp = [rng.choice(dbs) for _ in dbs]
        sel = [i for d in smp for i in ren if dbid[i] == d]
        boot.append(100 * sum(ren[i] for i in sel) / len(sel))
    boot.sort()
    lo, hi = boot[249], boot[9749]
    kept_share, kept_acc = 0.49, 90.6
    proj = lambda r: round(kept_share * kept_acc + (1 - kept_share) * r, 1)
    out = {
        "n": n, "missing_answers": missing, "own_sql": own, "picked": n - own - missing,
        "ex_renamed": ex(ren), "ex_renamed_ci95_dbboot": [round(lo, 2), round(hi, 2)],
        "ex_original_run1": ex(runs[0]), "ex_original_runs": [ex(r) for r in runs],
        "ex_original_mean": round(100 * sum(mean_orig.values()) / n, 2),
        "renamed_minus_run1": cluster_signflip(runs[0], ren, dbid),
        "renamed_minus_mean": signflip_frac(mean_orig, {i: float(v) for i, v in ren.items()}, dbid),
        "per_db": {d: {"n": sum(1 for i in ren if dbid[i] == d),
                       "renamed": sum(ren[i] for i in ren if dbid[i] == d),
                       "run1": sum(runs[0][i] for i in ren if dbid[i] == d)} for d in dbs},
        "projection_test_ex": {"point": proj(ex(ren)), "range": [proj(lo), proj(hi)],
                               "assumes": f"kept share {kept_share}, kept accuracy {kept_acc} (RESERVED)"},
    }
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
