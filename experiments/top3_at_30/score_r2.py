"""Score round 2 (experiments/top3_at_30/PREREG.md, "Round 2"): agent 1 = V3 box, agent 2 = s2r2 box (disagreement
questions only), final = majority of {agent 1, agent 2, System 1's top candidate A} by execution result, else agent 1.
Cost = agent 1 + agent 2 dollars per routed question. Same formulas as score_variants.py.

  .venv/bin/python experiments/top3_at_30/score_r2.py --half dev --a1 s2v3 --a2 s2r2 --scratch <dir> --tasks <dir> --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/top3_at_30"))
from score_variants import (HALVES, S1_Q, B1_Q, C_ROUTED_Q, SHARE, KEPT_SHARE, KEPT_EX, V0_ROUTED_ALL,  # noqa: E402
                            answers, box_eval, build, score, score_ext, cluster_signflip)


def picks(box: Path):
    ans = answers(str(box / "answers/C/*.jsonl"))
    out = {}
    for f in (box / "chunks/C").glob("*.jsonl"):
        for ln in open(f):
            r = json.loads(ln)
            cands = {c["label"]: c["sql"] for c in r["candidates"]}
            a = ans.get(r["question_id"]) or {}
            p = str(a.get("pick", "NONE")).strip()
            out[r["question_id"]] = (cands.get(p) if p.upper() != "NONE" else a.get("sql"), cands.get("A"), r["db_id"])
    return out


def dollars(box: Path, tasks: Path):
    pat = re.compile(re.escape(str(box)) + r"/chunks/C/([A-Za-z0-9_]+_\d+)\.jsonl")
    return sum(score.dollars(score_ext._one(f)) for f in glob.glob(str(tasks / "*.output")) if pat.search(open(f).read(20000)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", choices=list(HALVES), required=True)
    ap.add_argument("--a1", required=True)
    ap.add_argument("--a2", required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(ROOT / "experiments/renamed_routed/build/items.jsonl")]
    items = [x for x in items if "excluded" not in x and x["orig_db_id"] in HALVES[a.half]]
    dbid = {x["id"]: x["orig_db_id"] for x in items}
    b1, b2 = a.scratch / a.a1, a.scratch / a.a2
    p1, p2 = picks(b1), picks(b2)
    v0_ok, _, v0_q = box_eval(a.scratch / "s2ren", items, a.tasks)
    final_ok, used2, majority = {}, 0, 0
    for x in items:
        qid = f"q{x['id'].rsplit('_', 1)[1]}"
        dbp = b1 / "dbs" / x["db_id"] / f"{x['db_id']}.sqlite"
        run = lambda s: build.execute(dbp, s) if s else None
        s1sql, top, _ = p1[qid]
        r1, rA = run(s1sql), run(top)
        choice = r1
        if qid in p2:
            used2 += 1
            r2 = run(p2[qid][0])
            for c in (r1, r2, rA):
                if c is not None and sum(c == o for o in (r1, r2, rA)) >= 2:
                    choice = c
                    majority += 1
                    break
        g = run(x["gold"])
        final_ok[x["id"]] = g is not None and choice is not None and choice == g
    n = len(items)
    q = (dollars(b1, a.tasks) + dollars(b2, a.tasks)) / n
    ratio = q / v0_q
    ex = 100 * sum(final_ok.values()) / n
    delta = ex - 100 * sum(v0_ok[i] for i in final_ok) / n
    saving = 1 - (S1_Q + SHARE * C_ROUTED_Q * ratio) / (S1_Q + B1_Q)
    out = {"half": a.half, "n": n, "agent2_questions": used2, "majority_decided": majority,
           "ex": round(ex, 2), "minus_V0": round(delta, 2),
           "paired": cluster_signflip({i: v0_ok[i] for i in final_ok}, final_ok, dbid),
           "dollars_per_q": round(q, 5), "cost_ratio_vs_V0": round(ratio, 3),
           "projected_saving_vs_B1": round(saving, 4),
           "projected_test_ex": round(KEPT_SHARE * KEPT_EX + (1 - KEPT_SHARE) * (V0_ROUTED_ALL + delta), 2)}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
