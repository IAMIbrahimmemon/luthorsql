"""Where does the model go wrong? Reads the per-question predictions written by eval_sealed.py.

  python scripts/error_breakdown.py runs/sq4b_v1/predictions_bird_dev.jsonl [--examples 3]

For every covered question answered wrongly, the FIRST plan slot that differs from gold is
blamed (in the order below). A wrong answer with an identical plan is a join/compile issue.
For unsupported questions answered anyway, the grammar reason is tallied.
"""
import argparse
import collections
import json
from pathlib import Path


def cid(c):
    return f"{c['table']}.{c['column']}" if c else None


def proj_key(p):
    return (p["agg"] if p["col"]["column"] != "*" else "count", cid(p["col"]))


def filt_key(f):
    return (cid(f["col"]), f["op"], f.get("part", "none"), tuple(map(str, f.get("values") or [])))


def first_diff(g, p):
    """Name of the first slot where predicted plan p differs from gold plan g."""
    gp, pp = [proj_key(x) for x in g["projections"]], [proj_key(x) for x in p["projections"]]
    if len(gp) != len(pp):
        return "projection: how many"
    if [c for _, c in gp] != [c for _, c in pp]:
        # count(*) vs count(id) of the same table is the same answer; only blame real column changes
        return "projection: column"
    if [a for a, _ in gp] != [a for a, _ in pp]:
        return "projection: aggregation"
    if g.get("compose", "none") != p.get("compose", "none"):
        return "projection: compose (ratio/percent)"
    gc = [filt_key(x["cond"]) if x.get("cond") else None for x in g["projections"]]
    pc = [filt_key(x["cond"]) if x.get("cond") else None for x in p["projections"]]
    if gc != pc:
        return "conditional aggregate"
    gf, pf = sorted(filt_key(f) for f in g.get("filters") or []), sorted(filt_key(f) for f in p.get("filters") or [])
    if len(gf) != len(pf):
        return "filter: how many"
    if sorted(x[0] for x in gf) != sorted(x[0] for x in pf):
        return "filter: column"
    if sorted(x[:3] for x in gf) != sorted(x[:3] for x in pf):
        return "filter: operator / date part"
    if gf != pf:
        return "filter: value"
    if sorted(map(cid, g.get("group_by") or [])) != sorted(map(cid, p.get("group_by") or [])):
        return "group by"
    if (g.get("order_dir"), g["order"] and proj_key(g["order"])) != (p.get("order_dir"), p["order"] and proj_key(p["order"])):
        return "order by"
    if g.get("limit") != p.get("limit"):
        return "limit"
    if bool(g.get("distinct")) != bool(p.get("distinct")):
        return "distinct"
    if (g.get("grain"), cid(g.get("time_col"))) != (p.get("grain"), cid(p.get("time_col"))):
        return "time grain"
    return "same plan, different rows (join path / compile)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path, nargs="+")
    ap.add_argument("--examples", type=int, default=2)
    a = ap.parse_args()
    for path in a.predictions:
        recs = [json.loads(l) for l in open(path)]
        plan = [r for r in recs if r["outcome"] == "plan"]
        uns = [r for r in recs if r["outcome"] == "unsupported"]
        wrong = [r for r in plan if not r["correct"]]
        blame = collections.Counter()
        ex = collections.defaultdict(list)
        for r in wrong:
            k = "abstained (should have answered)" if r["pred_plan"] is None else first_diff(r["gold_plan"], r["pred_plan"])
            blame[k] += 1
            if len(ex[k]) < a.examples:
                ex[k].append(r["question"][:110])
        print(f"\n== {path.name}")
        print(f"covered questions: {len(plan)}, right {len(plan) - len(wrong)} ({(len(plan) - len(wrong)) / max(1, len(plan)):.1%}), wrong {len(wrong)}")
        print("wrong answers by first differing slot:")
        for k, v in blame.most_common():
            print(f"  {v:5d}  {v / max(1, len(wrong)):5.1%}  {k}")
            for q in ex[k]:
                print(f"           e.g. {q}")
        answered = [r for r in uns if not r["correct"]]
        reasons = collections.Counter((r.get("reason") or "?").split(":")[0] for r in answered)
        print(f"unsupported questions: {len(uns)}, correctly abstained {len(uns) - len(answered)} "
              f"({(len(uns) - len(answered)) / max(1, len(uns)):.1%}); answered anyway by reason: {dict(reasons.most_common(8))}")
        ms = sorted(r["ms"] for r in recs if r["ms"])
        if ms:
            print(f"latency ms: p50 {ms[len(ms) // 2]:.0f}, p90 {ms[int(0.9 * len(ms))]:.0f}, min {ms[0]:.0f}")


if __name__ == "__main__":
    main()
