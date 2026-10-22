"""Score the top-3-at-30% variants (experiments/top3_at_30/PREREG.md) against V0 on the renamed routed questions.

Per variant box: routed EX on the given half, paired with V0 (s2ren); agent dollars per question from the
transcripts (same pricing as the cost tests), ratio to V0 on the same chunks; projected saving vs B1 and projected
BIRD-test EX by the pre-registered formulas.

  .venv/bin/python experiments/top3_at_30/score_variants.py --half dev --boxes s2v1 s2v2 s2v3 --scratch <dir> \
      --tasks <dir> --out experiments/top3_at_30/dev.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cost_confirmation"))
sys.path.insert(0, str(ROOT / "experiments/external_cascade"))
sys.path.insert(0, str(ROOT / "experiments/renamed_routed"))
sys.path.insert(0, str(ROOT / "src"))
import build  # noqa: E402
import score  # noqa: E402
import score_ext  # noqa: E402
from luthor.stats import cluster_signflip  # noqa: E402

HALVES = {"dev": ("car_retails", "law_episode", "synthea", "university"),
          "confirm": ("cs_semester", "hockey", "movie", "student_loan")}
# pooled RESERVED cost facts (PREREG): $/question
S1_Q, B1_Q, C_ROUTED_Q, SHARE = 0.000849, 0.0222, 0.0278, 0.5086
KEPT_SHARE, KEPT_EX, V0_ROUTED_ALL = 0.49, 90.6, 72.88


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


def box_eval(box: Path, items, tasks: Path):
    ans = answers(str(box / "answers/C/*.jsonl"))
    chunks = {}
    for f in (box / "chunks/C").glob("*.jsonl"):
        for ln in open(f):
            r = json.loads(ln)
            chunks[r["question_id"]] = (f.stem, {c["label"]: c["sql"] for c in r["candidates"]})
    ok, own = {}, 0
    for x in items:
        qid = f"q{x['id'].rsplit('_', 1)[1]}"
        if qid not in chunks:
            continue
        r = ans.get(qid)
        dbp = box / "dbs" / x["db_id"] / f"{x['db_id']}.sqlite"
        if r is None:
            ok[x["id"]] = False
            continue
        p = str(r.get("pick", "NONE")).strip()
        cands = chunks[qid][1]
        if p.upper() == "NONE" or p not in cands:
            own += 1
            sql = r.get("sql") or ""
        else:
            sql = cands[p]
        g = build.execute(dbp, x["gold"])
        res = build.execute(dbp, sql) if sql else None
        ok[x["id"]] = g is not None and res is not None and res == g
    halfdbs = {x["db_id"] for x in items}  # cost on the given half only (fixed 2026-10-22: V0 was costed over all chunks)
    chunks = {k: v for k, v in chunks.items() if v[0].rsplit("_", 1)[0] in halfdbs}
    stems = {v[0] for v in chunks.values()}
    pat = re.compile(re.escape(str(box)) + r"/chunks/C/([A-Za-z0-9_]+_\d+)\.jsonl")
    dollars, nq = 0.0, 0
    for f in glob.glob(str(tasks / "*.output")):
        m = pat.search(open(f).read(20000))
        if m and m.group(1) in stems:
            dollars += score.dollars(score_ext._one(f))
    nq = sum(1 for v in chunks.values())
    return ok, own, dollars / max(1, nq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", choices=list(HALVES), required=True)
    ap.add_argument("--boxes", nargs="+", required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(ROOT / "experiments/renamed_routed/build/items.jsonl")]
    items = [x for x in items if "excluded" not in x and x["orig_db_id"] in HALVES[a.half]]
    dbid = {x["id"]: x["orig_db_id"] for x in items}
    v0_ok, v0_own, v0_q = box_eval(a.scratch / "s2ren", items, a.tasks)
    v0_ok = {i: v for i, v in v0_ok.items() if i in dbid}
    out = {"half": a.half, "n": len(items), "V0": {"ex": round(100 * sum(v0_ok.values()) / len(v0_ok), 2),
                                                     "own_sql": v0_own, "dollars_per_q": round(v0_q, 5)}}
    for b in a.boxes:
        ok, own, q = box_eval(a.scratch / b, items, a.tasks)
        common = sorted(set(ok) & set(v0_ok))
        ratio = q / v0_q if v0_q else float("nan")
        saving = 1 - (S1_Q + SHARE * C_ROUTED_Q * ratio) / (S1_Q + B1_Q)
        ex = 100 * sum(ok[i] for i in common) / len(common)
        delta = ex - 100 * sum(v0_ok[i] for i in common) / len(common)
        routed_proj = V0_ROUTED_ALL + delta
        out[b] = {"n": len(common), "ex": round(ex, 2), "minus_V0": round(delta, 2), "own_sql": own,
                  "paired": cluster_signflip({i: v0_ok[i] for i in common}, {i: ok[i] for i in common},
                                             {i: dbid[i] for i in common}),
                  "dollars_per_q": round(q, 5), "cost_ratio_vs_V0": round(ratio, 3),
                  "projected_saving_vs_B1": round(saving, 4),
                  "projected_test_ex": round(KEPT_SHARE * KEPT_EX + (1 - KEPT_SHARE) * routed_proj, 2)}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
