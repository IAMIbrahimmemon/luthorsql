"""Offline evaluation of the stricter gate G (experiments/top3_at_30/PREREG.md).

G also routes a kept RESERVED question (margin >= 3) when System 1's top group returns 0 rows, or a first row that is
all NULL, or holds fewer than 8 of the 32 samples. A newly routed question is scored with B1's answers (mean over
RESERVED runs 1-3); everything else is unchanged. Reported per half: questions newly routed, EX change, cost.

  .venv/bin/python experiments/top3_at_30/gate_eval.py --boxes <s2res> <s2res_r2> <s2res_r3> --out experiments/top3_at_30/gate.json
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
sys.path.insert(0, str(ROOT / "experiments/top3_at_30"))
import score  # noqa: E402
from score_variants import HALVES, S1_Q, B1_Q, C_ROUTED_Q  # noqa: E402

T = 3.0


def suspicious(row):
    c = row.get("candidates") or []
    if not c:
        return False
    a = c[0]
    res = str(a.get("result", ""))
    allnull = bool(re.search(r"first: \((None(, None)*)?\)$", res)) or res.endswith("first: (None,)")
    return res.startswith("0 rows") or allnull or int(a.get("n_of_32", 32)) < 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", type=Path, nargs=3, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tr = json.load(open(score.TRAIN))
    s1 = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "runs/phase2/cost_confirmation/s1_reserved.jsonl")}
    ids = sorted(s1)
    db = {i: s1[i]["db_id"] for i in ids}
    gold = {i: score.run((db[i], tr[int(i.rsplit("_", 1)[1])]["SQL"]))[1] for i in ids}
    ok = lambda i, sql: bool(sql) and gold[i] is not None and score.run((db[i], sql))[1] == gold[i]
    kept = [i for i in ids if not (s1[i]["margin"] is None or s1[i]["margin"] < T)]
    newly = [i for i in kept if suspicious(s1[i])]
    b1 = {}
    for box in a.boxes:
        ans = {}
        for f in sorted(glob.glob(str(box / "answers/B1/*.jsonl"))):
            for ln in open(f):
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                ans.setdefault(r.get("question_id"), r)
        for i in newly:
            r = ans.get(i) or {}
            p = str(r.get("pick", "NONE")).strip()
            cands = {c["label"]: c["sql"] for c in s1[i].get("candidates", [])}
            b1.setdefault(i, []).append(ok(i, cands.get(p) if p.upper() != "NONE" else r.get("sql")))
    out = {"kept": len(kept), "newly_routed": len(newly), "halves": {}}
    for h, dbs in HALVES.items():
        sel = [i for i in ids if db[i] in dbs]
        nw = [i for i in newly if db[i] in dbs]
        s1_ok = sum(ok(i, s1[i]["sql"]) for i in nw)
        b1_ok = sum(sum(b1[i]) / 3 for i in nw)
        routed = sum(1 for i in sel if s1[i]["margin"] is None or s1[i]["margin"] < T)
        share0, share1 = routed / len(sel), (routed + len(nw)) / len(sel)
        sav = lambda share, ratio=1.0: round(1 - (S1_Q + share * C_ROUTED_Q * ratio) / (S1_Q + B1_Q), 4)
        out["halves"][h] = {"n": len(sel), "newly_routed": len(nw), "s1_correct_on_newly": s1_ok,
                            "b1_mean_correct_on_newly": round(b1_ok, 2),
                            "ex_change_points": round(100 * (b1_ok - s1_ok) / len(sel), 2),
                            "routed_share": [round(share0, 4), round(share1, 4)],
                            "saving_at_V0_cost": [sav(share0), sav(share1)],
                            "saving_at_V3_cost_explor": [sav(share0, 0.8), sav(share1, 0.8)]}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
