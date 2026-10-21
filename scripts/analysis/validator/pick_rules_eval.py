"""Single pre-registered RESERVED evaluation of the frozen pick rules (runs/phase1/pickrules/PREREG.md).

  .venv/bin/python scripts/analysis/validator/pick_rules_eval.py --decision-check   # DECISION only: reproduce note 10
  .venv/bin/python scripts/analysis/validator/pick_rules_eval.py --reserved         # THE one RESERVED look (locks)

--reserved refuses to run unless the PREREG exists and contains the FROZEN marker, and refuses if
runs/phase1/pickrules/LOOK_USED exists. It writes LOOK_USED before reading any RESERVED row, then writes
runs/phase1/pickrules/result.json. Rules come from pick_rules.py (no fitted parameters; GAMMA = 0.2 frozen).

Universe: every id of the split (RESERVED 348 q, 8 DBs; DECISION 796 q); uncovered ids count wrong in every arm.
Pools (VALHARD rr): k32 = runs/phase1/valhard/valhard/heldout_by_new.jsonl (R1 share: src in {r1, both});
k64+repair = runs/phase1/repair/repair/k64_repair_pool_by_valhard.jsonl.
Stats: delta (points), one-sided exact McNemar p (P[Bin(m, 1/2) >= wins]), exact DB sign-flip p (two-sided, 8 DBs)
and its 95% CI, two-sided McNemar (info).
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pick_rules as PR  # noqa: E402
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402

SPLITS = {"decision": ROOT / "data/sys1/heldout_v2/decision.json",
          "reserved": ROOT / "data/sys1/heldout_v2/reserved.json"}
POOLS = {"k32": (ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl", True),
         "k64r": (ROOT / "runs/phase1/repair/repair/k64_repair_pool_by_valhard.jsonl", False)}
OUT = ROOT / "runs/phase1/pickrules"
PREREG = OUT / "PREREG.md"
LOCK = OUT / "LOOK_USED"
FROZEN_MARKER = "PREREG-FROZEN-2026-10-11"
FORBIDDEN = ("bird_dev", "gate.json", "sealed", "/score_", "poolA_by_valhard", "look_")
PASS_DELTA, PASS_MCN, PASS_DBP, KILL_DELTA = 1.0, 0.05, 0.25, 0.3

# (name, pool, rule, baseline pool, baseline rule, role)
CELLS = [
    ("PRIMARY  R1+R2 k64+repair vs VALHARD k32", "k64r", "r1r2", "k32", "max", "primary"),
    ("R2 alone k32 vs VALHARD k32", "k32", "r2", "k32", "max", "secondary"),
    ("R1+R2 k64+repair vs VALHARD k64+repair", "k64r", "r1r2", "k64r", "max", "secondary"),
    ("info: VALHARD k64+repair vs VALHARD k32 (width+repair)", "k64r", "max", "k32", "max", "info"),
    ("info: R1 alone k64+repair vs VALHARD k32", "k64r", "r1", "k32", "max", "info"),
    ("info: R2 alone k64+repair vs VALHARD k32", "k64r", "r2", "k32", "max", "info"),
]


def load(path: Path, ids: set, r1_share: bool) -> dict:
    assert not any(f in str(path) for f in FORBIDDEN), path
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["id"] not in ids:
            continue
        cs = r["cands"]
        if r1_share and any("src" in c for c in cs):
            cs = [c for c in cs if c.get("src") in ("r1", "both")]
        out[r["id"]] = cs
    return out


def mcnemar_one_sided(w: int, l_: int) -> float:
    m = w + l_
    if m == 0:
        return 1.0
    return sum(math.comb(m, j) for j in range(w, m + 1)) / 2 ** m


def evaluate(split: str) -> dict:
    meta = json.load(open(SPLITS[split]))
    ids = list(meta["ids"])
    idset = set(ids)
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    db = {i: tr[int(i.rsplit("_", 1)[1])]["db_id"] for i in ids}
    P = {k: load(f, idset, sh) for k, (f, sh) in POOLS.items()}
    n = len(ids)

    def hits(pool, rule):
        return {i: bool(i in P[pool] and (p := PR.RULES[rule](P[pool][i])) and p.get("ok")) for i in ids}

    H = {(pool, rule): hits(pool, rule) for pool in POOLS for rule in PR.RULES}
    res = {"split": split, "n": n, "dbs": sorted(set(db.values())),
           "covered": {k: len(v) for k, v in P.items()},
           "acc": {f"{p}/{r}": round(100 * sum(h.values()) / n, 2) for (p, r), h in H.items()},
           "oracle": {p: round(100 * sum(1 for i in ids if i in P[p] and any(c.get("ok") for c in P[p][i])) / n, 2)
                      for p in POOLS},
           "cells": []}
    # how often R2 fires / changes the pick (diagnostic)
    for p in POOLS:
        fired = sum(1 for i in ids if i in P[p] and PR.pick_max(P[p][i]) is not PR.pick_r2(P[p][i]))
        res[f"r2_changes_pick_{p}"] = fired
    for name, pool, rule, bpool, brule, role in CELLS:
        a, b = H[(bpool, brule)], H[(pool, rule)]
        c = cluster_signflip(a, b, db)
        q = question_signflip(a, b)
        w, l_ = q["wins"], q["losses"]
        cell = {"name": name, "role": role, "base": res["acc"][f"{bpool}/{brule}"], "new": res["acc"][f"{pool}/{rule}"],
                "delta": round(c["delta"], 2), "wins": w, "losses": l_,
                "mcnemar_1sided_p": round(mcnemar_one_sided(w, l_), 4), "mcnemar_2sided_p": round(q["p"], 4),
                "db_signflip_p": round(c["p"], 4), "db_ci95": [round(float(x), 2) for x in c["ci"]],
                "per_db": {k: int(v) for k, v in c["per_db"].items()}}
        if role == "primary":
            d, pm, pd = cell["delta"], cell["mcnemar_1sided_p"], cell["db_signflip_p"]
            cell["verdict"] = ("PASS" if d >= PASS_DELTA and pm < PASS_MCN and pd < PASS_DBP
                               else "KILL" if d < KILL_DELTA else "NO PASS")
            res["verdict"] = cell["verdict"]
        res["cells"].append(cell)
    return res


def show(res: dict) -> None:
    print(f"== {res['split'].upper()}  n={res['n']}  covered {res['covered']}  DBs {len(res['dbs'])}")
    print("  accuracy:", res["acc"])
    print("  oracle:", res["oracle"], " R2 changes pick:",
          {k: v for k, v in res.items() if k.startswith("r2_changes")})
    for c in res["cells"]:
        print(f"  {c['name']:56s} {c['base']:6.2f} -> {c['new']:6.2f}  d {c['delta']:+5.2f}  +{c['wins']}/-{c['losses']}"
              f"  McN1 p {c['mcnemar_1sided_p']:.4f}  DB p {c['db_signflip_p']:.4f}  CI {c['db_ci95']}"
              + (f"  => {c['verdict']}" if "verdict" in c else ""))
        print(f"      per-DB {c['per_db']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--decision-check", action="store_true")
    g.add_argument("--reserved", action="store_true")
    a = ap.parse_args(argv)
    if a.decision_check:
        res = evaluate("decision")
        show(res)
        json.dump(res, open(OUT / "decision_check.json", "w"), indent=1)
        return
    if not PREREG.exists() or FROZEN_MARKER not in PREREG.read_text():
        raise SystemExit(f"refusing: {PREREG} missing or not frozen ({FROZEN_MARKER})")
    if LOCK.exists():
        raise SystemExit(f"refusing: RESERVED look already used ({LOCK}: {LOCK.read_text().strip()})")
    LOCK.write_text(f"RESERVED look used {datetime.datetime.now().isoformat(timespec='seconds')} by pick_rules_eval.py\n")
    res = evaluate("reserved")
    show(res)
    json.dump(res, open(OUT / "result.json", "w"), indent=1)
    print("wrote", OUT / "result.json", "and lock", LOCK)


if __name__ == "__main__":
    main()
