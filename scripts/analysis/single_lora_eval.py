"""Single-LoRA System 1 vs the frozen System 1 (R1 k=32 + VALHARD) on DECISION (796 q, 8 DBs, original gold):
runs/phase1/single_lora/PREREG.md. The two systems have different candidates (the single LoRA writes its own k=32),
so this compares whole systems: the pick (valhard_eval.py's rule: max rr over live candidates) and the oracle (a right
candidate anywhere in the pool). Uncovered DECISION questions count wrong. RESERVED stays blind (dropped at load).

  .venv/bin/python scripts/analysis/single_lora_eval.py                 # runs/phase1/single_lora/single/heldout_by_single.jsonl
  .venv/bin/python scripts/analysis/single_lora_eval.py --new F --base F --json out.json

Secondary, when --same is given (the single LoRA's validator scoring R1's own k=32 pool): the selection alone,
through valhard_eval.py on identical candidates (run it directly for the full report).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
sys.path.insert(0, str(ROOT / "scripts/valhard"))
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402
from valhard_eval import DECISION, load, pick  # noqa: E402

FROZEN = ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl"   # R1 k=32 + VALHARD, DECISION 59.05
NEW = ROOT / "runs/phase1/single_lora/single/heldout_by_single.jsonl"
SUBMIT_AT, KILL_AT, ORACLE_MAX_DROP = -1.0, -2.0, 2.0


def verdict(delta: float, oracle_drop: float) -> str:
    """PREREG: SUBMIT if delta >= -1.0 and the oracle fell by at most 2.0; KILL if delta < -2.0; else HOLD."""
    if delta < KILL_AT:
        return "KILL"
    return "SUBMIT" if delta >= SUBMIT_AT and oracle_drop <= ORACLE_MAX_DROP else "HOLD"


def hits(S: dict, ids: list) -> tuple[dict, dict]:
    h, o = {}, {}
    for i in ids:
        p = pick(S[i]["cands"]) if i in S else None
        h[i] = bool(p and p.get("ok"))
        o[i] = i in S and any(c.get("ok") for c in S[i]["cands"])
    return h, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, default=NEW)
    ap.add_argument("--base", type=Path, default=FROZEN)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--train-json", type=Path, default=ROOT / "data/raw/bird_train/train.json",
                    help="BIRD train.json (question id -> db_id)")
    a = ap.parse_args()
    dec = json.load(open(DECISION))
    ids = list(dec["ids"])
    B, N = load(a.base, set(ids)), load(a.new, set(ids))
    tj = a.train_json
    if tj.exists():
        tr = json.load(open(tj))
        db = {i: tr[int(i.rsplit("_", 1)[1])]["db_id"] for i in ids}
    else:  # train.json not on this machine: the scored files carry db_id
        db = {i: (B.get(i) or N.get(i) or {}).get("db_id") for i in ids}
        if None in db.values():
            raise SystemExit(f"{sum(v is None for v in db.values())} DECISION ids in neither file and no {tj}")
    assert collections.Counter(db.values()) == collections.Counter(dec["database_counts"])
    n = len(ids)
    hb, ob = hits(B, ids)
    hn, on = hits(N, ids)
    pct = lambda h: 100 * sum(h.values()) / n  # noqa: E731
    c = cluster_signflip(hb, hn, db)
    q = question_signflip(hb, hn)
    drop = pct(ob) - pct(on)
    v = verdict(c["delta"], drop)
    cand = lambda S: sum(len({x["sql"] for x in S[i]["cands"] if x.get("sql")}) for i in S) / max(len(S), 1)  # noqa: E731
    print(f"DECISION n={n}, {len(set(db.values()))} DBs; covered frozen {len(B)} single {len(N)}; "
          f"distinct candidates per question frozen {cand(B):.1f} single {cand(N):.1f}")
    print("PRIMARY (whole System 1: own k=32 candidates, own validator picks)")
    print(f"  frozen (R1 + VALHARD) {pct(hb):6.2f}   single LoRA {pct(hn):6.2f}   delta {c['delta']:+.2f}  "
          f"DB sign-flip p={c['p']:.3f}  CI [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]  McNemar p={q['p']:.3f} "
          f"(+{q['wins']}/-{q['losses']})")
    print(f"  oracle (right answer anywhere): frozen {pct(ob):6.2f}   single {pct(on):6.2f}   drop {drop:+.2f}")
    print(f"  -> {v}   (SUBMIT: delta >= {SUBMIT_AT:+.1f} and oracle drop <= {ORACLE_MAX_DROP}; KILL: delta < "
          f"{KILL_AT:+.1f}; otherwise HOLD)")
    print("  per DB (single - frozen, questions): " + ", ".join(
        f"{d} {sum(hn[i] - hb[i] for i in ids if db[i] == d):+d}" for d in sorted(set(db.values()))))
    out = {"new": str(a.new), "base": str(a.base), "n": n, "frozen": pct(hb), "single": pct(hn),
           "delta": c["delta"], "p": c["p"], "ci": c["ci"],
           "mcnemar": {"p": q["p"], "wins": q["wins"], "losses": q["losses"]},
           "oracle_frozen": pct(ob), "oracle_single": pct(on), "oracle_drop": drop, "verdict": v}
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
