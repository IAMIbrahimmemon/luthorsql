"""Fast path + fallback (Mac, no GPU): the slot model answers when its validator is confident, otherwise an
autoregressive model's vote answers (e.g. Arctic-Text2SQL-R1-7B from gen6.py --exec-only).

  python scripts/hybrid_arctic.py --slot runs/sys1/r6c/sys1_full/rr/bird_dev_scored.jsonl \
      --ar runs/arc/arc_bird_dev.jsonl [--fit-frac 0.5]

Prints, for each validator cut-off t: share of questions on the fast path, fast-path accuracy, and overall
accuracy (slot pick if max validator score >= t, else the AR vote). Questions without slot candidates always go
to the AR model. The cut-off is chosen on a random half of the databases (--fit-frac) and reported on the other
half, so the headline number is not tuned on the questions it is measured on.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sqlw.evalkit import pick_vote  # noqa: E402
from select_groups import pick  # noqa: E402


def load(slot_f, ar_f):
    ar = {r["id"]: r for r in map(json.loads, open(ar_f))}
    slot = {r["id"]: r for r in map(json.loads, open(slot_f))}
    rows = []
    for qid, r in ar.items():
        gold = r["gold_sig"]
        ok = lambda c: bool(c and c.get("runs") and c.get("sig") == gold)
        greedy = next((c for c in r["cands"] if c["src"].endswith("_g")), None)
        v = pick_vote(r["cands"])
        s = slot.get(qid)
        sp = pick(s, 0.0, 0.0, grouped=False) if s else None
        conf = max((c["rr"] for c in s["cands"] if "rr" in c), default=None) if s else None
        rows.append({"id": qid, "db": r["db_id"], "ar_g": ok(greedy), "ar_v": ok(v), "ar_o": any(map(ok, r["cands"])),
                     "slot": bool(sp and sp.get("ok")), "conf": conf})
    return rows


def acc(rows, t):
    fast = [r for r in rows if r["conf"] is not None and r["conf"] >= t]
    right = sum(r["slot"] for r in fast) + sum(r["ar_v"] for r in rows if not (r["conf"] is not None and r["conf"] >= t))
    return right / len(rows), len(fast) / len(rows), (sum(r["slot"] for r in fast) / len(fast) if fast else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=Path, required=True)
    ap.add_argument("--ar", type=Path, required=True)
    ap.add_argument("--fit-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = load(a.slot, a.ar)
    n = len(rows)
    print(f"n={n}  AR greedy {sum(r['ar_g'] for r in rows) / n:.1%}  AR vote {sum(r['ar_v'] for r in rows) / n:.1%}  "
          f"AR oracle {sum(r['ar_o'] for r in rows) / n:.1%}  slot {sum(r['slot'] for r in rows) / n:.1%}  "
          f"either (slot or AR vote) {sum(r['slot'] or r['ar_v'] for r in rows) / n:.1%}")
    confs = sorted({r["conf"] for r in rows if r["conf"] is not None})
    ts = [confs[int(q * (len(confs) - 1))] for q in [i / 20 for i in range(21)]] + [float("inf")]
    print(f"{'cut-off':>8} {'fast share':>10} {'fast acc':>9} {'overall':>8}")
    for t in ts:
        o, share, fa = acc(rows, t)
        print(f"{t:8.2f} {share:10.1%} {fa:9.1%} {o:8.1%}")
    dbs = sorted({r["db"] for r in rows})
    random.Random(a.seed).shuffle(dbs)
    fit_dbs = set(dbs[:max(1, int(len(dbs) * a.fit_frac))])
    fit = [r for r in rows if r["db"] in fit_dbs]
    test = [r for r in rows if r["db"] not in fit_dbs]
    t = max(ts, key=lambda t: acc(fit, t)[0])
    o, share, fa = acc(test, t)
    base = sum(r["ar_v"] for r in test) / len(test)
    print(f"cut-off {t:.2f} fitted on {len(fit_dbs)} databases ({len(fit)} q); on the other {len(dbs) - len(fit_dbs)} "
          f"({len(test)} q): hybrid {o:.1%} (AR vote alone {base:.1%}), fast path {share:.1%} at {fa:.1%}")


if __name__ == "__main__":
    main()
