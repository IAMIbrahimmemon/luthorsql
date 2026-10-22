"""Choosing the System 1 -> Opus cascade threshold from data instead of "send half" (docs/paper_notes/
cascade_threshold_2026-10-17.md). DECISION only (796 held-out BIRD-train questions, original gold); CPU; no new model
calls. The threshold is a VALHARD MARGIN (score gap between the top two distinct result classes of System 1's live
candidates), not a fraction of questions, so it transfers to another question set (BIRD dev) unchanged.

Inputs:
  runs/phase1/valhard/valhard/heldout_by_new.jsonl  frozen System 1 (R1 k=32 + VALHARD): rr, sig, runs, nonnull, ok
  runs/analysis/hybrid/opus_s1.json                 per_question on the 400-question sample: s1, alone, plus_s1
  data/sys1/heldout_v2/decision.json                DECISION ids
Prints: the calibration of P(System 1 right | margin), the full cascade curve on the 400 sample, the three
pre-stated rules' thresholds, leave-one-database-out estimates for each rule, and the expected Opus share on BIRD dev
(label-free: margins only) from runs/phase1/devlook/look_A.json if it carries candidate scores.

  .venv/bin/python scripts/analysis/cascade_threshold.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S1 = ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl"
OPUS = ROOT / "runs/analysis/hybrid/opus_s1.json"
DEC = ROOT / "data/sys1/heldout_v2/decision.json"
TRAIN_JSON = ROOT / "data/raw/bird_train/train.json"  # question id -> db_id for uncovered questions
TRAIN_JSON_COPY = Path(os.environ.get("LUTHOR_TRAIN_JSON", ""))  # a copy of train.json when data/raw is unmounted
COST_OPUS = 0.025     # $ per Opus handoff call (recomputed 2026-10-17 from usage logs at standard cache prices)
COST_S1 = 0.00064     # $ per System 1 question (batched H100 PCIe, $2.50/h)
DELTA = 1.0           # rule B tolerance: the documented run-to-run noise of one System 1 recipe (about 1 point)


def margin(cands: list) -> tuple[float, bool]:
    """(top-2 distinct-result margin of the validator over live candidates, pick ok). No live candidate -> -inf.
    One result class -> -inf too: with nothing to compare, the margin carries no information, and on DECISION those
    questions are right only 40% of the time (6 of 15; they usually have one or two live candidates)."""
    live = [c for c in cands if c.get("sql") and c.get("runs") and c.get("nonnull")]
    if not live:
        u = [c for c in cands if c.get("sql")]
        p = max(u, key=lambda c: c.get("rr", -1e9)) if u else None
        return -math.inf, bool(p and p.get("ok"))
    best = {}
    for c in live:
        s = c.get("sig") or c["sql"]
        if s not in best or c["rr"] > best[s]["rr"]:
            best[s] = c
    cls = sorted(best.values(), key=lambda c: -c["rr"])
    m = -math.inf if len(cls) == 1 else cls[0]["rr"] - cls[1]["rr"]
    if len(cls) == 1:
        return m, bool(cls[0].get("ok"))
    return m, bool(cls[0].get("ok"))


def load():
    ids = set(json.load(open(DEC))["ids"])
    s1 = {}
    for line in open(S1):
        r = json.loads(line)
        if r["id"] in ids:
            m, ok = margin(r["cands"])
            s1[r["id"]] = {"m": m, "ok": ok, "db": r["db_id"]}
    tj = TRAIN_JSON if TRAIN_JSON.exists() else TRAIN_JSON_COPY
    tr = json.load(open(tj))
    for i in ids - set(s1):  # uncovered questions: no candidate at all
        s1[i] = {"m": -math.inf, "ok": False, "db": tr[int(i.rsplit("_", 1)[1])]["db_id"]}
    opus = json.load(open(OPUS))["per_question"]
    return s1, opus


def curve(rows: list[dict]) -> list[dict]:
    """Cascade on rows (each with m, s1_ok, plus_ok): for every candidate threshold t (skip Opus when m >= t)."""
    ts = sorted({r["m"] for r in rows if math.isfinite(r["m"])} | {math.inf}, reverse=True)
    out = []
    n = len(rows)
    for t in ts + [-math.inf]:
        skip = [r for r in rows if r["m"] >= t and t != -math.inf] if t != -math.inf else []
        rest = [r for r in rows if r not in skip] if skip else rows
        acc = (sum(r["s1_ok"] for r in skip) + sum(r["plus_ok"] for r in rest)) / n
        out.append({"t": t, "skip": len(skip) / n, "acc": 100 * acc, "opus_calls": len(rest) / n,
                    "cost": (len(rest) * COST_OPUS + n * COST_S1) / n,
                    "s1_acc_skipped": 100 * sum(r["s1_ok"] for r in skip) / max(len(skip), 1),
                    "opus_acc_skipped": 100 * sum(r["plus_ok"] for r in skip) / max(len(skip), 1)})
    return out


def rule_a(cv):  # highest accuracy; ties -> fewest Opus calls
    return max(cv, key=lambda x: (round(x["acc"], 6), -x["opus_calls"]))


def rule_b(cv, full_acc):  # fewest Opus calls with accuracy within DELTA of always sending to Opus
    ok = [x for x in cv if x["acc"] >= full_acc - DELTA]
    return min(ok, key=lambda x: (x["opus_calls"], -x["acc"]))


def rule_c(cv, value):  # max (correct answers x value - cost) per question; value = $ an extra right answer is worth
    return max(cv, key=lambda x: (x["acc"] / 100 * value - x["cost"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    s1, opus = load()
    out = {}
    # 1. calibration of System 1 on all 796, by margin bin
    fin = sorted(v["m"] for v in s1.values() if math.isfinite(v["m"]))
    q = [fin[int(len(fin) * k / 8)] for k in range(1, 8)]
    edges = [-math.inf, -1e-12] + q + [math.inf]  # -inf bucket = no live candidate; +inf = all candidates agree
    print("CALIBRATION (all 796 DECISION questions): P(System 1 right | margin bin)")
    cal = []
    groups = [("no margin (no live cand. / one class)", [v for v in s1.values() if v["m"] == -math.inf])]
    lo = -1e-12
    for hi in q + [math.inf]:
        groups.append((f"margin [{lo:.3f}, {hi:.3f})", [v for v in s1.values() if math.isfinite(v["m"]) and lo <= v["m"] < hi]))
        lo = hi
    for name, g in groups:
        if g:
            acc = 100 * sum(v["ok"] for v in g) / len(g)
            cal.append({"bin": name, "n": len(g), "s1_acc": acc})
            print(f"  {name:32s} n={len(g):4d}  System 1 right {acc:5.1f}%")
    out["calibration"] = cal
    # 2. cascade curve on the 400-question sample (Opus handoff answers exist only there)
    rows = [{"id": i, "m": s1[i]["m"], "db": s1[i]["db"], "s1_ok": bool(v["s1"]), "plus_ok": bool(v["plus_s1"]),
             "alone_ok": bool(v["alone"])} for i, v in opus.items()]
    agree = sum(r["s1_ok"] == s1[r["id"]]["ok"] for r in rows)
    print(f"\n400-question sample: System 1 correctness from the cascade file agrees with VALHARD's pick on {agree}/400")
    print("\nMARGINAL COMPARISON on the 400 sample, by the same margin bins (who is right more often in each band):")
    lo = None
    bands = [("no margin", lambda m: m == -math.inf)]
    prev = -1e-12
    for hi in q + [math.inf]:
        bands.append((f"[{prev:.2f}, {hi:.2f})", (lambda a, b: (lambda m: math.isfinite(m) and a <= m < b))(prev, hi)))
        prev = hi
    marg = []
    for name, f in bands:
        g = [r for r in rows if f(r["m"])]
        if g:
            s1a = 100 * sum(r["s1_ok"] for r in g) / len(g)
            opa = 100 * sum(r["plus_ok"] for r in g) / len(g)
            marg.append({"band": name, "n": len(g), "s1": s1a, "opus_handoff": opa, "opus_gain": opa - s1a})
            print(f"  {name:20s} n={len(g):3d}  System 1 {s1a:5.1f}%  Opus+S1 {opa:5.1f}%  Opus adds {opa - s1a:+5.1f} pt"
                  f"  -> ${COST_OPUS / max((opa - s1a) / 100, 1e-9):.2f} per extra right answer" if opa > s1a else
                  f"  {name:20s} n={len(g):3d}  System 1 {s1a:5.1f}%  Opus+S1 {opa:5.1f}%  Opus adds {opa - s1a:+5.1f} pt")
    out["marginal"] = marg
    cv = curve(rows)
    full = cv[-1]["acc"]
    print(f"  always Opus (handoff): {full:.2f}   System 1 alone: {100 * sum(r['s1_ok'] for r in rows) / 400:.2f}")
    A, B = rule_a(cv), rule_b(cv, full)
    print(f"\nRULE A (highest accuracy; ties -> fewer calls): t = {A['t']:.4f}  skip {100*A['skip']:.1f}%  EX {A['acc']:.2f}  "
          f"Opus calls {100*A['opus_calls']:.1f}%  ${A['cost']:.4f}/q")
    print(f"RULE B (fewest calls within {DELTA} pt of always-Opus {full:.2f}): t = {B['t']:.4f}  skip {100*B['skip']:.1f}%  "
          f"EX {B['acc']:.2f}  Opus calls {100*B['opus_calls']:.1f}%  ${B['cost']:.4f}/q")
    for value in (0.05, 0.10, 0.25, 1.0):
        C = rule_c(cv, value)
        print(f"RULE C (value ${value:.2f} per right answer): t = {C['t']:.4f}  skip {100*C['skip']:.1f}%  EX {C['acc']:.2f}  "
              f"Opus calls {100*C['opus_calls']:.1f}%")
    out.update(full_handoff=full, rule_a=A, rule_b=B)
    # 3. leave-one-database-out: choose t on 7 DBs, apply to the held-out DB
    dbs = sorted({r["db"] for r in rows if r["db"]})
    print(f"\nLEAVE-ONE-DATABASE-OUT ({len(dbs)} DBs): threshold chosen on the other DBs, scored on the held-out one")
    for name, rule in (("A", lambda c, f: rule_a(c)), ("B", rule_b)):
        tot_ok = tot_n = tot_calls = 0
        ts = []
        for d in dbs:
            train = [r for r in rows if r["db"] != d]
            test = [r for r in rows if r["db"] == d]
            ctr = curve(train)
            t = rule(ctr, ctr[-1]["acc"])["t"]
            ts.append(t)
            for r in test:
                use_s1 = r["m"] >= t and t != -math.inf
                tot_ok += r["s1_ok"] if use_s1 else r["plus_ok"]
                tot_calls += not use_s1
                tot_n += 1
        print(f"  rule {name}: held-out EX {100 * tot_ok / tot_n:.2f} (always-Opus {full:.2f})  Opus calls "
              f"{100 * tot_calls / tot_n:.1f}%  thresholds per fold {[round(x, 3) for x in ts]}")
        out[f"lodo_{name}"] = {"ex": 100 * tot_ok / tot_n, "opus_calls": tot_calls / tot_n, "thresholds": ts}
    # 4. bootstrap by database for rule B's accuracy loss at its threshold
    rnd = random.Random(0)
    byd = {d: [r for r in rows if r["db"] == d] for d in dbs}
    losses = []
    for _ in range(2000):
        sample = [r for d in rnd.choices(dbs, k=len(dbs)) for r in byd[d]]
        t = B["t"]
        acc = sum(r["s1_ok"] if r["m"] >= t else r["plus_ok"] for r in sample) / len(sample)
        fullb = sum(r["plus_ok"] for r in sample) / len(sample)
        losses.append(100 * (acc - fullb))
    losses.sort()
    print(f"\nRule B threshold {B['t']:.4f}: EX change vs always-Opus, DB bootstrap 95% interval "
          f"[{losses[50]:+.2f}, {losses[1949]:+.2f}] (point {B['acc'] - full:+.2f})")
    out["rule_b_bootstrap_95"] = [losses[50], losses[1949]]
    # 5. how the frozen thresholds would split all 796 DECISION questions (System 1 margins only)
    for name, t in (("A", A["t"]), ("B", B["t"])):
        sk = 0.0 if t == -math.inf else sum(v["m"] >= t for v in s1.values()) / len(s1)
        print(f"Rule {name} threshold on all 796 DECISION questions: System 1 answers alone {100 * sk:.1f}%")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else x))


if __name__ == "__main__":
    main()
