"""System-1 repair stage (runs/phase1/repair/PREREG.md): ONE grammar generator (R1, grammar v3) + ONE validator, on
DECISION (796 q, 8 DBs, original gold). Three candidate pools from the same generator:

  k32      R1's own k=32 held-out list (the VALHARD / pick-gap baseline files, scored before this stage)
  k64      k=64 distinct lazy k-best list + anything of k32 it lacks (repair_box.sh stage 2)
  k64r     k64 + the repair candidates (constrained re-decoding: F1-F4 feedback bans, E1-E3 evidence forcing)

each scored by two validators: VALHARD (runs/phase1/valhard/valhard/rr) and the R1 wide validator (p1w r1/rr).

  .venv/bin/python scripts/analysis/repair_eval.py                      # after the box: runs/phase1/repair/repair/
  .venv/bin/python scripts/analysis/repair_eval.py --dry-run            # code path only: k32 files stand in for k64/k64r
  .venv/bin/python scripts/analysis/repair_eval.py --json runs/phase1/repair/eval.json

Rows whose id is not in DECISION are dropped at load (RESERVED stays blind). Pick = max rr over live candidates (runs
and nonnull; all if none live); uncovered DECISION questions count wrong in every arm. k32 r1w file: src in {r1, both}
(the R1 share of the R1 u R2 union, as valhard_eval.py); the new pools are R1-only and are not filtered.
PRIMARY: VALHARD pick on k64r vs VALHARD pick on k32 (59.05): delta, exact DB sign-flip p (8 DBs).
  PASS delta >= +1.5 and p < 0.10; KILL delta < +0.5; otherwise NO PASS.
Secondary (info): every pool x validator cell vs the same baseline (DB p, McNemar), oracle per pool, per-DB deltas,
the R1-validator within-validator deltas, repair diagnostics (coverage, variants, oracle gain, picks of repair /
beyond-k32 candidates), and controls (candidate-set nesting k32 <= k64 <= k64r; re-scoring drift = the pick restricted
to the k32 candidates inside each new file vs the k32 file).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402

DECISION = ROOT / "data/sys1/heldout_v2/decision.json"
NEW_DIR = ROOT / "runs/phase1/repair/repair"
K32 = {"valhard": ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl",
       "r1w": ROOT / "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl"}
POOL_FILE = {"k64": "k64_pool_by_{v}.jsonl", "k64r": "k64_repair_pool_by_{v}.jsonl"}
VALS = ("valhard", "r1w")
ARMS = ("k32", "k64", "k64r")
PASS_DELTA, PASS_P, KILL_DELTA = 1.5, 0.10, 0.5
FORBIDDEN = ("bird_dev", "gate.json", "sealed", "/score_")


def guard(p: Path) -> Path:
    if any(f in str(p) for f in FORBIDDEN):
        raise SystemExit(f"refusing to read {p}: BIRD dev / gate / sealed files are off limits")
    return p


def load(path: Path, ids: set, r1_share: bool) -> dict:
    out = {}
    for line in open(guard(path)):
        r = json.loads(line)
        if r["id"] not in ids:
            continue  # never look at non-DECISION rows
        if r1_share and any("src" in c for c in r["cands"]):
            r["cands"] = [c for c in r["cands"] if c.get("src") in ("r1", "both")]
        out[r["id"]] = r
    return out


def live(cands):
    L = [c for c in cands if c.get("runs") and c.get("nonnull")]
    return L or cands


def pick(cands):
    u = [c for c in cands if c.get("sql")]
    if not u:
        return None
    return max(live(u), key=lambda c: c.get("rr", -1e9))


def verdict(delta, p):
    if delta >= PASS_DELTA and p < PASS_P:
        return "PASS"
    return "KILL" if delta < KILL_DELTA else "NO PASS"


def sqls(r) -> set:
    return {c["sql"] for c in r["cands"] if c.get("sql")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=NEW_DIR, help="pulled box outputs (k64[_repair]_pool_by_<val>.jsonl)")
    ap.add_argument("--k32-valhard", type=Path, default=K32["valhard"])
    ap.add_argument("--k32-r1w", type=Path, default=K32["r1w"])
    ap.add_argument("--dry-run", action="store_true", help="k32 files stand in for the k64 / k64r arms (delta 0)")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    dec = json.load(open(DECISION))
    ids = list(dec["ids"])
    idset = set(ids)
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    db = {i: tr[int(i.rsplit("_", 1)[1])]["db_id"] for i in ids}
    assert collections.Counter(db.values()) == collections.Counter(dec["database_counts"])
    n = len(ids)
    k32f = {"valhard": a.k32_valhard, "r1w": a.k32_r1w}
    files, S = {}, {}
    for v in VALS:
        files[("k32", v)] = (k32f[v], True)
        for arm in ("k64", "k64r"):
            files[(arm, v)] = (k32f[v], True) if a.dry_run else (a.dir / POOL_FILE[arm].format(v=v), False)
    missing = [str(f) for f, _ in files.values() if not f.exists()]
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))
    for key, (f, share) in files.items():
        S[key] = load(f, idset, share)
    out = {"n": n, "dry_run": a.dry_run, "files": {f"{k[0]}/{k[1]}": str(f) for k, (f, _) in files.items()}}
    if a.dry_run:
        print("DRY RUN: k32 files stand in for k64 and k64r (every delta must be 0)")

    # ---- controls: same candidate SQL across validators per pool; nesting k32 <= k64 <= k64r --------------------
    ctl = {}
    for arm in ARMS:
        A, B = S[(arm, "valhard")], S[(arm, "r1w")]
        ctl[f"{arm}_validator_set_mismatch"] = sum(1 for i in ids if (i in A) != (i in B) or
                                                   (i in A and sqls(A[i]) != sqls(B[i])))
    for lo, hi in (("k32", "k64"), ("k64", "k64r")):
        L, H = S[(lo, "valhard")], S[(hi, "valhard")]
        ctl[f"{lo}_not_subset_of_{hi}"] = sum(1 for i in L if i not in H or not sqls(L[i]) <= sqls(H[i]))
    print(f"DECISION n={n}, {len(set(db.values()))} DBs; covered " +
          ", ".join(f"{arm} {len(S[(arm, 'valhard')])}" for arm in ARMS))
    print("controls (must be 0): " + ", ".join(f"{k} {v}" for k, v in ctl.items()))
    out["controls"] = ctl

    # ---- hits ---------------------------------------------------------------------------------------------------
    H, ORC, P = {}, {}, {}
    for key, sc in S.items():
        H[key], P[key] = {}, {}
        for i in ids:
            p = pick(sc[i]["cands"]) if i in sc else None
            P[key][i] = p
            H[key][i] = bool(p and p.get("ok"))
    for arm in ARMS:  # oracle of a pool (the same candidate set under both validators: controls above)
        sc = S[(arm, "valhard")]
        ORC[arm] = {i: i in sc and any(c.get("ok") for c in sc[i]["cands"]) for i in ids}
    pct = lambda h: 100 * sum(h.values()) / n  # noqa: E731
    base = H[("k32", "valhard")]

    def cmp(h, ref):
        c, q = cluster_signflip(ref, h, db), question_signflip(ref, h)
        return {"delta": c["delta"], "p": c["p"], "ci": c["ci"], "mcnemar_p": q["p"], "wins": q["wins"],
                "losses": q["losses"],
                "per_db": {d: sum(h[i] - ref[i] for i in ids if db[i] == d) for d in sorted(set(db.values()))}}

    prim = cmp(H[("k64r", "valhard")], base)
    v = verdict(prim["delta"], prim["p"])
    print("\nPRIMARY (R1 generator, VALHARD validator picks): k=64 + repair pool vs k=32 pool")
    print(f"  k32 {pct(base):6.2f}   k64+repair {pct(H[('k64r', 'valhard')]):6.2f}   "
          f"oracle k32 {pct(ORC['k32']):6.2f} -> k64+repair {pct(ORC['k64r']):6.2f}")
    print(f"  delta {prim['delta']:+.2f}  DB sign-flip p={prim['p']:.3f}  CI [{prim['ci'][0]:+.2f}, {prim['ci'][1]:+.2f}]  "
          f"McNemar p={prim['mcnemar_p']:.3f} (+{prim['wins']}/-{prim['losses']})")
    print(f"  -> {v}   (PASS: delta >= +{PASS_DELTA} and p < {PASS_P}; KILL: delta < +{KILL_DELTA})")
    print("  per DB (questions): " + ", ".join(f"{d} {x:+d}" for d, x in prim["per_db"].items()))
    out["primary"] = {"base_k32_valhard": pct(base), "k64r_valhard": pct(H[("k64r", "valhard")]), **prim, "verdict": v}

    # ---- secondary: every cell -------------------------------------------------------------------------------------
    print("\nSECONDARY (info): pick % by pool x validator; delta / DB p / McNemar vs VALHARD on k32")
    print(f"  {'pool':6} {'oracle':>7} | {'VALHARD':>8} {'d':>6} {'p':>6} {'McN':>6} | {'R1 val':>8} {'d':>6} {'p':>6} {'McN':>6}")
    cells = {}
    for arm in ARMS:
        row = f"  {arm:6} {pct(ORC[arm]):7.2f} |"
        for val in VALS:
            h = H[(arm, val)]
            c = cmp(h, base)
            cells[f"{arm}/{val}"] = {"pick": pct(h), **c}
            row += f" {pct(h):8.2f} {c['delta']:+6.2f} {c['p']:6.3f} {c['mcnemar_p']:6.3f} |"
        print(row.rstrip("|"))
    out["cells"] = cells
    out["oracle"] = {arm: pct(ORC[arm]) for arm in ARMS}
    print("  within the R1 validator (vs R1 validator on k32 = the pick-gap 57.29):")
    for arm in ("k64", "k64r"):
        c = cmp(H[(arm, "r1w")], H[("k32", "r1w")])
        out[f"r1w_{arm}_vs_r1w_k32"] = c
        print(f"    {arm}: {c['delta']:+.2f} (DB p={c['p']:.3f}, McNemar p={c['mcnemar_p']:.3f}, +{c['wins']}/-{c['losses']})")
    for arm in ("k64r",):
        c = cmp(H[(arm, "valhard")], H[("k64", "valhard")])
        out["valhard_k64r_vs_k64"] = c
        print(f"  repair alone (VALHARD, k64+repair vs k64): {c['delta']:+.2f} (DB p={c['p']:.3f}, "
              f"McNemar p={c['mcnemar_p']:.3f}, +{c['wins']}/-{c['losses']})")
    print("  per DB, VALHARD k64 vs k32: " + ", ".join(f"{d} {x:+d}" for d, x in cells["k64/valhard"]["per_db"].items()))

    # ---- repair / width diagnostics (labels on DECISION only) -------------------------------------------------------
    K32S = {i: sqls(r) for i, r in S[("k32", "valhard")].items()}
    R = S[("k64r", "valhard")]
    rep_q = [i for i in ids if i in R and any(c.get("src") == "repair" for c in R[i]["cands"])]
    var = collections.Counter(c.get("repair", "?").split(":")[0] for i in rep_q for c in R[i]["cands"]
                              if c.get("src") == "repair")
    var_ok = collections.Counter(c.get("repair", "?").split(":")[0] for i in rep_q for c in R[i]["cands"]
                                 if c.get("src") == "repair" and c.get("ok"))
    only_rep = sum(1 for i in rep_q if any(c.get("ok") for c in R[i]["cands"] if c.get("src") == "repair")
                   and not any(c.get("ok") for c in R[i]["cands"] if c.get("src") != "repair"))
    K64 = S[("k64", "valhard")]
    only_wide = sum(1 for i in ids if i in K64 and not ORC["k32"][i] and ORC["k64"][i])
    diag = {"questions_with_repair": len(rep_q), "repair_cands": sum(var.values()), "variants": dict(var),
            "variants_ok": dict(var_ok), "oracle_gain_from_repair_q": only_rep, "oracle_gain_from_k64_q": only_wide}
    print(f"\nrepair: {len(rep_q)} DECISION questions got {sum(var.values())} repair candidates; variants {dict(var)}; "
          f"ok by variant {dict(var_ok)}")
    print(f"  oracle gain: k32->k64 +{only_wide} questions, k64->k64+repair +{only_rep} questions")
    for val in VALS:
        pk = P[("k64r", val)]
        pr = [i for i in ids if pk[i] and pk[i].get("src") == "repair"]
        pw = [i for i in ids if pk[i] and i in K32S and pk[i]["sql"] not in K32S[i] and pk[i].get("src") != "repair"]
        d = {"picked_repair": len(pr), "picked_repair_ok": sum(H[("k64r", val)][i] for i in pr),
             "picked_beyond_k32": len(pw), "picked_beyond_k32_ok": sum(H[("k64r", val)][i] for i in pw)}
        diag[f"picks_{val}"] = d
        print(f"  {val:7} on k64+repair picks a repair candidate {d['picked_repair']}x ({d['picked_repair_ok']} right), "
              f"a k64-only candidate {d['picked_beyond_k32']}x ({d['picked_beyond_k32_ok']} right)")
    out["diagnostics"] = diag

    # ---- control: re-scoring drift (pick restricted to the k32 candidates inside each new file) ----------------------
    drift = {}
    for arm in ("k64", "k64r"):
        for val in VALS:
            sc = S[(arm, val)]
            h = {}
            for i in ids:
                if i not in sc or i not in K32S:
                    h[i] = False
                    continue
                p = pick([c for c in sc[i]["cands"] if c.get("sql") in K32S[i]])
                h[i] = bool(p and p.get("ok"))
            drift[f"{arm}/{val}"] = pct(h) - pct(H[("k32", val)])
    print("control, k32-restricted pick inside each new file minus the k32 file (scoring-context drift, ~0 expected): "
          + ", ".join(f"{k} {x:+.2f}" for k, x in drift.items()))
    out["rescoring_drift"] = drift
    if v == "PASS" and abs(drift["k64r/valhard"]) > 0.5:  # PREREG: a PASS stands only with |drift| <= 0.5
        out["primary"]["verdict"] = "PASS NOT CONFIRMED (re-scoring drift > 0.5)"
        print(f"-> primary verdict downgraded: {out['primary']['verdict']}")
    print(f"\nVERDICT {out['primary']['verdict']}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
