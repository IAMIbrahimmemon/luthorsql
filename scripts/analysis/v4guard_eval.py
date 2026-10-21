"""v4guard PREREG evaluation (runs/phase1/v4guard/PREREG.md): the v4desc generator re-decoded with the join-route guard
(LUTHOR_JOIN_GUARD=direct), no retraining. DECISION only (796 q, ORIGINAL gold); rows of any other id are skipped
while parsing (RESERVED stays blind); no BIRD dev file is read. Mac CPU.

  .venv/bin/python scripts/analysis/v4guard_eval.py                     # after the box: runs/phase1/v4guard/v4guard/
  .venv/bin/python scripts/analysis/v4guard_eval.py --dry               # dry run: unguarded v4desc files in every slot

PRIMARY: own k=8 + own validator, guarded (k8_guard_by_own.jsonl) vs unguarded v4desc (runs/phase1/v4desc/r1/rr/
dev_scored.jsonl, 52.51): delta, exact DB sign-flip (8 DBs), McNemar. Info: vs R3a 53.77 and the v3 seed mean 54.27.
Scoring-path check: k8_off_by_own.jsonl (the unguarded list through the same `score` path) must give 418 +- 2 hits;
if it does not, the GO test on k=8 uses the like-for-like delta (guarded - repro) instead of the 52.51 file.
GO for the v4fix retrain (decision rule, fixed before launch):
    GO if  guarded own-k8 >= 53.5 (>= 426 / 796 hits, i.e. >= +1.0 over 52.51)
       or  guarded k=32 --distinct + VALHARD-v4desc >= 59.0 (>= 470 / 796 hits)
    otherwise NO-GO (no v4fix).
SECONDARY (info): first pick, oracle@8, coverage; per DB; k=32 oracle vs 67.71; k=32 + own validator vs 57.66; k=32 +
VALHARD-v4desc vs 58.17 (both unguarded, runs/phase1/valhard_v4desc); decode determinism (guard off vs v4desc's list).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_eval import IDS, db_of  # noqa: E402
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402
from v4fix_eval import CONTROLS, frac_signflip, hits, load_ids, oracle  # noqa: E402

N_DEC = 796
GO_K8, GO_K32 = 53.5, 59.0
OFF = {"k8": "runs/phase1/v4desc/r1/rr/dev_scored.jsonl",
       "k32_list": "runs/phase1/v4desc/r1/rr/wide/cands_dev_k32.jsonl",
       "k32_own": "runs/phase1/valhard_v4desc/valhard/heldout_by_base.jsonl",
       "k32_valhard": "runs/phase1/valhard_v4desc/valhard/heldout_by_new.jsonl"}
NEW = {"k8": "k8_guard_by_own.jsonl", "k8_repro": "k8_off_by_own.jsonl", "k8_list": "k8_guard.jsonl",
       "k8_offgen": "k8_off_gen.jsonl", "k32_list": "k32_guard.jsonl", "k32_own": "k32_guard_by_own.jsonl",
       "k32_valhard": "k32_guard_by_valhard.jsonl"}


def min_hits(pct: float, n: int = N_DEC) -> int:
    """Smallest hit count whose percentage (2 dp, as reported) is >= pct."""
    k = math.ceil(pct * n / 100 - 1e-9)
    return k if round(100 * k / n, 2) >= pct else k + 1


def decide(k8_pct: float, k32_vh_pct: float | None) -> str:
    go8 = k8_pct >= GO_K8
    go32 = k32_vh_pct is not None and k32_vh_pct >= GO_K32
    return "GO (v4fix retrain)" if go8 or go32 else "NO-GO (no v4fix)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "runs/phase1/v4guard/v4guard", help="pulled box outputs")
    ap.add_argument("--dry", action="store_true", help="unguarded v4desc files in place of the guarded ones")
    a = ap.parse_args()
    ids = list(IDS)
    idset = set(ids)
    assert len(ids) == N_DEC, len(ids)
    DB = {i: d for i, d in db_of().items() if i in idset}
    pct = lambda h: 100 * sum(h.values()) / len(ids)

    def new(key):
        if a.dry:
            p = ROOT / {"k8": OFF["k8"], "k8_repro": OFF["k8"], "k32_list": OFF["k32_list"], "k32_own": OFF["k32_own"],
                        "k32_valhard": OFF["k32_valhard"]}.get(key, "")
            return p if p.is_file() else None
        p = a.dir / NEW[key]
        return p if p.exists() else None

    Off = load_ids(ROOT / OFF["k8"], idset)
    hoff = hits(Off, ids)
    C = {k: load_ids(ROOT / p, idset) for k, p in CONTROLS.items()}
    hc = {k: hits(S, ids) for k, S in C.items()}
    mean_c = {i: float(np.mean([hc[k][i] for k in C])) for i in ids}
    print(f"{'DRY RUN (unguarded files in the guarded slots)' if a.dry else 'v4guard'}: DECISION n={len(ids)}, "
          f"GO thresholds own-k8 >= {GO_K8} ({min_hits(GO_K8)} hits), k=32+VALHARD-v4desc >= {GO_K32} ({min_hits(GO_K32)} hits)")
    f8 = new("k8")
    if f8 is None:
        print(f"no {a.dir / NEW['k8']}: nothing to evaluate")
        return 1
    A = load_ids(f8, idset)
    ha = hits(A, ids)
    k8 = pct(ha)
    print(f"PRIMARY own k=8 + own validator: guarded {k8:.2f} ({sum(ha.values())}/{len(ids)}) vs unguarded v4desc "
          f"{pct(hoff):.2f} ({sum(hoff.values())})")
    c, q = cluster_signflip(hoff, ha, DB), question_signflip(hoff, ha)
    print(f"  delta {c['delta']:+.2f} (DB sign-flip p = {c['p']:.3f}, CI [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}], "
          f"McNemar +{q['wins']}/-{q['losses']})")
    for k in ("R3a",):
        c2 = cluster_signflip(hc[k], ha, DB)
        print(f"  info vs {k} {pct(hc[k]):.2f}: {c2['delta']:+.2f} (DB p = {c2['p']:.3f})")
    d, p, ci = frac_signflip({i: float(ha[i]) for i in ids}, mean_c, DB)
    print(f"  info vs v3 seed mean {100 * np.mean(list(mean_c.values())):.2f}: {d:+.2f} (DB p = {p:.3f})")
    # scoring-path check
    k8_go = k8
    fr = new("k8_repro")
    if fr is not None:
        hr = hits(load_ids(fr, idset), ids)
        nr, no = sum(hr.values()), sum(hoff.values())
        ok = abs(nr - no) <= 2
        print(f"scoring-path check: unguarded list via `score` {pct(hr):.2f} ({nr}) vs dev_scored {no}: "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            k8_go = pct(hoff) + (k8 - pct(hr))
            print(f"  -> like-for-like k=8 for the GO test: 52.51-equivalent {k8_go:.2f} (guarded - repro {k8 - pct(hr):+.2f})")
    else:
        print("scoring-path check: (no k8_off_by_own.jsonl)")
    # secondary
    g = {i: bool(A.get(i)) and bool(A[i]["cands"]) and bool(A[i]["cands"][0].get("ok")) for i in ids}
    goff = {i: bool(Off.get(i)) and bool(Off[i]["cands"]) and bool(Off[i]["cands"][0].get("ok")) for i in ids}
    print(f"secondary: first pick {pct(g):.2f} (unguarded {pct(goff):.2f}), oracle@8 {pct(oracle(A, ids, 8)):.2f} "
          f"(unguarded {pct(oracle(Off, ids, 8)):.2f}), coverage {len(A)} (unguarded {len(Off)})")
    print("per DB (guarded / unguarded): " + ", ".join(
        f"{k} {sum(ha[i] for i in ids if DB[i] == k)}/{sum(hoff[i] for i in ids if DB[i] == k)}" for k in sorted(set(DB.values()))))
    fl = new("k8_list")
    fo = new("k8_offgen")
    if fo is not None:
        Og = load_ids(fo, idset)
        same = sum(1 for i in Off if i in Og and [x.get("sql") for x in Og[i]["cands"]] == [x.get("sql") for x in Off[i]["cands"]])
        print(f"decode determinism (DECISION): guard-off regen identical SQL lists {same}/{len(Off)}, "
              f"oracle@8 {pct(oracle(Og, ids, 8)):.2f} vs {pct(oracle(Off, ids, 8)):.2f}")
    if fl is not None:
        Gl = load_ids(fl, idset)
        ch = sum(1 for i in Gl if i in Off and [x.get("sql") for x in Gl[i]["cands"]] != [x.get("sql") for x in Off[i]["cands"]])
        print(f"guard changed the k=8 SQL list on {ch}/{len(Gl)} DECISION questions")
    k32vh = None
    fw = new("k32_list")
    if fw is not None:
        W = load_ids(fw, idset)
        Wo = load_ids(ROOT / OFF["k32_list"], idset)
        print(f"k=32 oracle: guarded {pct(oracle(W, ids, 32)):.2f} vs unguarded {pct(oracle(Wo, ids, 32)):.2f}")
    for name in ("own", "valhard"):
        hb = hits(load_ids(ROOT / OFF[f"k32_{name}"], idset), ids)
        fv = new(f"k32_{name}")
        if fv is None:
            print(f"k=32 + {name} validator: (missing); unguarded {pct(hb):.2f}")
            continue
        hv = hits(load_ids(fv, idset), ids)
        cv = cluster_signflip(hb, hv, DB)
        print(f"k=32 + {name} validator: guarded {pct(hv):.2f} ({sum(hv.values())}) vs unguarded {pct(hb):.2f}: "
              f"{cv['delta']:+.2f} (DB p = {cv['p']:.3f})")
        if name == "valhard":
            k32vh = pct(hv)
    print(f"DECISION RULE: own-k8 {k8_go:.2f} (GO >= {GO_K8}), k=32+VALHARD-v4desc "
          f"{'n/a' if k32vh is None else f'{k32vh:.2f}'} (GO >= {GO_K32}) -> {decide(k8_go, k32vh)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
