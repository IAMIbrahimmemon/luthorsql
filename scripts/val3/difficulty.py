"""VAL3 Mac difficulty check (run after scripts/val3/build.py, BEFORE any validator training). Diagnostic, nothing fitted.

  .venv/bin/python scripts/val3/difficulty.py [--box runs/phase1/val3_lists/val3] [--lists data/val3/lists] \
      > runs/analysis/validator/val3_difficulty.out

Compares R1's own k=32 lists (labels as generated; original gold):
  OLD       runs/phase1/p1w/p1w/r1/cands_train.jsonl on the 42 train DBs (what VALHARD trained on)
  NEW/<src> the box lists of scripts/chains/val3_lists_box.sh, by source and kind (bird a = in-sample, bird b and
            synsql = out-of-sample), train split only; NEW-VAL = the val split
  DECISION  runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl, src in {r1, both}, DECISION ids only (RESERVED dropped at
            load, as in 04_list_difficulty.py)
Rows: greedy right, oracle, mixed (a live right and a live wrong class), greedy dead, right class at position >= 2 / >= 4
among mixed lists, mean live classes, greedy SQL == gold string. Then the BUILT lists (data/val3/lists): lists, hard
lists (R1's rank-0 class not right under the list's labels), x3 lists, masked, weighted hard share, and the bin mix next
to DECISION's (diagnostic only; the build never reads DECISION).
Pre-registered Mac gate (04_training_data.md #1), on the pooled train lists (old + new train split), raw R1 lists:
  greedy right within 50-68 %, right class at position >= 4 in >= 4 % of mixed lists, built hard lists >= 2,000.
Exit status 0 = gate PASS, 3 = gate FAIL (rebalance sources; do not launch).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as stt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VAL_DBS = {"citeseer", "disney", "professional_basketball", "trains"}
SETS = ("bird_rr", "bird_big", "synsql")


def norm(s):
    return re.sub(r"\s+", " ", (s or "").replace('"', "").replace("`", "")).strip().lower().rstrip(";")


def live(c):
    return bool(c.get("runs")) and bool(c.get("nonnull"))


def feats(r: dict) -> dict | None:
    """One raw generator list -> difficulty features (04_list_difficulty.feats, same definitions)."""
    cs = sorted([c for c in r.get("cands", []) if c.get("sql")], key=lambda c: c["rank"])
    if not cs:
        return None
    L = [c for c in cs if live(c)]
    classes, seen = [], set()
    for c in L:
        s = c.get("sig") or c["sql"]
        if s not in seen:
            seen.add(s)
            classes.append(bool(c.get("ok")))
    fr = next((i for i, ok in enumerate(classes) if ok), None)
    return {"greedy": bool(cs[0].get("ok")) and cs[0]["rank"] == 0, "dead": not live(cs[0]),
            "oracle": any(c.get("ok") for c in cs), "mixed": any(classes) and not all(classes), "first_right": fr,
            "ncls": len(classes), "gold_string": norm(cs[0]["sql"]) == norm(r.get("gold_sql"))}


def summarize(F: list[dict]) -> dict:
    pc = lambda xs: round(100 * sum(xs) / max(1, len(xs)), 1)  # noqa: E731
    m = [f for f in F if f["mixed"]]
    return {"lists": len(F), "greedy_right": pc([f["greedy"] for f in F]), "oracle": pc([f["oracle"] for f in F]),
            "mixed": pc([f["mixed"] for f in F]), "greedy_dead": pc([f["dead"] for f in F]),
            "mixed_right_pos_ge2": pc([f["first_right"] >= 2 for f in m]),
            "mixed_right_pos_ge4": pc([f["first_right"] >= 4 for f in m]),
            "live_classes_mean": round(stt.mean(f["ncls"] for f in F), 2) if F else 0.0,
            "greedy_is_gold_string": pc([f["gold_string"] for f in F])}


def frb(f: dict) -> str:
    x = f["first_right"]
    return "none" if x is None else "0" if x == 0 else "1" if x == 1 else "2-3" if x < 4 else "4+"


def unit(*parts) -> float:
    import hashlib
    return int(hashlib.md5(":".join(map(str, parts)).encode()).hexdigest()[:12], 16) / float(16 ** 12)


def jl(p: Path):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, default=ROOT / "runs/phase1/val3_lists/val3")
    ap.add_argument("--lists", type=Path, default=ROOT / "data/val3/lists")
    ap.add_argument("--meta", type=Path, default=ROOT / "data/val3/q/meta.jsonl")
    a = ap.parse_args()
    meta = {m["id"]: m for m in jl(a.meta)}
    dec_ids = set(json.load(open(ROOT / "data/sys1/heldout_v2/decision.json"))["ids"])
    groups = collections.OrderedDict()
    groups["OLD (R1 train lists, 42 DBs)"] = [f for r in jl(ROOT / "runs/phase1/p1w/p1w/r1/cands_train.jsonl")
                                             if r["db_id"] not in VAL_DBS and (f := feats(r))]
    new = collections.defaultdict(list)
    for s in SETS:
        for r in jl(a.box / f"cands_{s}_r1.jsonl"):
            if r.get("gold_error") or r["id"] not in meta:
                continue
            m = meta[r["id"]]
            f = feats(r)
            if f:
                f["id"] = r["id"]
                key = f"NEW {m['src']} {m['kind']} ({'in' if m['kind'] == 'a' else 'out-of'}-sample)"
                new[(m["split"], key)].append(f)
    for (split, key), F in sorted(new.items()):
        groups[key if split == "train" else f"NEW-VAL {key[4:]}"] = F
    dec = []
    for r in jl(ROOT / "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl"):
        if r["id"] not in dec_ids:
            continue  # RESERVED and everything else dropped at load
        r["cands"] = [c for c in r["cands"] if c.get("src", "r1") in ("r1", "both")]
        if (f := feats(r)):
            dec.append(f)
    groups["DECISION (R1 k=32, covered)"] = dec
    rows = {k: summarize(v) for k, v in groups.items()}
    cols = list(next(iter(rows.values())).keys())
    print("RAW R1 k=32 LISTS (labels as generated)")
    print(f"{'':48s}" + "".join(f"{c[:12]:>13s}" for c in cols))
    for k, r in rows.items():
        print(f"{k[:48]:48s}" + "".join(f"{r[c]:>13}" for c in cols))

    # ---- built lists ---------------------------------------------------------------------------------------------------
    tr = jl(a.lists / "train_lists.jsonl")
    va = jl(a.lists / "val_lists.jsonl")
    print("\nBUILT LISTS (data/val3/lists)")
    by = collections.defaultdict(list)
    for r in tr:
        v = r.get("v3", {})
        by[f"{v.get('src')} {v.get('kind')}"].append(r)
    tot_w = sum(r["weight"] for r in tr) or 1.0
    for k, rs in sorted(by.items()):
        hard = [r for r in rs if r["v3"]["hard"]]
        print(f"  train {k:10s} lists {len(rs):6d}  hard {len(hard):5d} ({100 * len(hard) / max(1, len(rs)):4.1f}%)  "
              f"x3 {sum(r['v3']['w_hard'] == 3.0 for r in rs):5d}  masked {sum(r['v3']['masked'] for r in rs):4d}  "
              f"weight share {100 * sum(r['weight'] for r in rs) / tot_w:5.1f}%  entries/list "
              f"{stt.mean(len(r['cands']) for r in rs):5.2f}")
    n_hard = sum(r["v3"]["hard"] for r in tr)
    w_hard = sum(r["weight"] for r in tr if r["v3"]["hard"]) / tot_w
    g_w = sum(r["weight"] for r in tr if not r["v3"]["hard"]) / tot_w
    print(f"  built train lists: greedy class right {100 * sum(not r['v3']['hard'] for r in tr) / max(1, len(tr)):.1f}% "
          f"(weighted {100 * g_w:.1f}%); DECISION mixed lists: greedy right "
          f"{100 * sum(f['greedy'] for f in dec if f['mixed']) / max(1, sum(f['mixed'] for f in dec)):.1f}%")
    print(f"  train total lists {len(tr)}, hard {n_hard}, weighted hard share {100 * w_hard:.1f}%, "
          f"entries {sum(len(r['cands']) for r in tr)}")
    print(f"  val lists {len(va)} on {len({r['db_id'] for r in va})} DBs "
          f"({collections.Counter(r['v3']['src'] for r in va)}); VALHARD val was 179 in-sample lists on 4 DBs")
    bins = collections.Counter(tuple(r["v3"]["bin"]) for r in tr)
    bw = collections.Counter()
    for r in tr:
        bw[tuple(r["v3"]["bin"])] += r["weight"]
    db = collections.Counter()
    for f in dec:
        if f["mixed"]:
            n = f["ncls"]
            db[(frb(f), "1" if n <= 1 else "2-4" if n <= 4 else "5-9" if n <= 9 else "10+")] += 1
    nd, nt = sum(db.values()) or 1, len(tr) or 1
    print("\n  bin (first right class, #classes): train % | train weighted % | DECISION mixed lists % (diagnostic)")
    for b in sorted(set(bins) | set(db)):
        print(f"    {str(b):18s} {100 * bins[b] / nt:6.1f} {100 * bw[b] / tot_w:8.1f} {100 * db[b] / nd:8.1f}")

    # ---- gate --------------------------------------------------------------------------------------------------------
    # SynSQL enters the train set subsampled (build.py --synsql-max): its raw lists count at the same rate
    bs = json.loads((a.lists / "build_stats.json").read_text()) if (a.lists / "build_stats.json").exists() else {}
    cnt = bs.get("counts", {})
    kept = bs.get("train", {}).get("by_source", {}).get("synsql_b", {}).get("lists", 0) + sum(
        v for k, v in cnt.items() if k.startswith("masked_dropped_synsql"))
    frac = min(1.0, kept / cnt["synsql_train_lists_all"]) if cnt.get("synsql_train_lists_all") else 1.0
    pooled = list(groups["OLD (R1 train lists, 42 DBs)"])
    for (split, key), F in new.items():
        if split == "train":
            pooled += [f for f in F if "synsql" not in key or unit("difficulty", f["id"]) < frac]
    print(f"\n(SynSQL raw lists counted at rate {frac:.3f} = built SynSQL train lists kept / built)")
    p = summarize(pooled)
    checks = [("pooled greedy right within 50-68 %", 50 <= p["greedy_right"] <= 68, p["greedy_right"]),
              ("right class at position >= 4 in >= 4 % of mixed lists", p["mixed_right_pos_ge4"] >= 4,
               p["mixed_right_pos_ge4"]),
              ("built hard train lists >= 2,000", n_hard >= 2000, n_hard)]
    print("\nGATE (pooled train lists: old + new train split)")
    for name, ok, v in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {v}")
    ok = all(c[1] for c in checks)
    print("GATE " + ("PASS" if ok else "FAIL: rebalance the sources; do not launch"))
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
