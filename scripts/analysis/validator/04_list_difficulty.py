"""04 training_data: list difficulty of the validator's training lists vs DECISION lists (diagnostic, nothing fitted).

  .venv/bin/python scripts/analysis/validator/04_list_difficulty.py > runs/analysis/validator/04_list_difficulty.out

Lists: R1's own k=32 lists. TRAIN = runs/phase1/p1w/p1w/r1/cands_train.jsonl (in-sample for the generator; split into
the 42 train DBs and the 4 pinned val DBs, which the generator also trained on). DECISION = R1's own k=32 held-out lists
(heldout_union_by_r1.jsonl, src in {r1, both}; DECISION ids only, RESERVED rows dropped at load). Original gold labels.
Also: the built VALHARD train/val lists (data/valhard/r1), pick accuracy of the R1 and VALHARD validators on DECISION by
difficulty bin, and the effective sample size left if train lists were importance-weighted to DECISION's bin mix.
"""
from __future__ import annotations

import collections
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VAL_DBS = {"citeseer", "disney", "professional_basketball", "trains"}
DEC = set(json.load(open(ROOT / "data/sys1/heldout_v2/decision.json"))["ids"])


def norm(s):
    return re.sub(r"\s+", " ", (s or "").replace('"', "").replace("`", "")).strip().lower().rstrip(";")


def live(c):
    return c.get("runs") and c.get("nonnull")


def feats(r):
    cs = sorted([c for c in r["cands"] if c.get("sql")], key=lambda c: c["rank"])
    if not cs:
        return None
    L = [c for c in cs if live(c)]
    classes = []  # live result classes in generator order
    seen = {}
    for c in L:
        s = c.get("sig") or c["sql"]
        if s not in seen:
            seen[s] = len(classes)
            classes.append(bool(c.get("ok")))
    g = cs[0]
    first_right = next((i for i, ok in enumerate(classes) if ok), None)
    gold = norm(r.get("gold_sql"))
    right = [c for c in cs if c.get("ok")]
    return {
        "db": r["db_id"], "n": len(cs), "nlive": len(L), "ncls": len(classes),
        "greedy": bool(g.get("ok")), "greedy_dead": not live(g),
        "firstlive": bool(L and L[0].get("ok")),
        "oracle": any(c.get("ok") for c in cs),
        "mixed": any(classes) and not all(classes),
        "first_right": first_right,
        "frac_ok": sum(bool(c.get("ok")) for c in cs) / len(cs),
        "nright_cls": sum(classes),
        "greedy_is_gold": norm(g["sql"]) == gold,
        "any_is_gold": any(norm(c["sql"]) == gold for c in cs),
        "right_is_gold_frac": (sum(norm(c["sql"]) == gold for c in right) / len(right)) if right else None,
        "gap_cost": (L[1]["cost"] - L[0]["cost"]) if len(L) > 1 else None,
    }


def load_train():
    tr, va = [], []
    for line in open(ROOT / "runs/phase1/p1w/p1w/r1/cands_train.jsonl"):
        r = json.loads(line)
        assert r["id"] not in DEC
        f = feats(r)
        if f:
            (va if r["db_id"] in VAL_DBS else tr).append(f | {"id": r["id"]})
    return tr, va


def load_dec(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["id"] not in DEC:
            continue
        r["cands"] = [c for c in r["cands"] if c.get("src", "r1") in ("r1", "both")]
        out[r["id"]] = r
    return out


def summarize(name, F, n_total=None):
    n = len(F)
    m = [f for f in F if f["mixed"]]
    pc = lambda xs: 100 * sum(xs) / max(1, len(xs))  # noqa: E731
    fr = [f["first_right"] for f in m]
    row = {
        "lists": n,
        "greedy_right": pc([f["greedy"] for f in F]),
        "firstlive_right": pc([f["firstlive"] for f in F]),
        "oracle": pc([f["oracle"] for f in F]),
        "mixed_pct": pc([f["mixed"] for f in F]),
        "greedy_dead_pct": pc([f["greedy_dead"] for f in F]),
        "ncls_mean": st.mean(f["ncls"] for f in F),
        "ncls_median": st.median(f["ncls"] for f in F),
        "nlive_mean": st.mean(f["nlive"] for f in F),
        "frac_ok_mean": 100 * st.mean(f["frac_ok"] for f in F),
        "mixed_first_right_at0": pc([x == 0 for x in fr]),
        "mixed_first_right_ge2": pc([x >= 2 for x in fr]),
        "mixed_first_right_ge4": pc([x >= 4 for x in fr]),
        "greedy_is_gold_string": pc([f["greedy_is_gold"] for f in F]),
        "gold_string_in_list": pc([f["any_is_gold"] for f in F]),
        "right_cands_gold_string": 100 * st.mean(f["right_is_gold_frac"] for f in F if f["right_is_gold_frac"] is not None),
        "all_live_right_pct": pc([f["oracle"] and not f["mixed"] for f in F]),
        "no_right_pct": pc([not f["oracle"] for f in F]),
    }
    print(f"\n== {name} ==")
    for k, v in row.items():
        print(f"  {k:28s} {v:8.2f}" if isinstance(v, float) else f"  {k:28s} {v}")
    return row


def bins(f):
    """difficulty bin: greedy right/wrong x first-right-class rank bucket x #classes bucket."""
    fr = f["first_right"]
    frb = "none" if fr is None else ("0" if fr == 0 else "1" if fr == 1 else "2-3" if fr < 4 else "4+")
    nb = "1" if f["ncls"] <= 1 else "2-4" if f["ncls"] <= 4 else "5-9" if f["ncls"] <= 9 else "10+"
    return (frb, nb)


def main():
    tr, va = load_train()
    U = load_dec(ROOT / "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl")
    V = load_dec(ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl")
    dec = [feats(r) | {"id": i} for i, r in U.items() if feats(r)]
    res = {"train_42db": summarize("TRAIN lists (R1 in-sample, 42 train DBs, original gold)", tr),
           "val_4db": summarize("VAL lists (R1 in-sample, 4 pinned val DBs)", va),
           "decision": summarize(f"DECISION lists (R1 k=32, held-out DBs; covered {len(dec)} of {len(DEC)})", dec)}

    # per-DB spread of greedy-right on train, to see whether any train DB already looks test-like
    by = collections.defaultdict(list)
    for f in tr:
        by[f["db"]].append(f["greedy"])
    g = sorted((100 * sum(v) / len(v), d, len(v)) for d, v in by.items())
    print("\nTRAIN greedy-right per DB (lowest 8 / highest 4):", [(d, round(x, 1), n) for x, d, n in g[:8]],
          [(d, round(x, 1), n) for x, d, n in g[-4:]])
    dby = collections.defaultdict(list)
    for f in dec:
        dby[f["db"]].append(f["greedy"])
    print("DECISION greedy-right per DB:", sorted((d, round(100 * sum(v) / len(v), 1), len(v)) for d, v in dby.items()))
    res["train_db_greedy"] = {d: x for x, d, _ in g}

    # importance weighting train -> DECISION bin mix: effective sample size
    ct, cd = collections.Counter(bins(f) for f in tr), collections.Counter(bins(f) for f in dec)
    nt, nd = len(tr), len(dec)
    w = {b: (cd[b] / nd) / (ct[b] / nt) for b in ct if cd[b]}
    ws = [w.get(bins(f), 0.0) for f in tr]
    ess = sum(ws) ** 2 / sum(x * x for x in ws)
    print("\nBin mix (first-right-class rank, #classes): train% vs DECISION%")
    for b in sorted(set(ct) | set(cd)):
        print(f"  {str(b):18s} train {100*ct[b]/nt:5.1f}  dec {100*cd[b]/nd:5.1f}  w {w.get(b, 0):5.2f}")
    print(f"importance-weighted ESS: {ess:.0f} of {nt} train lists ({100*ess/nt:.1f}%)")
    res["ess"] = ess

    # validator pick accuracy on DECISION by bin (R1 validator rr and VALHARD rr)
    def pick_ok(r):
        cs = [c for c in r["cands"] if c.get("sql")]
        L = [c for c in cs if live(c)] or cs
        return bool(max(L, key=lambda c: c.get("rr", -1e9)).get("ok")) if L else False

    def follows_greedy(r):
        cs = sorted([c for c in r["cands"] if c.get("sql")], key=lambda c: c["rank"])
        L = [c for c in cs if live(c)] or cs
        p = max(L, key=lambda c: c.get("rr", -1e9))
        return (p.get("sig") or p["sql"]) == (L[0].get("sig") or L[0]["sql"])

    print("\nDECISION pick accuracy by bin (covered lists): n, R1-val %, VALHARD %, oracle-in-bin %, R1 picks first-live class %")
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for f in dec:
        i = f["id"]
        frb = bins(f)[0]
        key = ("greedy right" if f["firstlive"] else "greedy wrong") + f" / first right class {frb}"
        a = agg[key]
        a[0] += 1
        a[1] += pick_ok(U[i])
        a[2] += pick_ok(V[i]) if i in V else 0
        a[3] += f["oracle"]
        a[4] += follows_greedy(U[i])
    for k in sorted(agg):
        n, a, b, o, fg = agg[k]
        print(f"  {k:45s} n={n:4d}  R1 {100*a/n:5.1f}  VALHARD {100*b/n:5.1f}  oracle {100*o/n:5.1f}  follows-first {100*fg/n:5.1f}")
    res["dec_by_bin"] = {k: v for k, v in agg.items()}

    # built VALHARD lists
    for name in ("train_lists", "val_lists"):
        L = [json.loads(x) for x in open(ROOT / f"data/valhard/r1/{name}.jsonl")]
        n = len(L)
        ent = st.mean(len(r["cands"]) for r in L)
        w3 = sum(r.get("weight", 1) > 1 for r in L)
        base = collections.Counter(c.get("prov", "base") for r in L for c in r["cands"])
        gr = []
        for r in L:
            b = sorted([c for c in r["cands"] if c.get("prov", "base") == "base"], key=lambda c: c["rank"])
            if b:
                gr.append(bool(b[0].get("ok")) and b[0]["rank"] == 0)
        corr = sum(bool(r.get("gold_corrected")) for r in L)
        print(f"\nVALHARD {name}: {n} lists, {ent:.2f} entries/list, weight>1 on {w3} ({100*w3/n:.1f}%), "
              f"gold_corrected {corr}, entry provenance {dict(base)}, rank-0 base entry right {100*sum(gr)/max(1,len(gr)):.1f}%")
    json.dump(res, open(ROOT / "runs/analysis/validator/04_list_difficulty.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
