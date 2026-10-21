"""05 pairwise top-2/top-3 sizing on DECISION (diagnostic, no fitted parameters).

For each pool/validator: class = result sig over live candidates, class score = max rr. At margin gates, count
A = pick-only right, B = runner-only right (top-2) and B3 = right only at #3 (top-3 within gate); the accuracy a
second-stage pairwise judge needs for +1/+2/+3 points; Monte-Carlo power of the pre-registered bar (delta >= +1.5,
DB sign-flip p < 0.10) for an independent judge of accuracy p; clause-diff stats on gated one-right pairs.
DECISION ids only.  .venv/bin/python scripts/analysis/validator/05_pairwise_top2.py
"""
from __future__ import annotations
import collections, json, random, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/valhard"))
from luthor.stats import cluster_signflip  # noqa: E402
import sqldiff  # noqa: E402

dec = json.load(open(ROOT / "data/sys1/heldout_v2/decision.json")); IDS = list(dec["ids"]); IDSET = set(IDS)
tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
DB = {i: tr[int(i.rsplit("_", 1)[1])]["db_id"] for i in IDS}
N = len(IDS); PT = 100 / N
POOLS = {
    "R1 val, R1 k32": ROOT / "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl",
    "VALHARD, R1 k32": ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl",
    "VALHARD, k64+repair": ROOT / "runs/phase1/repair/repair/k64_repair_pool_by_valhard.jsonl",
}
GATES = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, float("inf")]


def load(p):
    out = {}
    for line in open(p):
        r = json.loads(line)
        if r["id"] not in IDSET:
            continue
        if "union_by_r1" in str(p) and any("src" in c for c in r["cands"]):  # R1 share of the R1 u R2 pool only
            r["cands"] = [c for c in r["cands"] if c.get("src") in ("r1", "both")]
        out[r["id"]] = r
    return out


def classes(cands):
    u = [c for c in cands if c.get("sql")]
    L = [c for c in u if c.get("runs") and c.get("nonnull")] or u
    d = {}
    for c in L:
        s = c.get("sig") or c["sql"]
        if s not in d or c.get("rr", -1e9) > d[s][0]:
            d[s] = (c.get("rr", -1e9), bool(c.get("ok")), c)
    return sorted(d.values(), key=lambda x: -x[0])


def need(k, A, B):
    return (k / PT + A) / (A + B) if A + B else float("nan")


def sim_power(rows, base, p, reps=400, seed=0):
    """rows: list of (id, kind) kind in {'A','B'} for gated one-right questions; judge right w.p. p independently."""
    rng = random.Random(seed); ok = 0; ds = []
    for _ in range(reps):
        new = dict(base)
        for i, kind in rows:
            right = rng.random() < p
            if kind == "A" and not right: new[i] = False
            if kind == "B" and right: new[i] = True
        c = cluster_signflip(base, new, DB)
        ds.append(c["delta"]); ok += c["delta"] >= 1.5 and c["p"] < 0.10
    return ok / reps, float(np.mean(ds))


def main():
    for name, path in POOLS.items():
        R = load(path)
        base = {}; per = {}
        for i in IDS:
            cl = classes(R[i]["cands"]) if i in R else []
            per[i] = cl; base[i] = bool(cl and cl[0][1])
        orc = sum(i in R and any(c.get("ok") for c in R[i]["cands"]) for i in IDS)
        t2 = sum(any(x[1] for x in per[i][:2]) for i in IDS); t3 = sum(any(x[1] for x in per[i][:3]) for i in IDS)
        print(f"\n=== {name}  covered {len(R)}/{N}  pick {sum(base.values())*PT:.2f}  top2-class oracle {t2*PT:.2f}  "
              f"top3 {t3*PT:.2f}  full oracle {orc*PT:.2f}")
        print("gate      n2   A    B  both-wrong  ceil2   pt-acc  need+1 need+2 need+3 | n3  B3 ceil3  need3+1 need3+2")
        for g in GATES:
            A = B = W = 0; n3 = B3 = A3 = 0
            for i in IDS:
                cl = per[i]
                if len(cl) < 2: continue
                if cl[0][0] - cl[1][0] < g:
                    a, b = cl[0][1], cl[1][1]
                    A += a and not b; B += b and not a; W += not a and not b
                if len(cl) >= 3 and cl[0][0] - cl[2][0] < g:
                    n3 += 1; A3 += cl[0][1]; B3 += (not cl[0][1]) and (cl[1][1] or cl[2][1])
                elif len(cl) >= 2 and cl[0][0] - cl[1][0] < g:  # only 2 within gate
                    n3 += 1; A3 += cl[0][1]; B3 += (not cl[0][1]) and cl[1][1]
            f = lambda x: f"{x:.3f}" if x <= 1 else " >1  "  # noqa: E731
            n2 = A + B + W
            print(f"<{g:<5}  {n2:4d} {A:4d} {B:4d} {W:6d}     +{B*PT:4.2f}  {A/(A+B) if A+B else 0:.3f}   "
                  f"{f(need(1,A,B))}  {f(need(2,A,B))}  {f(need(3,A,B))} | {n3:3d} {B3:3d} +{B3*PT:4.2f}  "
                  f"{f(need(1,A3,B3))}   {f(need(2,A3,B3))}")
        # per-DB at gate < 1
        A_ = collections.Counter(); B_ = collections.Counter()
        rows = {}
        for g in (0.5, 1.0, 2.0):
            rows[g] = []
            for i in IDS:
                cl = per[i]
                if len(cl) >= 2 and cl[0][0] - cl[1][0] < g and cl[0][1] != cl[1][1]:
                    rows[g].append((i, "A" if cl[0][1] else "B"))
                    if g == 1.0: (A_ if cl[0][1] else B_)[DB[i]] += 1
        print("per-DB at gate<1 (A/B): " + ", ".join(f"{d} {A_[d]}/{B_[d]}" for d in sorted(set(DB.values()))))
        if name != "R1 val, R1 k32":
            print("MC power (independent judge acc p; pass = delta>=+1.5 & DB p<0.10):")
            for g in (0.5, 1.0, 2.0):
                s = []
                for p in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
                    pw, md = sim_power(rows[g], base, p, reps=300)
                    s.append(f"p={p:.2f}: E[d]={md:+.2f} P(pass)={pw:.2f}")
                print(f"  gate<{g}: " + "; ".join(s))
        if name == "VALHARD, R1 k32":
            ct = collections.Counter(); nc = collections.Counter(); bk = collections.Counter(); ln = []
            for i, kind in rows[1.0]:
                cl = per[i]; right = cl[0][2] if kind == "A" else cl[1][2]; wrong = cl[1][2] if kind == "A" else cl[0][2]
                t, b, k = sqldiff.classify(right["sql"], wrong["sql"])
                ct[t] += 1; bk[(b, kind)] += 1; nc[min(k, 4)] += 1
                ln.append((len(right["sql"]) + len(wrong["sql"])))
            print("gate<1 one-right pairs: n components differing:", dict(sorted(nc.items())))
            print("  bucket x (A=pick right,B=runner right):", dict(sorted(bk.items())))
            print("  types:", ct.most_common(14))
            print(f"  both-SQL chars median {int(np.median(ln))}")
            lab = json.load(open(ROOT / "scripts/analysis/system1_pick_gap_manual.json"))
            v = collections.Counter()
            for i, kind in rows[1.0]:
                k = i.rsplit("_", 1)[1]
                if kind == "B" and k in lab: v[lab[k][1]] += 1
            print("  VALHARD runner-right(B) at gate<1 that are in the R1 hand-labelled misses, by verdict:", dict(v))


if __name__ == "__main__" and "--proxies" not in sys.argv:
    main()


def proxies():
    """How well do OTHER existing judges do on VALHARD's gated one-right top-2 pairs? (no new model, no fitting)
    (a) the R1 validator (a second 7B pointwise cross-encoder) and (b) the zero-shot 27B duel verdicts
    (runs/duel/calls.jsonl, both orders, only where the duel pair has the same two result sigs)."""
    V = load(POOLS["VALHARD, R1 k32"]); R = load(POOLS["R1 val, R1 k32"])
    calls = collections.defaultdict(list)
    for l in open(ROOT / "runs/duel/calls.jsonl"):
        c = json.loads(l); calls[c["id"]].append(c["choice"])
    pairs = {}
    for l in open(ROOT / "data/duel/decision_pairs.jsonl"):
        p = json.loads(l)
        if p["id"] in IDSET: pairs[p["id"]] = p
    print("\n=== proxies on VALHARD gated one-right top-2 pairs")
    for g in (0.5, 1.0, 2.0, float("inf")):
        n = r1c = 0; dn = dc = 0
        for i in IDS:
            cl = classes(V[i]["cands"]) if i in V else []
            if len(cl) < 2 or cl[0][0] - cl[1][0] >= g or cl[0][1] == cl[1][1]: continue
            sA, sB = cl[0][2]["sig"], cl[1][2]["sig"]; right = sA if cl[0][1] else sB
            rc = {c[2]["sig"]: c[0] for c in classes(R[i]["cands"])} if i in R else {}
            if sA in rc and sB in rc:
                n += 1; r1c += (sA if rc[sA] > rc[sB] else sB) == right
            p = pairs.get(i)
            if p and {p["sig_pick"], p["sig_runner"]} == {sA, sB} and len(set(calls[i])) == 1:
                ch = p["sig_pick"] if calls[i][0] == "pick" else p["sig_runner"]
                dn += 1; dc += ch == right
        print(f"gate<{g}: R1-validator acc {r1c}/{n} = {r1c/max(n,1):.3f};  27B zero-shot duel (agreeing) {dc}/{dn}")


if __name__ == "__main__" and "--proxies" in sys.argv:
    proxies()
