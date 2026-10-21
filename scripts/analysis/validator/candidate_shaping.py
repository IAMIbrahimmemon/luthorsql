"""Candidate-list shaping for the System-1 validator (research note 10_candidate_shaping).

DECISION only (796 q, 8 DBs; uncovered questions count wrong). Reads the k32 / k64 / k64+repair pools scored by VALHARD
and the R1 wide validator (runs/phase1/repair/eval.json lists the files). No GPU: every rule here re-uses the stored
per-candidate validator score `rr`; nothing is re-scored. A rule with a fitted knob is reported leave-one-DB-out
(knob chosen on 7 DBs by pick count, applied to the 8th).

  .venv/bin/python scripts/analysis/validator/candidate_shaping.py [--json runs/analysis/validator/10_candidate_shaping.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from luthor.stats import cluster_signflip  # noqa: E402

DEC = ROOT / "data/sys1/heldout_v2/decision.json"
FILES = {
    ("k32", "valhard"): (ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl", True),
    ("k32", "r1w"): (ROOT / "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl", True),
    ("k64", "valhard"): (ROOT / "runs/phase1/repair/repair/k64_pool_by_valhard.jsonl", False),
    ("k64", "r1w"): (ROOT / "runs/phase1/repair/repair/k64_pool_by_r1w.jsonl", False),
    ("k64r", "valhard"): (ROOT / "runs/phase1/repair/repair/k64_repair_pool_by_valhard.jsonl", False),
    ("k64r", "r1w"): (ROOT / "runs/phase1/repair/repair/k64_repair_pool_by_r1w.jsonl", False),
}
FORBIDDEN = ("bird_dev", "gate.json", "sealed", "/score_", "poolA_by_valhard", "look_")


def load(path, ids, r1_share):
    assert not any(f in str(path) for f in FORBIDDEN)
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["id"] not in ids:
            continue
        if r1_share and any("src" in c for c in r["cands"]):
            r["cands"] = [c for c in r["cands"] if c.get("src") in ("r1", "both")]
        cs = [c for c in r["cands"] if c.get("sql")]
        for c in cs:
            if c.get("rr") is None:
                c["rr"] = -1e9
        out[r["id"]] = cs
    return out


def live(cs):
    L = [c for c in cs if c.get("runs") and c.get("nonnull")]
    return L or cs


DEGEN = re.compile(r"^1 row x 1 col \((integer|real|null)\); first: \((0|0\.0|None),?\)$")


def degenerate(c):
    return bool(DEGEN.match(c.get("res") or ""))


# ---- pick rules: each returns the picked candidate (or None) ---------------------------------------------------
def pick_max(cs):
    L = live(cs)
    return max(L, key=lambda c: c["rr"]) if L else None


def groups(L):
    g = collections.defaultdict(list)
    for c in L:
        g[c["sig"]].append(c)
    return g


def pick_group(cs, agg, tau=1.0, beta=0.0):
    """Class-level pick: score each result class by agg over its members' rr (+ beta*log size), return its best member."""
    L = live(cs)
    if not L:
        return None
    best, bs = None, -1e18
    for sig, m in groups(L).items():
        s = [c["rr"] for c in m]
        mx = max(s)
        if agg == "max":
            v = mx
        elif agg == "lme":  # log-mean-exp at temperature tau (VALHARD's group loss uses tau=1)
            v = mx + tau * math.log(sum(math.exp((x - mx) / tau) for x in s) / len(s))
        elif agg == "lse":  # log-sum-exp: rewards class size
            v = mx + tau * math.log(sum(math.exp((x - mx) / tau) for x in s))
        elif agg == "mean":
            v = sum(s) / len(s)
        else:
            raise ValueError(agg)
        v += beta * math.log(len(m))
        if v > bs:
            bs, best = v, max(m, key=lambda c: c["rr"])
    return best


def gen_order(cs):
    """Generator order: the k-best rank of the generator's own list; k32-only rescues and repair cands go after."""
    def key(c):
        src = c.get("src")
        return (0 if src in (None, "r1", "both") else 1 if src == "k32" else 2, c.get("rank", 999), c.get("cost", 0))
    return sorted(cs, key=key)


def pick_topn(cs, n):
    return pick_max(gen_order(cs)[:n])


def pick_prune(cs, drop_single=False, drop_degen=False, drop_repair=False, min_votes=1):
    L = live(cs)
    K = L
    if drop_repair:
        K = [c for c in K if c.get("src") != "repair"] or K
    if drop_degen:
        K = [c for c in K if not degenerate(c)] or K
    if drop_single:
        g = groups(K)
        K2 = [c for c in K if len(g[c["sig"]]) >= min_votes]
        K = K2 or K
    return max(K, key=lambda c: c["rr"]) if K else None


def pick_hyst(cs, base_sqls, delta):
    """Hysteresis: keep the k32 pick unless a new (beyond-k32) candidate beats it by more than delta."""
    L = live(cs)
    if not L:
        return None
    old = [c for c in L if c["sql"] in base_sqls]
    new = [c for c in L if c["sql"] not in base_sqls]
    bo = max(old, key=lambda c: c["rr"]) if old else None
    bn = max(new, key=lambda c: c["rr"]) if new else None
    if bo is None:
        return bn
    if bn is not None and bn["rr"] > bo["rr"] + delta:
        return bn
    return bo


def gcost(c, cs, mode):
    """Generator cost of a candidate. Repair candidates carry a cost from their own constrained list (a different
    scale): mode 'raw' keeps it, 'cap' gives them the question's max non-repair cost (no prior credit)."""
    if c.get("src") == "repair" and mode == "cap":
        return max((x.get("cost", 0.0) for x in cs if x.get("src") != "repair"), default=0.0)
    return c.get("cost", 0.0)


def pick_cost(cs, gamma, mode="cap", degen=False, beta=0.0):
    L = live(cs)
    if degen:
        L = [c for c in L if not degenerate(c)] or L
    if not L:
        return None
    g = groups(L)
    return max(L, key=lambda c: c["rr"] - gamma * gcost(c, cs, mode) + beta * math.log(len(g[c["sig"]])))


def pick_rankpen(cs, gamma):
    L = live(cs)
    if not L:
        return None
    o = {id(c): k for k, c in enumerate(gen_order(cs))}
    return max(L, key=lambda c: c["rr"] - gamma * math.log1p(o[id(c)]))


def class_table(cs, mode="cap"):
    """Live result classes sorted by max rr: (max rr, ok, size, min generator cost, best member)."""
    L = live(cs)
    t = []
    for m in groups(L).values():
        b = max(m, key=lambda c: c["rr"])
        t.append((b["rr"], bool(b["ok"]), len(m), min(gcost(c, cs, mode) for c in m), b))
    return sorted(t, key=lambda x: -x[0])


def pick_tiebreak(cs, margin):
    """If the validator's top-2 classes are within `margin` rr, take the one the generator prefers (lower min cost)."""
    t = class_table(cs)
    if not t:
        return None
    if len(t) > 1 and t[0][0] - t[1][0] < margin and t[1][3] < t[0][3]:
        return t[1][4]
    return t[0][4]


def pick_linear(cs, beta, gamma):
    """rr + beta*log(class size) - gamma*generator cost (cost = generator -logprob gap to its rank 0)."""
    L = live(cs)
    if not L:
        return None
    g = groups(L)
    return max(L, key=lambda c: c["rr"] + beta * math.log(len(g[c["sig"]])) - gamma * c.get("cost", 0.0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=ROOT / "runs/analysis/validator/10_candidate_shaping.json")
    a = ap.parse_args(argv)
    dec = json.load(open(DEC))
    ids = list(dec["ids"])
    idset = set(ids)
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    db = {i: tr[int(i.rsplit("_", 1)[1])]["db_id"] for i in ids}
    dbs = sorted(set(db.values()))
    S = {k: load(f, idset, sh) for k, (f, sh) in FILES.items()}
    n = len(ids)
    K32SQL = {i: {c["sql"] for c in cs} for i, cs in S[("k32", "valhard")].items()}
    out = {"n": n}

    def hits(pool, rule):
        sc = S[pool]
        h = {}
        for i in ids:
            p = rule(sc[i], i) if i in sc else None
            h[i] = bool(p and p.get("ok"))
        return h

    pct = lambda h, sub=None: 100 * sum(h[i] for i in (sub or ids)) / n  # noqa: E731
    base = hits(("k32", "valhard"), lambda cs, i: pick_max(cs))
    ref = {v: hits(("k32", v), lambda cs, i: pick_max(cs)) for v in ("valhard", "r1w")}

    def cmpd(h, r=None):
        r = r or base
        c = cluster_signflip(r, h, db)
        w = sum(1 for i in ids if h[i] and not r[i])
        l_ = sum(1 for i in ids if r[i] and not h[i])
        return {"pick": round(pct(h), 2), "delta": round(c["delta"], 2), "p": round(c["p"], 3), "w": w, "l": l_}

    rows = {}

    def rep(name, pool, rule, r=None):
        h = hits(pool, rule)
        rows[name] = cmpd(h, r)
        x = rows[name]
        print(f"  {name:58s} {x['pick']:6.2f}  d {x['delta']:+5.2f}  p {x['p']:.3f}  +{x['w']}/-{x['l']}")
        return h

    print("== baselines (max rr over live) vs VALHARD k32 59.05")
    for pool in ("k32", "k64", "k64r"):
        for v in ("valhard", "r1w"):
            rep(f"{pool}/{v} max", (pool, v), lambda cs, i: pick_max(cs))

    print("\n== oracle / class stats")
    for pool in ("k32", "k64", "k64r"):
        sc = S[(pool, "valhard")]
        orc = sum(1 for i in ids if i in sc and any(c["ok"] for c in sc[i]))
        lorc = sum(1 for i in ids if i in sc and any(c["ok"] for c in live(sc[i])))
        ncl = [len(groups(live(sc[i]))) for i in sc]
        nc = [len(sc[i]) for i in sc]
        out[f"stats_{pool}"] = {"oracle": 100 * orc / n, "live_oracle": 100 * lorc / n,
                                "mean_cands": sum(nc) / len(nc), "mean_live_classes": sum(ncl) / len(ncl)}
        print(f"  {pool}: oracle {100*orc/n:.2f} live-oracle {100*lorc/n:.2f} cands {sum(nc)/len(nc):.1f} "
              f"live classes {sum(ncl)/len(ncl):.1f}")

    print("\n== top-N by generator rank, then validator max (VALHARD; R1 val)")
    for v in ("valhard", "r1w"):
        for pool in ("k32", "k64", "k64r"):
            for N in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 999):
                if pool == "k32" and N > 32:
                    continue
                rep(f"{pool}/{v} top{N}", (pool, v), lambda cs, i, N=N: pick_topn(cs, N))
        # oracle by top-N
    for pool in ("k64",):
        sc = S[(pool, "valhard")]
        o = {N: 100 * sum(1 for i in ids if i in sc and any(c["ok"] for c in gen_order(sc[i])[:N])) / n
             for N in (1, 2, 4, 8, 16, 32, 48, 64, 999)}
        out["oracle_topN_k64"] = o
        print("  oracle by top-N (k64):", {k: round(x, 2) for k, x in o.items()})

    print("\n== class aggregation (pick class, then its best member)")
    for v in ("valhard", "r1w"):
        for pool in ("k32", "k64", "k64r"):
            for agg in ("lme", "lse", "mean"):
                rep(f"{pool}/{v} class-{agg}", (pool, v), lambda cs, i, agg=agg: pick_group(cs, agg))
            for beta in (0.25, 0.5, 1.0):
                rep(f"{pool}/{v} max+{beta}*log|class|", (pool, v), lambda cs, i, b=beta: pick_group(cs, "max", beta=b))

    print("\n== pruning before the validator")
    for v in ("valhard", "r1w"):
        for pool in ("k32", "k64", "k64r"):
            rep(f"{pool}/{v} drop-degenerate", (pool, v), lambda cs, i: pick_prune(cs, drop_degen=True))
            rep(f"{pool}/{v} drop-singletons", (pool, v), lambda cs, i: pick_prune(cs, drop_single=True, min_votes=2))
            rep(f"{pool}/{v} drop-classes<3", (pool, v), lambda cs, i: pick_prune(cs, drop_single=True, min_votes=3))
            if pool == "k64r":
                rep(f"{pool}/{v} drop-repair", (pool, v), lambda cs, i: pick_prune(cs, drop_repair=True))

    # ---- LODO-fitted rules ---------------------------------------------------------------------------------------
    def lodo(pool, family, grid, label):
        sc = S[pool]
        cache = {}
        for g in grid:
            cache[g] = {i: bool(i in sc and (p := family(sc[i], i, g)) and p.get("ok")) for i in ids}
        h, chosen = {}, {}
        for d in dbs:
            tr_ids = [i for i in ids if db[i] != d]
            gbest = max(grid, key=lambda g: (sum(cache[g][i] for i in tr_ids), -grid.index(g)))
            chosen[d] = gbest
            for i in ids:
                if db[i] == d:
                    h[i] = cache[gbest][i]
        rows[label] = {**cmpd(h), "chosen": {k: str(x) for k, x in chosen.items()},
                       "insample_best": max(((round(pct(cache[g]), 2), str(g)) for g in grid))}
        x = rows[label]
        print(f"  {label:58s} {x['pick']:6.2f}  d {x['delta']:+5.2f}  p {x['p']:.3f}  +{x['w']}/-{x['l']}  "
              f"in-sample best {x['insample_best']}  chosen {sorted(set(chosen.values()))}")

    print("\n== LODO-fitted (knob chosen on 7 DBs)")
    for v in ("valhard", "r1w"):
        for pool in ("k64", "k64r"):
            lodo((pool, v), lambda cs, i, N: pick_topn(cs, N), [4, 8, 12, 16, 24, 32, 40, 48, 64, 999],
                 f"{pool}/{v} LODO top-N")
            lodo((pool, v), lambda cs, i, d: pick_hyst(cs, K32SQL.get(i, set()), d),
                 [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 1e9], f"{pool}/{v} LODO hysteresis vs k32")
        for pool in ("k32", "k64", "k64r"):
            grid = [(b, gm) for b in (0, 0.25, 0.5, 0.75, 1.0, 1.5) for gm in (0, 0.05, 0.1, 0.2, 0.4)]
            lodo((pool, v), lambda cs, i, bg: pick_linear(cs, *bg), grid, f"{pool}/{v} LODO rr+b*log|cls|-g*cost")
            grid = [("lme", t) for t in (0.5, 1, 2, 4)] + [("lse", t) for t in (0.25, 0.5, 1, 2)] + [("max", 1)]
            lodo((pool, v), lambda cs, i, at: pick_group(cs, at[0], tau=at[1]), grid, f"{pool}/{v} LODO class agg")

    print("\n== generator-prior penalty rr - gamma*cost (repair cost capped unless 'raw'); fixed gammas, then LODO")
    G = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    for v in ("valhard", "r1w"):
        for pool in ("k32", "k64", "k64r"):
            for gm in (0.1, 0.2, 0.4):
                rep(f"{pool}/{v} cost-pen g={gm} cap", (pool, v), lambda cs, i, gm=gm: pick_cost(cs, gm))
            if pool == "k64r":
                rep(f"{pool}/{v} cost-pen g=0.2 raw", (pool, v), lambda cs, i: pick_cost(cs, 0.2, "raw"))
            lodo((pool, v), lambda cs, i, gm: pick_cost(cs, gm), G, f"{pool}/{v} LODO cost-pen (cap)")
            lodo((pool, v), lambda cs, i, gm: pick_cost(cs, gm, degen=True), G, f"{pool}/{v} LODO cost-pen+drop-degen")
            lodo((pool, v), lambda cs, i, gm: pick_rankpen(cs, gm), [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5],
                 f"{pool}/{v} LODO rr-g*log(1+genrank)")
    # in-sample curves (diagnostic only) and per-DB direction of the fixed g=0.2 rule
    curves = {}
    for v in ("valhard", "r1w"):
        for pool in ("k32", "k64", "k64r"):
            sc = S[(pool, v)]
            curves[f"{pool}/{v}"] = {gm: round(pct({i: bool(i in sc and (p := pick_cost(sc[i], gm)) and p["ok"])
                                                     for i in ids}), 2) for gm in G}
            print(f"  curve {pool}/{v}: {curves[f'{pool}/{v}']}")
    out["cost_curves"] = curves
    for v in ("valhard", "r1w"):
        for pool in ("k32", "k64", "k64r"):
            h = hits((pool, v), lambda cs, i: pick_cost(cs, 0.2))
            r0 = hits((pool, v), lambda cs, i: pick_max(cs))
            perdb = {d: sum(h[i] - r0[i] for i in ids if db[i] == d) for d in dbs}
            c = cmpd(h, r0)
            rows[f"{pool}/{v} g=0.2 vs own max"] = {**c, "per_db": perdb}
            print(f"  {pool}/{v} g=0.2 vs same pool max: d {c['delta']:+.2f} p {c['p']:.3f} +{c['w']}/-{c['l']} per-DB {perdb}")

    print("\n== top-2 tie-break by generator prior (LODO margin) and top-m class oracle")
    for v in ("valhard", "r1w"):
        for pool in ("k32", "k64", "k64r"):
            lodo((pool, v), lambda cs, i, m: pick_tiebreak(cs, m), [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                 f"{pool}/{v} LODO top-2 tie-break by gen cost")
            sc = S[(pool, v)]
            tm = {m: round(100 * sum(1 for i in ids if i in sc and any(x[1] for x in class_table(sc[i])[:m])) / n, 2)
                  for m in (1, 2, 3, 5)}
            out[f"top_class_oracle_{pool}_{v}"] = tm
            print(f"  {pool}/{v} right class within validator top-m classes: {tm}")

    out["rows"] = rows

    # ---- diagnostics: why extra recall does not become picks ------------------------------------------------------
    print("\n== distractor diagnostics (VALHARD)")
    k32, k64, k64r = S[("k32", "valhard")], S[("k64", "valhard")], S[("k64r", "valhard")]
    diag = {}
    for name, P in (("k64", k64), ("k64r", k64r)):
        gain_q = [i for i in ids if i in P and any(c["ok"] for c in P[i])
                  and not (i in k32 and any(c["ok"] for c in k32[i]))]
        # rank of best right class among live classes (by max rr)
        ranks, gaps = [], []
        for i in gain_q:
            L = live(P[i])
            cls = sorted(((max(c["rr"] for c in m), any(c["ok"] for c in m), len(m)) for m in groups(L).values()),
                         reverse=True)
            r = next((k for k, x in enumerate(cls) if x[1]), None)
            ranks.append(r)
            if r is not None and r > 0:
                gaps.append(cls[0][0] - cls[r][0])
        # changed picks
        flips = collections.Counter()
        for i in ids:
            if i not in P:
                continue
            p_old, p_new = pick_max(k32.get(i, [])) if i in k32 else None, pick_max(P[i])
            if p_old is None or p_new is None or p_old["sql"] == p_new["sql"]:
                continue
            src = "repair" if p_new.get("src") == "repair" else ("old" if p_new["sql"] in K32SQL.get(i, ()) else "new")
            flips[(src, bool(p_old["ok"]), bool(p_new["ok"]))] += 1
        rc = collections.Counter(ranks)
        diag[name] = {"gain_q": len(gain_q), "right_class_rank": {str(k): v for k, v in sorted(rc.items(), key=str)},
                      "median_gap_to_top": sorted(gaps)[len(gaps) // 2] if gaps else None,
                      "flips(src,old_ok,new_ok)": {str(k): v for k, v in flips.items()}}
        print(f"  {name}: questions whose only right answer is new: {len(gain_q)}; rank of right class there "
              f"{dict(sorted(rc.items(), key=str))}; median rr gap to top {diag[name]['median_gap_to_top']}")
        print(f"    pick flips vs k32 (src of new pick, old ok, new ok): {dict(flips)}")
    # where do the new wrong picks come from: rr distribution of new vs old candidates, and generator rank of new picks
    newpick_rank = collections.Counter()
    for i in ids:
        if i not in k64:
            continue
        p = pick_max(k64[i])
        if p and p["sql"] not in K32SQL.get(i, ()):
            b = p.get("rank", -1)
            newpick_rank[("<32" if b < 32 else "32-47" if b < 48 else "48+", bool(p["ok"]))] += 1
    diag["k64_new_pick_gen_rank"] = {str(k): v for k, v in newpick_rank.items()}
    print("  k64 picks that are not in k32, by generator rank and ok:", dict(newpick_rank))
    # how often a k64 pick is a NEW sql but the SAME class as the k32 pick (harmless churn)
    same_cls = sum(1 for i in ids if i in k64 and i in k32 and (pa := pick_max(k64[i])) and (pb := pick_max(k32[i]))
                   and pa["sql"] != pb["sql"] and pa["sig"] == pb["sig"])
    diag["k64_pick_new_sql_same_class"] = same_cls
    print("  k64 pick is a different SQL but the same result class as k32 pick:", same_cls)
    # rescoring drift on candidates shared by k32 and k64 files
    import statistics
    dr = []
    for i in ids:
        if i in k32 and i in k64:
            m = {c["sql"]: c["rr"] for c in k32[i]}
            dr += [c["rr"] - m[c["sql"]] for c in k64[i] if c["sql"] in m and c["rr"] > -1e8 and m[c["sql"]] > -1e8]
    diag["rr_drift_shared"] = {"n": len(dr), "mean": statistics.mean(dr), "sd": statistics.pstdev(dr),
                               "abs>0.5": sum(abs(x) > 0.5 for x in dr)}
    print("  rr drift on shared candidates k64 - k32:", diag["rr_drift_shared"])
    # repair picks
    rp = collections.Counter()
    for i in ids:
        if i in k64r:
            p = pick_max(k64r[i])
            if p and p.get("src") == "repair":
                rp[(p.get("repair", "?").split(":")[0], bool(p["ok"]))] += 1
    diag["repair_picks"] = {str(k): v for k, v in rp.items()}
    print("  repair picks (variant, ok):", dict(rp))
    out["diag"] = diag
    a.json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.json, "w"), indent=1)
    print("wrote", a.json)


if __name__ == "__main__":
    main()
