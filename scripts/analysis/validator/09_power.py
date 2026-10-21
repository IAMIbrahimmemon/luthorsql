"""09 eval methodology: observed validator pairs on DECISION, between-DB heterogeneity, and power / MDE by simulation.

DECISION rows only (data/sys1/heldout_v2/decision.json ids). No RESERVED, no BIRD dev. Diagnostics only, no tuning.
Writes runs/analysis/validator/09_power.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402

DEC = json.load(open(ROOT / "data/sys1/heldout_v2/decision.json"))
IDS = set(DEC["ids"])


def load(path, srcs=None):
    out = {}
    for line in open(ROOT / path):
        r = json.loads(line)
        if r["id"] not in IDS:
            continue
        cs = r["cands"]
        if srcs is not None and any("src" in c for c in cs):
            cs = [c for c in cs if c.get("src") in srcs]
        r["cands"] = cs
        out[r["id"]] = r
    return out


def live(cs):
    L = [c for c in cs if c.get("runs") and c.get("nonnull")]
    return L or cs


def hits(rows):
    h, db = {}, {}
    for i in IDS:
        r = rows.get(i)
        db[i] = DEC_DB[i]
        if r is None:
            h[i] = False
            continue
        u = [c for c in r["cands"] if c.get("sql")]
        h[i] = bool(u) and bool(max(live(u), key=lambda c: c.get("rr", -1e9)).get("ok"))
    return h, db


_TR = json.load(open(ROOT / "data/raw/bird_train/train.json"))
DEC_DB = {i: _TR[int(i.rsplit("_", 1)[1])]["db_id"] for i in IDS}
del _TR

R1SRC = ("r1", "both")
PAIRS = {
    # name: (base path, base srcs, new path, new srcs, kind)
    "VALHARD_vs_R1val (R1 pool)": ("runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl", R1SRC,
                                    "runs/phase1/valhard/valhard/heldout_by_new.jsonl", R1SRC, "treatment"),
    "VALHARD_v4desc_vs_own (v4desc pool)": ("runs/phase1/valhard_v4desc/valhard/heldout_by_base.jsonl", None,
                                            "runs/phase1/valhard_v4desc/valhard/heldout_by_new.jsonl", None, "treatment"),
    "R2val_vs_R1val (R1 pool)": ("runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl", R1SRC,
                                 "runs/phase1/p1w/p1w/heldout_union_by_r2.jsonl", R1SRC, "null-proxy"),
    "R2val_vs_R1val (R1uR2 pool)": ("runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl", None,
                                    "runs/phase1/p1w/p1w/heldout_union_by_r2.jsonl", None, "null-proxy"),
    "VAL2_vs_R1val (R1uR2 pool)": ("runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl", None,
                                   "runs/phase1/val2/val2/heldout_by_new.jsonl", None, "treatment"),
}


def observed():
    res = {}
    for name, (bp, bs, np_, ns, kind) in PAIRS.items():
        hb, db = hits(load(bp, bs))
        hn, _ = hits(load(np_, ns))
        c = cluster_signflip(hb, hn, db)
        q = question_signflip(hb, hn)
        per = {}
        for g in sorted(set(db.values())):
            ids = [i for i in IDS if db[i] == g]
            w = sum(hn[i] and not hb[i] for i in ids)
            l_ = sum(hb[i] and not hn[i] for i in ids)
            per[g] = {"n": len(ids), "w": w, "l": l_, "base": sum(hb[i] for i in ids)}
        res[name] = {"kind": kind, "base": 100 * np.mean([hb[i] for i in IDS]), "new": 100 * np.mean([hn[i] for i in IDS]),
                     "delta": c["delta"], "db_p": c["p"], "db_ci": c["ci"], "mcnemar_p": q["p"], "wins": q["wins"],
                     "losses": q["losses"], "per_db": per}
    return res


def tau_mom(per):
    """Method-of-moments between-DB SD of the true per-question effect (proportion units)."""
    n = np.array([v["n"] for v in per.values()], float)
    d = np.array([(v["w"] - v["l"]) / v["n"] for v in per.values()])
    m = np.array([(v["w"] + v["l"]) / v["n"] for v in per.values()])
    w = n / n.sum()
    mu = (w * d).sum()
    samp = (m - d ** 2) / n  # sampling variance of each per-DB delta
    q = (w * (d - mu) ** 2).sum() - (w * samp).sum()
    return math.sqrt(max(q, 0.0)), mu


# ---------------- power simulation ----------------
def signs(G):
    return np.array([[1 if (k >> j) & 1 else -1 for j in range(G)] for k in range(2 ** G)], float)


def sim(delta, sizes, pi, tau, R, rng, rules, Smat=None, seed_sd=0.0):
    """delta, tau in proportion; pi = discordance rate. returns dict rule -> power."""
    G = len(sizes)
    n = np.array(sizes, float)
    out = {k: 0 for k in rules}
    for _ in range(R):
        d0 = delta + (rng.normal(0, seed_sd) if seed_sd > 0 else 0.0)  # this checkpoint's effect = recipe + seed luck
        dg = rng.normal(d0, tau, G) if tau > 0 else np.full(G, d0)
        dg = np.clip(dg, -pi * 0.98, pi * 0.98)
        pw = pi * (0.5 + dg / (2 * pi))
        pl = pi - pw
        w = rng.binomial(n.astype(int), pw)
        l_ = rng.binomial((n - w).astype(int), pl / np.maximum(1 - pw, 1e-9))
        D = (w - l_).astype(float)
        est = D.sum() / n.sum()
        # DB sign-flip (sum statistic, as luthor.stats)
        if Smat is not None:
            p_db = float(np.mean(np.abs(Smat @ D) >= abs(D.sum()) - 1e-9))
        else:  # large G: Monte Carlo sign flips
            S = rng.choice([-1.0, 1.0], size=(2000, G))
            p_db = float((1 + np.sum(np.abs(S @ D) >= abs(D.sum()) - 1e-9)) / 2001)
        # McNemar (exact, two-sided)
        W, L = int(w.sum()), int(l_.sum())
        m = W + L
        k = min(W, L)
        p_mc = min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0
        z = (W - L) / math.sqrt(m) if m else 0.0
        for name, f in rules.items():
            out[name] += f(est, p_db, p_mc, z)
    return {k: v / R for k, v in out.items()}


RULES = {
    "current: d>=1.5 & DBp<0.10": lambda e, pd, pm, z: e >= 0.015 and pd < 0.10,
    "DBp<0.10 one-sided-ish (d>0)": lambda e, pd, pm, z: e > 0 and pd < 0.10,
    "McNemar p<0.05 (d>0)": lambda e, pd, pm, z: e > 0 and pm < 0.05,
    "McNemar one-sided 0.05 (z>1.645)": lambda e, pd, pm, z: z > 1.645,
    "McNemar 1-sided 0.05 & DBp<0.25": lambda e, pd, pm, z: z > 1.645 and pd < 0.25,
}


def mde(delta_grid, power, target=0.8):
    for d, p in zip(delta_grid, power):
        if p >= target:
            return d
    return None


def main():
    rng = np.random.default_rng(9)
    obs = observed()
    for k, v in obs.items():
        v["tau"], _ = tau_mom(v["per_db"])
    sizes8 = list(DEC["database_counts"].values())
    pi = np.mean([(v["wins"] + v["losses"]) / len(IDS) for k, v in obs.items() if k.startswith("VALHARD")])
    taus = [v["tau"] for v in obs.values()]
    designs = {
        "DECISION 8 DBs (796 q)": sizes8,
        "DECISION+RESERVED-size 8 DBs (~1140 q, info only)": [round(s / 0.7) for s in sizes8],
        "16 DBs x 100 q": [100] * 16,
        "24 DBs x 100 q": [100] * 24,
        "48 DBs x 80 q (5-fold cross-fit on BIRD-train)": [80] * 48,
        "48 DBs x 160 q (cross-fit, more q/DB)": [160] * 48,
        "BIRD dev 11 DBs (1534 q)": [1534 // 11] * 11,
    }
    grid = [x / 400 for x in range(0, 25)]  # 0..6 points step 0.25
    Sm = {G: signs(G) for G in (8, 11, 16)}
    table = {}
    for tau in (0.0, 0.01, 0.02):
        for dn, sz in designs.items():
            G = len(sz)
            pw = {r: [] for r in RULES}
            for d in grid:
                s = sim(d, sz, pi, tau, 600 if G <= 16 else 300, rng, RULES, Sm.get(G))
                for r in RULES:
                    pw[r].append(s[r])
            table[f"tau={tau*100:.0f}pt | {dn}"] = {
                r: {"mde80_pts": (None if mde(grid, pw[r]) is None else 100 * mde(grid, pw[r])),
                    "power_at_1.0": pw[r][4], "power_at_1.5": pw[r][6],
                    "power_at_2.0": pw[r][8], "false_pos_at_0": pw[r][0]} for r in RULES}
            print(dn, tau, {r: table[f"tau={tau*100:.0f}pt | {dn}"][r]["mde80_pts"] for r in RULES}, flush=True)
    seedtab = {}
    for sd in (0.005, 0.0075, 0.01):
        for dn in ("DECISION 8 DBs (796 q)", "48 DBs x 80 q (5-fold cross-fit on BIRD-train)"):
            sz = designs[dn]
            G = len(sz)
            pw = {r: [] for r in RULES}
            for d in grid[:13]:
                s_ = sim(d, sz, pi, 0.01, 400 if G <= 16 else 250, rng, RULES, Sm.get(G), seed_sd=sd)
                for r in RULES:
                    pw[r].append(s_[r])
            seedtab[f"seed_sd={sd*100:.2f}pt tau=1pt | {dn}"] = {
                r: {"mde80_pts": (None if mde(grid, pw[r]) is None else 100 * mde(grid, pw[r])),
                    "false_pos_recipe_null": pw[r][0], "power_at_1.5": pw[r][6], "power_at_2.0": pw[r][8]} for r in RULES}
            print("SEED", sd, dn, {r: (seedtab[f"seed_sd={sd*100:.2f}pt tau=1pt | {dn}"][r]["mde80_pts"],
                                       round(pw[r][0], 3)) for r in RULES}, flush=True)
    # minimum attainable p for DB sign-flip with 8 DBs, and what it takes
    S8 = Sm[8]
    D = np.array([1.0] * 8)
    minp = float(np.mean(np.abs(S8 @ D) >= 8 - 1e-9))
    # with the observed VALHARD per-DB D, how large would the smallest DB effects have to be
    out = {"pi_discordance": pi, "observed": obs, "taus_pts": [100 * t for t in taus], "min_db_p_8": minp,
           "power": table, "power_with_seed_noise": seedtab}
    Path(ROOT / "runs/analysis/validator").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "runs/analysis/validator/09_power.json", "w"), indent=1, default=float)
    for k, v in obs.items():
        print(k, {x: v[x] for x in ("base", "new", "delta", "db_p", "mcnemar_p", "wins", "losses")},
              "tau_pts=%.2f" % (100 * v["tau"]))
        print("   per-DB (w,l,n):", {g: (p["w"], p["l"], p["n"]) for g, p in v["per_db"].items()})


if __name__ == "__main__":
    main()
