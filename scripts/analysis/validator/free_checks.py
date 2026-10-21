"""Free ($0, Mac) pick-rule checks on DECISION with existing validator score files (diagnostic only, no decision).

1. group scoring: score a result class by its members (logsumexp / mean / max + lam*log count) instead of its best one
2. validator averaging: per-question z-scored rr averaged over VALHARD + VAL3 runs (proxy for a weight soup)
3. fallback: take the generator's top candidate (lowest rank) when the validator's top-2 class margin < tau

Parameters (lam, tau) are fitted leave-one-DB-out. Pools: R1 k=32 (VALHARD 59.05) and k=64+repair (VALHARD 59.55).
  .venv/bin/python scripts/analysis/validator/free_checks.py --json runs/analysis/validator/free_checks.json
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("val3_eval", ROOT / "scripts/analysis/val3_eval.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

RUNS = ("A_s1", "A_s2", "B_s1")
LAMS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
TAUS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def live_sql(cands):
    return V.live([c for c in cands if c.get("sql")])


def class_pick(cands, agg: str, lam: float = 0.0):
    """Pick the best class under agg, then its best-rr member (ok is per class: same sig = same result)."""
    groups = collections.defaultdict(list)
    for c in live_sql(cands):
        groups[c.get("sig") or c["sql"]].append(c)
    if not groups:
        return None

    def score(m):
        r = [c.get("rr", -1e9) for c in m]
        if agg == "max":
            return max(r)
        if agg == "lse":
            mx = max(r)
            return mx + math.log(sum(math.exp(x - mx) for x in r))
        if agg == "mean":
            return sum(r) / len(r)
        if agg == "maxcount":
            return max(r) + lam * math.log(len(m))
        raise ValueError(agg)
    best = max(groups.values(), key=score)
    return max(best, key=lambda c: c.get("rr", -1e9))


def fallback_pick(cands, tau: float):
    groups = {}
    for c in live_sql(cands):
        s = c.get("sig") or c["sql"]
        if s not in groups or c.get("rr", -1e9) > groups[s].get("rr", -1e9):
            groups[s] = c
    if not groups:
        return None
    top = sorted(groups.values(), key=lambda c: -c.get("rr", -1e9))
    if len(top) > 1 and top[0]["rr"] - top[1]["rr"] < tau:
        return min(live_sql(cands), key=lambda c: c.get("rank", 1e9))
    return top[0]


def zscore(cands):
    r = [c.get("rr", 0.0) for c in cands]
    m = sum(r) / len(r)
    sd = (sum((x - m) ** 2 for x in r) / len(r)) ** 0.5 or 1.0
    return [(x - m) / sd for x in r]


def averaged(files: list[dict]) -> dict:
    """Average per-question z-scored rr across score files, aligned by (sql, occurrence)."""
    base = files[0]
    out = {}
    for i, r in base.items():
        if not all(i in f for f in files):
            continue
        zs = []
        for f in files:
            occ = collections.Counter()
            m = {}
            for c, z in zip(f[i]["cands"], zscore(f[i]["cands"])):
                k = (c.get("sql") or "", occ[c.get("sql") or ""])
                occ[c.get("sql") or ""] += 1
                m[k] = z
            zs.append(m)
        occ = collections.Counter()
        cands = []
        for c in r["cands"]:
            k = (c.get("sql") or "", occ[c.get("sql") or ""])
            occ[c.get("sql") or ""] += 1
            cands.append({**c, "rr": sum(z[k] for z in zs) / len(zs)})
        out[i] = {**r, "cands": cands}
    return out


def hits_with(S, ids, fn):
    return {i: bool(i in S and (p := fn(S[i]["cands"])) and p.get("ok")) for i in ids}


def lodo(S, ids, db, grid, make_fn):
    """Leave-one-DB-out: for each DB pick the grid value best on the other DBs, apply it to the held-out DB."""
    per = {g: hits_with(S, ids, make_fn(g)) for g in grid}
    out, chosen = {}, {}
    for d in sorted(set(db.values())):
        tr = [i for i in ids if db[i] != d]
        g = max(grid, key=lambda g: (sum(per[g][i] for i in tr), -abs(g)))
        chosen[d] = g
        out.update({i: per[g][i] for i in ids if db[i] == d})
    return out, chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    dec = json.load(open(V.DECISION))
    ids = list(dec["ids"])
    idset = set(ids)
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    db = {i: tr[int(i.rsplit("_", 1)[1])]["db_id"] for i in ids}
    n = len(ids)
    res = {}
    for p in ("k32", "k64r"):
        lab = V.load(V.LABELS[p], idset)
        base = V.load(V.BASE[p], idset)
        runs = {r: V.join_labels(V.load(V.RUNS_DIR / r / "val3" / f"{p}_by_{r}.jsonl", idset), lab) for r in RUNS}
        HB = V.hits(base, ids)
        acc0 = 100 * sum(HB.values()) / n
        print(f"\n=== pool {p}: VALHARD {acc0:.2f} (n={n}) ===")
        rows = {}

        def report(name, H, extra=""):
            c = V.compare(HB, [H], ids, db)
            rows[name] = {k: c[k] for k in ("acc", "delta", "wins", "losses", "q_p_one_sided", "db_p")}
            print(f"  {name:38s} {c['acc'][0]:6.2f}  {c['delta']:+.2f}  +{c['wins']}/-{c['losses']}  "
                  f"q p {c['q_p_one_sided']:.3f}  DB p {c['db_p']:.3f} {extra}")

        # 1. group scoring (VALHARD scores)
        report("1 class max (control = VALHARD)", hits_with(base, ids, lambda cs: class_pick(cs, "max")))
        report("1 class logsumexp", hits_with(base, ids, lambda cs: class_pick(cs, "lse")))
        report("1 class mean", hits_with(base, ids, lambda cs: class_pick(cs, "mean")))
        H, ch = lodo(base, ids, db, LAMS, lambda lam: (lambda cs: class_pick(cs, "maxcount", lam)))
        report("1 class max + lam*log(count) LODO", H, f"lam {sorted(collections.Counter(ch.values()).items())}")
        # 2. averaging
        combos = {"VALHARD+A_s1+A_s2": ["A_s1", "A_s2"], "VALHARD+A_s1": ["A_s1"], "A_s1+A_s2": None,
                  "VALHARD+A_s1+A_s2+B_s1": ["A_s1", "A_s2", "B_s1"]}
        for name, extra in combos.items():
            files = [runs["A_s1"], runs["A_s2"]] if extra is None else [base] + [runs[r] for r in extra]
            report(f"2 avg {name}", V.hits(averaged(files), ids))
        avg3 = averaged([base, runs["A_s1"], runs["A_s2"]])
        report("2+1 avg3 + class logsumexp", hits_with(avg3, ids, lambda cs: class_pick(cs, "lse")))
        # 3. fallback to generator top
        H, ch = lodo(base, ids, db, TAUS, lambda t: (lambda cs: fallback_pick(cs, t)))
        report("3 fallback to gen top if margin<tau LODO", H, f"tau {sorted(collections.Counter(ch.values()).items())}")
        report("3 generator top alone (info)", hits_with(base, ids, lambda cs: fallback_pick(cs, 1e9)))
        res[p] = rows
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
