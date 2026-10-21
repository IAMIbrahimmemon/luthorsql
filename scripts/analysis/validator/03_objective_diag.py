"""Validator objective / optimisation diagnostics (research note 03_objective; Mac only, nothing fitted).

  .venv/bin/python scripts/analysis/validator/03_objective_diag.py > runs/analysis/validator/03_objective_diag.out

Reads only: training lists (stats), val_scored.jsonl of finished validator runs (4 pinned train DBs), training logs
(loss lines, GATE lines filtered out), and DECISION rows of held-out score files (RESERVED dropped at load).
Sections:
 A. train lists: groups per list, positives per list -> label-smoothing loss floor
 B. train-loss trajectory per run (epoch means vs floor)
 C. val diagnostics at the kept checkpoint: pick acc + binomial SE, group NLL, MRR, informative lists, margins
 D. DECISION (R1 k=32 pool): base / VALHARD / VAL2 -- score scale, rank of the best right class on misses, margin
    of misses (score units and per-list z units), top-2 near-ties; fixed 50/50 z-average of two validators (soup proxy,
    info only; no weight is fitted).
"""
from __future__ import annotations

import collections
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from luthor.stats import cluster_signflip  # noqa: E402

DEC = ROOT / "data/sys1/heldout_v2/decision.json"


def live(cands):
    L = [c for c in cands if c.get("runs") and c.get("nonnull") and c.get("sql")]
    return L or [c for c in cands if c.get("sql")]


def groups(cands, key="rr"):
    """[(score, ok)] per result class (max score over members), sorted desc."""
    g = {}
    for c in live(cands):
        s = c.get("sig") or c["sql"]
        v = c.get(key, -1e9)
        if s not in g or v > g[s][0]:
            g[s] = (v, bool(c.get("ok")))
    return sorted(g.values(), key=lambda x: -x[0])


# ---------------------------------------------------------------------------------------------- A
def list_stats(path):
    ng, npos, nent = [], [], []
    for l in open(path):
        r = json.loads(l)
        it = [c for c in {c["sql"]: c for c in r["cands"]}.values() if c.get("runs") or c.get("mutant")]
        sig = collections.defaultdict(list)
        for k, c in enumerate(it):
            sig[c.get("sig") or f"_r{k}"].append(bool(c.get("ok")))
        ng.append(len(sig))
        npos.append(sum(any(v) for v in sig.values()))
        nent.append(len(it))
    return ng, npos, nent


def smooth_floor(n, p, eps=0.05):
    """min over scores of (1-eps)(lse - lse_pos) + eps(lse - mean): target q = (1-eps)*uniform-on-pos + eps/n."""
    q = [(1 - eps) / p + eps / n] * p + [eps / n] * (n - p)
    return -sum(x * math.log(x) for x in q) - (1 - eps) * math.log(1) + 0  # CE(q, q) = H(q); pos mass split is free


# ---------------------------------------------------------------------------------------------- B
def losses(log):
    out = collections.defaultdict(list)
    for l in open(log, errors="ignore"):
        if "GATE" in l:
            continue
        m = re.match(r"rr step (\d+)/(\d+) ep (\d+) loss ([0-9.]+)", l)
        if m:
            out[int(m.group(3))].append(float(m.group(4)))
    return out


def main():
    print("A. TRAIN LISTS (groups = result classes; floor = min of the group loss with label smoothing 0.05)")
    for name, p in (("valhard_r1", "data/valhard/r1/train_lists.jsonl"), ("valhard_v4desc", "data/valhard/v4desc/train_lists.jsonl"),
                    ("val2", "data/val2/train_lists.jsonl")):
        p = ROOT / p
        if not p.exists():
            continue
        ng, npos, nent = list_stats(p)
        fl = st.mean(smooth_floor(n, max(k, 1)) for n, k in zip(ng, npos) if n > k)
        print(f"  {name}: lists {len(ng)}, entries/list {st.mean(nent):.2f}, groups/list {st.mean(ng):.2f} "
              f"(p10 {sorted(ng)[len(ng)//10]}, p90 {sorted(ng)[9*len(ng)//10]}), right groups/list {st.mean(npos):.2f}, "
              f"smoothing floor {fl:.3f}, ln(groups) chance loss {st.mean(math.log(n) for n in ng):.3f}")

    print("\nB. TRAIN LOSS (log 'rr step' lines every 25 steps; GATE lines dropped)")
    for name, p in (("VALHARD r1", "runs/phase1/valhard/valhard/valhard_train.log"),
                    ("VALHARD v4desc", "runs/phase1/valhard_v4desc/valhard/valhard_train.log"),
                    ("VAL2", "runs/phase1/val2/val2/val2_train.log"),
                    ("R1 wide (legacy)", "runs/phase1/p1w/logs/rrtrain_r1.log")):
        L = losses(ROOT / p)
        print(f"  {name}: " + "; ".join(
            f"ep{e} mean {st.mean(v):.3f} sd {st.pstdev(v):.3f} (n={len(v)}, se {st.pstdev(v)/math.sqrt(len(v)):.3f})"
            for e, v in sorted(L.items())))

    print("\nC. VAL LISTS at the kept checkpoint (4 pinned train DBs)")
    for name, p in (("VALHARD r1 (ep2)", "runs/phase1/valhard/valhard/rr/val_scored.jsonl"),
                    ("VALHARD v4desc (ep1)", "runs/phase1/valhard_v4desc/valhard/rr/val_scored.jsonl"),
                    ("VAL2 (ep2)", "runs/phase1/val2/val2/rr/val_scored.jsonl")):
        R = [json.loads(l) for l in open(ROOT / p)]
        acc, nll, rr_, inf, greedy_ok, marg_hit, marg_miss, ng, sm = [], [], [], 0, [], [], [], [], []
        for r in R:
            G = groups(r["cands"])
            if not G:
                continue
            ng.append(len(G))
            s = [x[0] for x in G]
            lse = math.log(sum(math.exp(v - s[0]) for v in s)) + s[0]
            pos = [v for v, ok in G if ok]
            if not pos or len(pos) == len(G):
                continue
            lp = math.log(sum(math.exp(v - s[0]) for v in pos)) + s[0]
            nll.append(lse - lp)
            sm.append(0.95 * (lse - lp) + 0.05 * (lse - sum(x for x in s) / len(s)))
            acc.append(G[0][1])
            first = next(k for k, (_, ok) in enumerate(G) if ok)
            rr_.append(1 / (first + 1))
            g0 = min(r["cands"], key=lambda c: c.get("rank", 99))
            greedy_ok.append(bool(g0.get("ok")))
            bw = max(v for v, ok in G if not ok)
            (marg_hit if G[0][1] else marg_miss).append((max(pos) - bw) if G[0][1] else (bw - max(pos)))
        n = len(acc)
        a = sum(acc) / n
        informative = sum(1 for g in greedy_ok if not g)
        print(f"  {name}: n {n}, groups/list {st.mean(ng):.1f}, pick {a:.4f} (binomial se {math.sqrt(a*(1-a)/n):.4f}, "
              f"1 list = {1/n:.4f}), greedy-wrong lists {informative}, group NLL {st.mean(nll):.3f} (smoothed 0.05, comparable to the train log: {st.mean(sm):.3f}), MRR {st.mean(rr_):.3f}, "
              f"misses {n-sum(acc)} with median margin {st.median(marg_miss) if marg_miss else float('nan'):.2f}; "
              f"hits with margin < 1: {sum(m < 1 for m in marg_hit)}")

    print("\nD. DECISION, R1 k=32 pool (src r1/both), pick = max over live")
    dec = json.load(open(DEC))
    ids = list(dec["ids"])
    idset = set(ids)
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    dbof = {i: tr[int(i.rsplit("_", 1)[1])]["db_id"] for i in ids}
    files = {"base": "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl",
             "valhard": "runs/phase1/valhard/valhard/heldout_by_new.jsonl",
             "val2": "runs/phase1/val2/val2/heldout_by_new.jsonl"}
    S = {}
    for k, f in files.items():
        d = {}
        for l in open(ROOT / f):
            r = json.loads(l)
            if r["id"] not in idset:
                continue
            r["cands"] = [c for c in r["cands"] if c.get("src", "r1") in ("r1", "both")]
            d[r["id"]] = {c["sql"]: c for c in r["cands"] if c.get("sql")}
        S[k] = d
    common = [i for i in ids if all(i in S[k] for k in S)]
    print(f"  questions covered by all three: {len(common)} of {len(ids)}")

    def zs(i, k):
        v = [c.get("rr", 0.0) for c in S[k][i].values()]
        m, sd = st.mean(v), (st.pstdev(v) or 1.0)
        return {q: (c.get("rr", 0.0) - m) / sd for q, c in S[k][i].items()}

    hits = {}
    for k in S:
        h, scale, rank_hist, mz, ms, near = {}, [], collections.Counter(), [], [], 0
        for i in ids:
            if i not in S[k]:
                h[i] = False
                continue
            cands = list(S[k][i].values())
            G = groups(cands)
            z = zs(i, k)
            h[i] = bool(G and G[0][1])
            v = [c.get("rr", 0.0) for c in live(cands)]
            if len(v) > 1:
                scale.append(st.pstdev(v))
            if G and not G[0][1] and any(ok for _, ok in G):
                first = next(j for j, (_, ok) in enumerate(G) if ok)
                rank_hist[min(first + 1, 6)] += 1
                ms.append(G[0][0] - G[first][0])
                mz.append(ms[-1] / (st.pstdev(v) or 1.0))
            if len(G) >= 2 and G[0][1] != G[1][1] and (G[0][0] - G[1][0]) / (st.pstdev(v) or 1) < 0.25:
                near += 1
        hits[k] = h
        miss_or = len(ms)
        print(f"  {k}: pick {100*sum(h.values())/len(ids):.2f}; list score sd median {st.median(scale):.2f}; "
              f"oracle-reachable misses {miss_or}: rank of best right class {dict(sorted(rank_hist.items()))} "
              f"(6 = >=6); miss margin median {st.median(ms):.2f} score / {st.median(mz):.2f} sd, "
              f"<0.25 sd {sum(x < 0.25 for x in mz)}, <0.5 sd {sum(x < 0.5 for x in mz)}; "
              f"one-right top-2 near-ties (<0.25 sd) {near}")

    for a_, b_ in (("base", "valhard"), ("valhard", "val2"), ("base", "val2")):
        hz = {}
        for i in ids:
            if i not in common:
                hz[i] = False
                continue
            za, zb = zs(i, a_), zs(i, b_)
            for q, c in S[a_][i].items():
                c["_z"] = 0.5 * za[q] + 0.5 * zb.get(q, 0.0)
            G = groups(list(S[a_][i].values()), key="_z")
            hz[i] = bool(G and G[0][1])
        c = cluster_signflip(hits[a_], hz, dbof)
        c2 = cluster_signflip(hits["valhard"], hz, dbof)
        both = sum(hits[a_][i] and hits[b_][i] for i in ids)
        print(f"  z-average {a_}+{b_} (fixed 0.5/0.5, soup proxy): {100*sum(hz.values())/len(ids):.2f} "
              f"(vs {a_} {c['delta']:+.2f}, vs valhard {c2['delta']:+.2f}, DB p {c2['p']:.2f}); both-right {both}, "
              f"{a_}-only {sum(hits[a_][i] and not hits[b_][i] for i in ids)}, {b_}-only {sum(hits[b_][i] and not hits[a_][i] for i in ids)}")


if __name__ == "__main__":
    main()
