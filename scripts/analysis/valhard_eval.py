"""VALHARD (runs/phase1/valhard/PREREG.md): the hard-negative validator vs the generator's own validator, System 1 only
(ONE generator's k=32 candidates, ONE validator picks), on DECISION (796 q, 8 DBs, original gold).

  .venv/bin/python scripts/analysis/valhard_eval.py                       # r1: runs/phase1/valhard/valhard/heldout_by_new.jsonl
  .venv/bin/python scripts/analysis/valhard_eval.py --gen v4desc          # runs/phase1/valhard_v4desc/valhard/...
  .venv/bin/python scripts/analysis/valhard_eval.py --new F --base F      # explicit files (e.g. a dry run: --new = --base)

Baseline: r1 -> the R1 wide validator's scores (runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl, the pick-gap note's
57.29); v4desc -> its own validator scored on the same pool by the VALHARD box (heldout_by_base.jsonl).
Candidates: src in {r1, both} when the file has src (the R1 share of the R1 u R2 pool), else all. Rows whose id is not
in DECISION are dropped at load (RESERVED stays blind). Pick = max rr over live (runs and nonnull; all if none live);
uncovered DECISION questions count wrong. Both arms must hold the same candidate SQL per question (checked).
PRIMARY: delta = new - base, exact DB sign-flip p (8 DBs). PASS: delta >= +1.5 and p < 0.10; KILL: delta < +0.5;
otherwise NO PASS. Secondary (info): McNemar; per-bucket miss counts (pick-gap taxonomy: the 84 hand-labelled R1
misses by theme/verdict, and an automatic clause-diff bucket of every miss of each arm); top-2 pairwise accuracy.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/valhard"))
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402
import sqldiff  # noqa: E402

DECISION = ROOT / "data/sys1/heldout_v2/decision.json"
MANUAL = ROOT / "scripts/analysis/system1_pick_gap_manual.json"
DEFAULTS = {"r1": (ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl",
                   ROOT / "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl"),
            "v4desc": (ROOT / "runs/phase1/valhard_v4desc/valhard/heldout_by_new.jsonl",
                       ROOT / "runs/phase1/valhard_v4desc/valhard/heldout_by_base.jsonl")}
PASS_DELTA, PASS_P, KILL_DELTA = 1.5, 0.10, 0.5


def load(path: Path, ids: set) -> dict:
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["id"] not in ids:
            continue  # never look at non-DECISION rows
        if any("src" in c for c in r["cands"]):
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


def class_scores(cands):
    """{sig: (best rr, ok, best member)} over live candidates."""
    out = {}
    for c in live([c for c in cands if c.get("sql")]):
        s = c.get("sig") or c["sql"]
        if s not in out or c.get("rr", -1e9) > out[s][0]:
            out[s] = (c.get("rr", -1e9), bool(c.get("ok")), c)
    return out


def verdict(delta, p):
    if delta >= PASS_DELTA and p < PASS_P:
        return "PASS"
    return "KILL" if delta < KILL_DELTA else "NO PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", choices=sorted(DEFAULTS), default="r1")
    ap.add_argument("--new", type=Path)
    ap.add_argument("--base", type=Path)
    ap.add_argument("--json", type=Path, help="also write the numbers here")
    a = ap.parse_args()
    new_p, base_p = a.new or DEFAULTS[a.gen][0], a.base or DEFAULTS[a.gen][1]
    dec = json.load(open(DECISION))
    ids = list(dec["ids"])
    idset = set(ids)
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    db = {i: tr[int(i.rsplit("_", 1)[1])]["db_id"] for i in ids}
    assert collections.Counter(db.values()) == collections.Counter(dec["database_counts"])
    B, N = load(base_p, idset), load(new_p, idset)
    mism = [i for i in ids if (i in B) != (i in N) or
            (i in B and sorted(c["sql"] or "" for c in B[i]["cands"]) != sorted(c["sql"] or "" for c in N[i]["cands"]))]
    n = len(ids)
    out = {"gen": a.gen, "new": str(new_p), "base": str(base_p), "n": n, "candidate_set_mismatch": len(mism)}
    print(f"DECISION n={n}, {len(set(db.values()))} DBs; covered base {len(B)} new {len(N)}; "
          f"questions whose candidate sets differ: {len(mism)} (must be 0)")
    hb, hn, orc = {}, {}, {}
    for i in ids:
        pb, pn = pick(B[i]["cands"]) if i in B else None, pick(N[i]["cands"]) if i in N else None
        hb[i], hn[i] = bool(pb and pb.get("ok")), bool(pn and pn.get("ok"))
        orc[i] = i in N and any(c.get("ok") for c in N[i]["cands"])
    pct = lambda h: 100 * sum(h.values()) / n  # noqa: E731
    c = cluster_signflip(hb, hn, db)
    q = question_signflip(hb, hn)
    v = verdict(c["delta"], c["p"])
    print("PRIMARY (one generator's k=32 pool, one validator picks)")
    print(f"  base validator {pct(hb):6.2f}   new (VALHARD) validator {pct(hn):6.2f}   oracle {pct(orc):6.2f}")
    print(f"  delta {c['delta']:+.2f}  DB sign-flip p={c['p']:.3f}  CI [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]  "
          f"McNemar p={q['p']:.3f} (+{q['wins']}/-{q['losses']})")
    print(f"  -> {v}   (PASS: delta >= +{PASS_DELTA} and p < {PASS_P}; KILL: delta < +{KILL_DELTA})")
    print("  per DB (new - base, questions): " + ", ".join(
        f"{d} {sum(hn[i] - hb[i] for i in ids if db[i] == d):+d}" for d in sorted(set(db.values()))))
    out.update(base=pct(hb), new=pct(hn), oracle=pct(orc), delta=c["delta"], p=c["p"], ci=c["ci"],
               mcnemar={"p": q["p"], "wins": q["wins"], "losses": q["losses"]}, verdict=v)

    # ---- secondary: pick-gap taxonomy -------------------------------------------------------------------------
    print("\nSECONDARY (info)")
    if MANUAL.exists():
        lab = json.load(open(MANUAL))
        rows = collections.defaultdict(lambda: [0, 0])
        for k, val in lab.items():
            if k.startswith("_"):
                continue
            i = f"bird_train_train_{k}"
            if i not in idset:
                continue
            t, ver = val[0], val[1]
            rows[(t, ver)][0] += 1
            rows[(t, ver)][1] += hn[i]
        print("  the 84 hand-labelled R1 pick-misses, fixed by the new validator (theme / verdict: fixed of n):")
        th = collections.defaultdict(lambda: [0, 0])
        for (t, ver), (m, f) in sorted(rows.items()):
            th[t][0] += m
            th[t][1] += f
        print("    " + ", ".join(f"{t} {f}/{m}" for t, (m, f) in sorted(th.items(), key=lambda kv: -kv[1][0])))
        vv = collections.defaultdict(lambda: [0, 0])
        for (t, ver), (m, f) in rows.items():
            vv[ver][0] += m
            vv[ver][1] += f
        print("    by verdict: " + ", ".join(f"{ver} {f}/{m}" for ver, (m, f) in sorted(vv.items())))
        out["manual_fixed"] = {f"{t}/{ver}": {"n": m, "fixed": f} for (t, ver), (m, f) in rows.items()}
        out["new_regressions_on_base_right"] = sum(hb[i] and not hn[i] for i in ids)

    def miss_buckets(scored, hits):
        ct = collections.Counter()
        for i in ids:
            if i not in scored or hits[i] or not orc[i]:
                continue
            C = scored[i]["cands"]
            p = pick(C)
            rights = [x for x in live(C) if x.get("ok")] or [x for x in C if x.get("ok")]
            if not any(x.get("ok") for x in live(C)) and any(x.get("runs") and x.get("nonnull") for x in C):
                ct["notlive"] += 1
                continue
            # the right candidate closest to the pick (hard types first, then fewest differing clauses)
            fp = sqldiff.feats(p["sql"])
            best = min(((not sqldiff.is_hard(k), nc, b) for k, b, nc in
                        (sqldiff.classify(r["sql"], p["sql"], fw=fp) if fp else ("unparsed", "other", 99)
                         for r in rights)), default=(True, 99, "other"))
            ct[best[2]] += 1
        return ct
    mb, mn = miss_buckets(B, hb), miss_buckets(N, hn)
    print("  misses by automatic clause-diff bucket (closest right vs pick; grain / sem / proj / ties / other / notlive):")
    for name, ct in (("base", mb), ("new ", mn)):
        print(f"    {name}: total {sum(ct.values())}  " + ", ".join(f"{k} {ct[k]}" for k in
                                                                  ("grain", "sem", "proj", "ties", "other", "notlive")))
    out["miss_buckets"] = {"base": dict(mb), "new": dict(mn)}

    # ---- secondary: top-2 pairwise accuracy -------------------------------------------------------------------
    # matched: the base validator's top-2 distinct live results with exactly one right; does each validator rank the
    # right class (its best member) above the other one?
    m_n = m_b = m_new = 0
    own = {}
    for name, S in (("base", B), ("new", N)):
        k = ok_ = 0
        for i in ids:
            if i not in S:
                continue
            cs = sorted(class_scores(S[i]["cands"]).values(), key=lambda x: -x[0])
            if len(cs) >= 2 and cs[0][1] != cs[1][1]:
                k += 1
                ok_ += cs[0][1]
        own[name] = (ok_, k)
    for i in ids:
        if i not in B or i not in N:
            continue
        cb = class_scores(B[i]["cands"])
        top = sorted(cb.items(), key=lambda kv: -kv[1][0])[:2]
        if len(top) < 2 or top[0][1][1] == top[1][1][1]:
            continue
        right_sig = top[0][0] if top[0][1][1] else top[1][0]
        wrong_sig = top[1][0] if top[0][1][1] else top[0][0]
        cn = class_scores(N[i]["cands"])
        if right_sig not in cn or wrong_sig not in cn:
            continue
        m_n += 1
        m_b += top[0][1][1]
        m_new += cn[right_sig][0] > cn[wrong_sig][0]
    print(f"  top-2 one-right pairwise accuracy, own top-2 (not matched): base {own['base'][0]}/{own['base'][1]} = "
          f"{own['base'][0] / max(own['base'][1], 1):.3f}, new {own['new'][0]}/{own['new'][1]} = "
          f"{own['new'][0] / max(own['new'][1], 1):.3f}")
    print(f"  top-2 pairwise accuracy on the base validator's one-right top-2 pairs (matched, n={m_n}): base "
          f"{m_b / max(m_n, 1):.3f}, new {m_new / max(m_n, 1):.3f}")
    out["top2_own"] = {k: {"right": x, "n": y} for k, (x, y) in own.items()}
    out["top2_matched"] = {"n": m_n, "base": m_b / max(m_n, 1), "new": m_new / max(m_n, 1)}
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
