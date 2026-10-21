"""Phase 1 gates (docs/BUDGET500_PLAN.md) on held-out v2 DECISION (796 questions, 8 held-out BIRD-train DBs).
Never reads BIRD dev.

  python scripts/analysis/phase1_eval.py wide     # 1a: wide (k=32) R1 ∪ R2 pool, wide validators, sum_z
  python scripts/analysis/phase1_eval.py soup     # 1b: R1+R2 LoRA soup, R2 validator on its own k=8 list

Questions of DECISION with no candidate file row (grammar-uncovered: System-1 writes nothing) count as wrong in
every arm. A pick is right when its candidate's ok flag is set (execution match against gold, as computed at gen
time; heldout_v2.out shows it agrees with ORIGINAL-gold re-execution on all 670 covered DECISION items).
Pick rule as ens1_pick.py: the best-scored candidate among those that run and return non-null rows, else among all.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402

# HELDOUT_SPLIT=reserved (with HELDOUT_CONFIRM=1) scores the RESERVED set instead: one confirmation per final claim
SPLIT = os.environ.get("HELDOUT_SPLIT", "decision")
if SPLIT == "reserved" and os.environ.get("HELDOUT_CONFIRM") != "1":
    sys.exit("refusing to score the reserved set without HELDOUT_CONFIRM=1 (one final confirmation per claim)")
DEC = json.load(open(ROOT / f"data/sys1/heldout_v2/{SPLIT}.json"))
IDS = DEC["ids"]


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


def z(s):
    return (s - s.mean()) / (s.std() + 1e-6) if len(s) > 1 else s * 0


def pick(cands, score):
    live = [j for j, c in enumerate(cands) if c.get("runs") and c.get("nonnull")] or list(range(len(cands)))
    return bool(live) and bool(cands[max(live, key=lambda j: score[j])].get("ok"))


def rr_arr(cands):
    # candidates whose SQL failed to compile (sql None) carry no validator score: list minimum (never picked: no run)
    v = [c.get("rr") for c in cands]
    lo = min((x for x in v if x is not None), default=0.0)
    return np.array([lo if x is None else x for x in v], float)


def sum_z_hits(by1, by2):
    hits, orc = {}, {}
    for i in IDS:
        if i not in by2:
            hits[i] = orc[i] = False
            continue
        c1, c2 = by1[i]["cands"], by2[i]["cands"]
        assert [c["sql"] for c in c1] == [c["sql"] for c in c2]
        s = z(rr_arr(c1)) + z(rr_arr(c2))
        hits[i] = pick(c2, s)
        orc[i] = any(c.get("ok") for c in c2)
    return hits, orc


def single_hits(scored):
    hits, orc = {}, {}
    for i in IDS:
        r = scored.get(i)
        hits[i] = bool(r) and pick(r["cands"], np.array([c.get("rr", -1e9) for c in r["cands"]], float))
        orc[i] = bool(r) and any(c.get("ok") for c in r["cands"])
    return hits, orc


def db_of():
    m = {}
    for p in (ROOT / "runs/r1r2/r2/rr/dev_scored.jsonl",):
        m.update({i: r["db_id"] for i, r in load(p).items()})
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    for i in IDS:
        if i not in m:
            m[i] = tr[int(i.rsplit("_", 1)[1])]["db_id"]
    return {i: m[i] for i in IDS}


def compare(name_a, a, name_b, b, oa, ob):
    db = db_of()
    n = len(IDS)
    pct = lambda h: 100 * sum(h.values()) / n
    c = cluster_signflip(a, b, db)
    q = question_signflip(a, b)
    print(f"{SPLIT.upper()} n={n}, {len(set(db.values()))} DBs")
    print(f"  {name_a:28s} {pct(a):6.2f}   oracle {pct(oa):6.2f}")
    print(f"  {name_b:28s} {pct(b):6.2f}   oracle {pct(ob):6.2f}")
    print(f"  delta {c['delta']:+.2f}  DB sign-flip p={c['p']:.3f} CI [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]  "
          f"McNemar p={q['p']:.3f} (+{q['wins']}/-{q['losses']})")
    return c


def main(which):
    if which == "wide":
        base, ob = sum_z_hits(load(ROOT / "data/ens1/out/heldout_by_r1.jsonl"), load(ROOT / "data/ens1/out/heldout_by_r2.jsonl"))
        w = ROOT / "runs/phase1/p1w/p1w"
        new, on = sum_z_hits(load(w / "heldout_union_by_r1.jsonl"), load(w / "heldout_union_by_r2.jsonl"))
        c = compare("k=8 pool, sum_z (ENS1)", base, "k=32 pool, wide validators", new, ob, on)
        orc_gain = 100 * (sum(on.values()) - sum(ob.values())) / len(IDS)
        verdict = "PASS" if c["delta"] >= 1.0 and c["p"] < 0.10 else ("KILL" if c["delta"] < 0 or orc_gain < 3 else "NO PASS")
        print(f"  oracle gain {orc_gain:+.2f}  -> 1a {verdict} (pass: >= +1.0, p < 0.10; kill: < 0 or oracle gain < +3)")
    elif which == "soup":
        base, ob = single_hits(load(ROOT / "runs/r1r2/r2/rr/dev_scored.jsonl"))
        new, on = single_hits(load(ROOT / "runs/phase1/p1s/p1s/soup/heldout_scored.jsonl"))
        c = compare("R2, own k=8, R2 validator", base, "soup R1+R2, R2 validator", new, ob, on)
        verdict = "PASS" if c["delta"] >= 1.5 else ("KILL" if c["delta"] < 0 else "NO PASS")
        print(f"  -> 1b {verdict} (pass: >= R2 + 1.5; kill: below R2)")


if __name__ == "__main__":
    main(sys.argv[1])
