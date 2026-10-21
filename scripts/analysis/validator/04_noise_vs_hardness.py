"""04: is BIRD-train gold noise concentrated in the 'hard' (greedy-wrong) training lists? Diagnostic, train side only.
Original labels: runs/phase1/p1w/p1w/r1/cands_train.jsonl; relabelled (BIRD-Verified) lists: data/valhard/r1/base/*.jsonl.
Verified coverage: data/raw/bird_verified (question_id = index into BIRD train.json).
"""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
fx = set()
for f in ("bird-verified-train.json", "bird-verified-val.json"):
    fx |= {int(r["question_id"]) for r in json.load(open(ROOT / "data/raw/bird_verified" / f))}
rel = {}
for f in ("train_lists", "val_lists"):
    for line in open(ROOT / f"data/valhard/r1/base/{f}.jsonl"):
        r = json.loads(line)
        rel[r["id"]] = r
tab = collections.Counter()
for line in open(ROOT / "runs/phase1/p1w/p1w/r1/cands_train.jsonl"):
    r = json.loads(line)
    cs = sorted([c for c in r["cands"] if c.get("sql")], key=lambda c: c["rank"])
    if not cs:
        continue
    g = "greedy_right" if cs[0].get("ok") else "greedy_wrong"
    ver = int(r["id"].rsplit("_", 1)[1]) in fx
    if not ver:
        tab[(g, "unverified")] += 1
        continue
    b = rel.get(r["id"])
    if b is None:
        tab[(g, "verified_dropped_from_lists")] += 1  # e.g. lost every right candidate after relabel, or no wrong
        continue
    if not b.get("gold_corrected"):
        tab[(g, "verified_same_gold")] += 1
        continue
    old = {c["sql"]: bool(c.get("ok")) for c in cs}
    new = {c["sql"]: bool(c.get("ok")) for c in b["cands"] if c.get("prov", "base") == "base"}
    flip = any(old.get(s) is not None and old[s] != v for s, v in new.items())
    tab[(g, "verified_corrected_" + ("flip" if flip else "noflip"))] += 1
for g in ("greedy_right", "greedy_wrong"):
    tot = sum(v for (gg, _), v in tab.items() if gg == g)
    print(g, tot, {k: v for (gg, k), v in sorted(tab.items()) if gg == g})
    vt = sum(v for (gg, k), v in tab.items() if gg == g and k != "unverified")
    bad = sum(v for (gg, k), v in tab.items() if gg == g and k in ("verified_corrected_flip", "verified_dropped_from_lists"))
    print(f"   verified {vt}; flipped-or-dropped {bad} = {100*bad/max(1,vt):.1f}% of verified")
