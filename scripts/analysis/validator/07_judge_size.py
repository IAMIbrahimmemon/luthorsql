"""07 backbone_scale: size effect among ZERO-SHOT decision models on the same k=4 DECISION shortlists (diagnostic only).

Inputs (DECISION ids only; *_reserved files are never opened):
  runs/judge/jev_*_k4.jsonl                    per-question pick over a 4-candidate shortlist (indices into the union pool)
  runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl union pool with ok labels (index check: shortlist sigs distinct)
  scripts/analysis/system1_pick_gap_manual.json hand verdicts (real/conv/noise) for the 84 R1-validator misses
Prints, per model: accuracy on discriminating shortlists (>=1 right and >=1 wrong), and on the hand-labelled misses
whose shortlist contains a right candidate, split by verdict. No fitting.

  python3 scripts/analysis/validator/07_judge_size.py > runs/analysis/validator/07_judge_size.out
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEC = set(json.load(open(ROOT / "data/sys1/heldout_v2/decision.json"))["ids"])
POOL = {}
for line in open(ROOT / "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl"):
    r = json.loads(line)
    if r["id"] in DEC:
        POOL[r["id"]] = r["cands"]
MAN = {k: v for k, v in json.load(open(ROOT / "scripts/analysis/system1_pick_gap_manual.json")).items()
       if not k.startswith("_")}
MISS = {f"bird_train_train_{k}": v for k, v in MAN.items()}


def main():
    files = sorted(f for f in glob.glob(str(ROOT / "runs/judge/jev_*_k4.jsonl")) if "reserved" not in f)
    print("model | n_disc | acc_disc | misses(right in shortlist): real / conv / noise  (right picks / n)")
    for f in files:
        name = Path(f).stem.removeprefix("jev_").removesuffix("_k4")
        nd = rd = 0
        bad_idx = 0
        mv = defaultdict(lambda: [0, 0])
        for line in open(f):
            j = json.loads(line)
            if j["id"] not in POOL:
                continue
            c = POOL[j["id"]]
            top = [i for i in j["top"] if i < len(c)]
            if len(top) != len(j["top"]) or len({c[i]["sig"] for i in top}) != len(top):
                bad_idx += 1
                continue
            oks = [bool(c[i]["ok"]) for i in top]
            if not (any(oks) and not all(oks)):
                continue
            right = bool(c[j["pick"]]["ok"]) if j.get("pick") is not None and j["pick"] < len(c) else False
            nd += 1
            rd += right
            if j["id"] in MISS:
                v = MISS[j["id"]][1]
                mv[v][0] += right
                mv[v][1] += 1
        s = " / ".join(f"{mv[v][0]}/{mv[v][1]}" for v in ("real", "conv", "noise"))
        print(f"{name} | {nd} | {rd / max(nd, 1):.3f} | {s} | index-check failures {bad_idx}")


if __name__ == "__main__":
    main()


def validator_on_shortlists():
    """R1 validator (rr) restricted to the same discriminating k=4 shortlists (argmax rr among the 4)."""
    f = ROOT / "runs/judge/jev_perplexity_pplx_decider_v1.1_27b_k4.jsonl"
    nd = rd = 0
    mv = defaultdict(lambda: [0, 0])
    for line in open(f):
        j = json.loads(line)
        if j["id"] not in POOL:
            continue
        c = POOL[j["id"]]
        top = j["top"]
        oks = [bool(c[i]["ok"]) for i in top]
        if not (any(oks) and not all(oks)):
            continue
        p = max(top, key=lambda i: c[i]["rr"] if c[i].get("rr") is not None else -1e9)
        right = bool(c[p]["ok"])
        nd += 1
        rd += right
        if j["id"] in MISS:
            v = MISS[j["id"]][1]
            mv[v][0] += right
            mv[v][1] += 1
    s = " / ".join(f"{mv[v][0]}/{mv[v][1]}" for v in ("real", "conv", "noise"))
    print(f"R1 validator rr (OmniSQL-7B LoRA) | {nd} | {rd / max(nd, 1):.3f} | {s}")


validator_on_shortlists()
