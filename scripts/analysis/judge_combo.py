"""Pre-registered judge rules (runs/judge/PREREG.md) over the wide pool, from saved jev_judge.py records.
  python scripts/analysis/judge_combo.py                                         # DECISION
  HELDOUT_SPLIT=reserved HELDOUT_CONFIRM=1 python scripts/analysis/judge_combo.py  # RESERVED, once
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from jev_judge import out_path  # noqa: E402
from llm_judge import IDS, W, load, shortlist  # noqa: E402
from phase1_eval import SPLIT, compare  # noqa: E402

MERCURY, DECIDER, THR = "inception/mercury-decide:free", "perplexity/pplx-decider-v1.1-27b", 0.8


def recs(model):
    return {r["id"]: r for r in map(json.loads, open(out_path(model, 4)))}


def conf(r):
    p = r.get("pick")
    return (p, (r.get("prob") or {}).get(str(p), 0.0)) if p is not None else (None, 0.0)


def main():
    b1, b2 = load(W / "heldout_union_by_r1.jsonl"), load(W / "heldout_union_by_r2.jsonl")
    M, D = recs(MERCURY), recs(DECIDER)
    base, combo, dec, mer, dec_alone, orc = {}, {}, {}, {}, {}, {}
    agree = 0
    for i in IDS:
        if i not in b2:
            base[i] = combo[i] = dec[i] = mer[i] = dec_alone[i] = orc[i] = False
            continue
        c = b2[i]["cands"]
        top = shortlist(b1[i], b2[i], 4)
        p0 = top[0] if top else None
        ok = lambda j: j is not None and bool(c[j].get("ok"))  # noqa: E731
        (pm, cm), (pd, cd) = conf(M.get(i, {})), conf(D.get(i, {}))
        both = pm is not None and pm == pd and cm >= THR and cd >= THR
        agree += both and pm != p0
        base[i], orc[i] = ok(p0), any(x.get("ok") for x in c)
        combo[i] = ok(pm if both else p0)
        dec[i] = ok(pd if (pd is not None and cd >= THR) else p0)
        mer[i] = ok(pm if (pm is not None and cm >= THR) else p0)
        dec_alone[i] = ok(pd if pd is not None else p0)
    print(f"[{SPLIT}] combo overrides the validator on {agree} questions")
    print("PRIMARY (pre-registered): COMBO, Mercury AND Decider agree, both p >= 0.8")
    c = compare("sum_z pick (wide pool)", base, "COMBO", combo, orc, orc)
    print(f"  -> {'PASS' if c['delta'] >= 1.0 else 'NO PASS'} on size (>= +1.0); see McNemar p above (< 0.10 to pass)")
    print("secondary (not claims):")
    for name, h in (("Decider override p>=0.8", dec), ("Mercury override p>=0.8", mer), ("Decider alone", dec_alone)):
        compare("sum_z pick (wide pool)", base, name, h, orc, orc)


if __name__ == "__main__":
    main()
