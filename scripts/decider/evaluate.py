"""Score Decider probabilities (scripts/decider/ft.py output) on DECISION (or RESERVED with HELDOUT_SPLIT=reserved
HELDOUT_CONFIRM=1), per runs/phase1/decider/PREREG.md.   python scripts/decider/evaluate.py runs/phase1/decider/dout"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from llm_judge import IDS, W, load, shortlist  # noqa: E402
from phase1_eval import SPLIT, compare  # noqa: E402

THR = 0.8


def picks(path):
    out = {}
    for ln in open(path):
        r = json.loads(ln)
        if r.get("probs"):
            j = max(range(len(r["probs"])), key=r["probs"].__getitem__)
            out[r["id"]] = (r["order"][j], r["probs"][j])
    return out


def hits(b1, b2, P, thr):
    base, h, orc = {}, {}, {}
    for i in IDS:
        if i not in b2:
            base[i] = h[i] = orc[i] = False
            continue
        c = b2[i]["cands"]
        top = shortlist(b1[i], b2[i], 4)
        p0 = top[0] if top else None
        ok = lambda j: j is not None and bool(c[j].get("ok"))  # noqa: E731
        p, conf = P.get(i, (None, 0.0))
        base[i], orc[i] = ok(p0), any(x.get("ok") for x in c)
        h[i] = ok(p if (p is not None and conf >= thr) else p0)
    return base, h, orc


def main():
    d = Path(sys.argv[1])
    b1, b2 = load(W / "heldout_union_by_r1.jsonl"), load(W / "heldout_union_by_r2.jsonl")
    print(json.dumps(json.load(open(d / "ft_meta.json"))) if (d / "ft_meta.json").exists() else "(no ft_meta)")
    zs, ft = picks(d / f"zs_{SPLIT}.jsonl"), picks(d / f"ft_{SPLIT}.jsonl")
    base, h, orc = hits(b1, b2, ft, THR)
    print("PRIMARY")
    c = compare("sum_z pick (wide pool)", base, f"FT Decider override p>={THR}", h, orc, orc)
    print(f"  -> {'PASS' if c['delta'] >= 1.0 and c['p'] < 0.10 else ('KILL' if c['delta'] < 0 else 'NO PASS')}")
    print("secondary (info)")
    _, za, _ = hits(b1, b2, zs, 0.0)
    _, fa, _ = hits(b1, b2, ft, 0.0)
    compare("zero-shot Decider alone (local)", za, "fine-tuned Decider alone", fa, orc, orc)
    _, zo, _ = hits(b1, b2, zs, THR)
    compare("sum_z pick", base, f"zero-shot override p>={THR} (local)", zo, orc, orc)


if __name__ == "__main__":
    main()
