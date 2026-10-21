"""Question-variant ensembling on DECISION (runs/phase1/qvar/PREREG.md). Reads runs/phase1/qvar/qvar/scored_<v>.jsonl."""
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from phase1_eval import IDS, compare, load, z  # noqa: E402

Q = ROOT / "runs/phase1/qvar/qvar"
VARS = ["orig", "p1", "p2", "p3", "f", "s"]


def keyed(row):
    """{key: (best z, ok, live)} for one scored list; key = result signature (sql when it does not run)."""
    c = row["cands"]
    s = z(np.array([x.get("rr", -1e9) if x.get("rr") is not None else -1e9 for x in c], float))
    out = {}
    for j, x in enumerate(c):
        if x.get("sql") is None:
            continue
        live = bool(x.get("runs") and x.get("nonnull"))
        k = x.get("sig") if live and x.get("sig") else "sql:" + x["sql"]
        if k not in out or s[j] > out[k][0]:
            out[k] = (s[j], bool(x.get("ok")), live)
    return out


def combine(lists, weights):
    hits, orc = {}, {}
    for i in IDS:
        per = [(w, keyed(L[i])) for L, w in zip(lists, weights) if i in L]
        if not per:
            hits[i] = orc[i] = False
            continue
        allk = {}
        for _, kd in per:
            for k, (_, ok, live) in kd.items():
                allk[k] = (ok or allk.get(k, (False, False))[0], live or allk.get(k, (False, False))[1])
        lo = [min(v[0] for v in kd.values()) if kd else 0.0 for _, kd in per]
        score = {k: sum(w * (kd[k][0] if k in kd else m) for (w, kd), m in zip(per, lo)) for k in allk}
        live = [k for k in allk if allk[k][1]] or list(allk)
        hits[i] = bool(live) and allk[max(live, key=lambda k: score[k])][0]
        orc[i] = any(v[0] for v in allk.values())
    return hits, orc


def main():
    L = {v: load(Q / f"scored_{v}.jsonl") for v in VARS if (Q / f"scored_{v}.jsonl").exists()}
    missing = [v for v in VARS if v not in L]
    if missing:
        print("missing variants:", missing)
    base, ob = combine([L["orig"]], [1])
    print("PRIMARY")
    arm = [v for v in VARS if v in L]
    h, o = combine([L[v] for v in arm], [2 if v == "orig" else 1 for v in arm])
    c = compare("orig alone (R2, same box)", base, "orig x2 + all variants", h, ob, o)
    print(f"  -> {'PASS' if c['delta'] >= 1.0 and c['p'] < 0.10 else ('KILL' if c['delta'] < 0 else 'NO PASS')}")
    print("info (not claims)")
    for name, vs in (("orig x2 + rephrasings p1-p3", ["orig", "p1", "p2", "p3"]), ("orig x2 + filler-drop", ["orig", "f"]),
                     ("orig x2 + scramble", ["orig", "s"])):
        vs = [v for v in vs if v in L]
        h, o = combine([L[v] for v in vs], [2 if v == "orig" else 1 for v in vs])
        compare("orig alone", base, name, h, ob, o)
    for v in VARS[1:]:
        if v in L:
            h, o = combine([L[v]], [1])
            print(f"  {v} alone: {100 * sum(h.values()) / len(IDS):.2f} (oracle {100 * sum(o.values()) / len(IDS):.2f})")


if __name__ == "__main__":
    main()
