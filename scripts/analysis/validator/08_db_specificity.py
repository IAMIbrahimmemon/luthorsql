"""08: how much of each convention is wording-predictable vs annotator(DB)-specific. Train rows only (held-out DBs excluded).
Accuracy of (a) global wording-context majority, (b) own-DB majority within the same context (leave-one-question-out).
Also lists the R1 DECISION 'conv' misses with their wording context (manual labels from system1_pick_gap)."""
import json, importlib.util
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
rows = json.load(open(ROOT / "runs/analysis/validator/08_train_rows.json"))
C = {
 "tie (superlative q; limit1 vs other)": (lambda r: ("evmax" if r["w"]["ev_max"] else "noev") if r["w"]["super"] and not r["w"]["howmany"] and not r["w"]["topn"] else None, lambda r: r["c"]["tie"] == "limit1"),
 "cnt (how many, COUNT form; distinct vs not)": (lambda r: (r["c"]["join"], r["c"]["cnt_pk"]) if r["w"]["howmany"] and r["c"]["cnt"] in ("col", "distinct", "star") else None, lambda r: r["c"]["cnt"] == "distinct"),
 "x100 (percent q)": (lambda r: r["w"]["ev_mult100"] if r["w"]["pct"] else None, lambda r: r["c"]["x100"]),
 "seldist (no how-many)": (lambda r: r["w"]["list"] if not r["w"]["howmany"] else None, lambda r: r["c"]["sel_distinct"]),
 "ncol>=2 ('name' list q)": (lambda r: r["w"]["fullname"] if r["w"]["name"] and r["w"]["list"] else None, lambda r: (r["c"]["ncol"] or 1) >= 2),
}
print("| convention | n | global-context majority acc | own-DB majority acc (LOO) | gain from knowing the DB |")
print("|---|---|---|---|---|")
for name, (kf, vf) in C.items():
    R = [(r["db"], kf(r), vf(r)) for r in rows if kf(r) is not None]
    g = defaultdict(Counter); d = defaultdict(Counter)
    for db, k, v in R: g[k][v] += 1; d[(db, k)][v] += 1
    ga = sum(1 for db, k, v in R if g[k].most_common(1)[0][0] == v) / len(R)
    da = 0
    for db, k, v in R:
        c = d[(db, k)].copy(); c[v] -= 1
        if sum(c.values()) == 0:
            pred = g[k].most_common(1)[0][0]
        else:
            pred = c[True] > c[False] if c[True] != c[False] else g[k].most_common(1)[0][0]
        da += pred == v
    da /= len(R)
    print(f"| {name} | {len(R)} | {ga:.3f} | {da:.3f} | {100*(da-ga):+.1f} pts |")
