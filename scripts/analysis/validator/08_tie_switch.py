"""08: gated tie-convention switch. If the pick's best rival (live, different result) within rr margin m differs from the
pick in tie form (limit1 vs maxsub/none) and the train prior prefers the rival's form, switch. m fitted LODO."""
import json, sys, runpy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "r1"]
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    G = runpy.run_path(str(ROOT / "scripts/analysis/validator/08_conv_prior_eval.py"))
P, ids, dbmap, base, live = G["P"], G["ids"], G["dbmap"], G["base"], G["live"]
from luthor.stats import cluster_signflip, question_signflip
SAME = ["cnt", "x100", "sel_distinct", "ncol"]
def ok_m(i, m):
    q = P.get(i)
    if not q: return False
    L = live(q["cands"])
    pk = max(L, key=lambda c: (c["rr"], -c["rank"]))
    riv = [c for c in L if c["sig"] != pk["sig"] and pk["rr"] - c["rr"] < m and c["conv"]["tie"] != pk["conv"]["tie"]
           and all(c["conv"][k] == pk["conv"][k] for k in SAME) and c["lp"]["tie"] > pk["lp"]["tie"] + 0.5]
    if riv:
        pk = max(riv, key=lambda c: c["rr"])
    return bool(pk["ok"])
GRID = [0, 0.25, 0.5, 1, 1.5, 2, 3, 5]
ins = {m: sum(ok_m(i, m) for i in ids) for m in GRID}
print(sys.argv[1], "in-sample:", {m: round(100 * v / len(ids), 2) for m, v in ins.items()})
res, ch = {}, []
for d in sorted(set(dbmap.values())):
    tr = [i for i in ids if dbmap[i] != d]
    b = max(GRID, key=lambda m: (sum(ok_m(i, m) for i in tr), -m)); ch.append(b)
    for i in ids:
        if dbmap[i] == d: res[i] = ok_m(i, b)
cs = cluster_signflip(base, res, dbmap); qs = question_signflip(base, res)
print(f"LODO {100*sum(res.values())/len(ids):.2f} delta {cs['delta']:+.2f} p={cs['p']:.3f} +{qs['wins']}/-{qs['losses']} chosen {ch} per-db {cs['per_db']}")
