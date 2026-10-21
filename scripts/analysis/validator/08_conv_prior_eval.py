"""08 gold_conventions: wording->convention prior fitted on BIRD train (held-out DBs excluded), applied to DECISION pools.
Prior is fitted on train only (LODO-free); the mixing weight delta IS fitted -> leave-one-DB-out. DECISION ids only.
  .venv/bin/python scripts/analysis/validator/08_conv_prior_eval.py [r1|valhard] > runs/analysis/validator/08_eval_<x>.out
"""
import json, math, sys, importlib.util
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from luthor.stats import cluster_signflip, question_signflip
spec = importlib.util.spec_from_file_location("cf", ROOT / "scripts/analysis/validator/08_conv_feats.py")
cf = importlib.util.module_from_spec(spec); spec.loader.exec_module(cf)

WHICH = sys.argv[1] if len(sys.argv) > 1 else "r1"
train_rows = json.load(open(ROOT / "runs/analysis/validator/08_train_rows.json"))
manual = {k: v for k, v in json.load(open(ROOT / "scripts/analysis/system1_pick_gap_manual.json")).items() if not k.startswith("_")}

# ---------------- components: key(w, c) -> context (None = not applicable), val(c) -> convention value
def ncol_b(c): return None if c["ncol"] is None else min(c["ncol"], 4)
COMP = {
    "tie": (lambda w, c: None if not (w["super"] or w["topn"]) or w["howmany"] else ("topn" if w["topn"] else ("evmax" if w["ev_max"] else "noevmax")),
            lambda c: c["tie"]),
    "cnt": (lambda w, c: None if not w["howmany"] or c["cnt"] == "none" else (c["join"], c["cnt_pk"]), lambda c: c["cnt"]),
    "x100": (lambda w, c: None if not w["pct"] else w["ev_mult100"], lambda c: c["x100"]),
    "cast": (lambda w, c: None if not w["pct"] else 1, lambda c: c["cast_real"]),
    "ordsel": (lambda w, c: None if not w["super"] or c["tie"] != "limit1" or c["order_in_sel"] is None else 1, lambda c: c["order_in_sel"]),
    "seldist": (lambda w, c: None if w["howmany"] else w["list"], lambda c: c["sel_distinct"]),
    "ncol": (lambda w, c: None if ncol_b(c) is None else ("full" if w["fullname"] else "howmany" if w["howmany"] else "namelist" if (w["name"] and w["list"]) else None),
             ncol_b),
}
CNT = {k: defaultdict(Counter) for k in COMP}
for r in train_rows:
    for k, (kf, vf) in COMP.items():
        ctx = kf(r["w"], r["c"])
        if ctx is not None:
            CNT[k][json.dumps(ctx, default=str)][json.dumps(vf(r["c"]), default=str)] += 1
VOCAB = {k: {v for ctx in CNT[k].values() for v in ctx} for k in COMP}

def logp(k, w, c):
    kf, vf = COMP[k]
    ctx = kf(w, c)
    if ctx is None:
        return 0.0
    cnt = CNT[k].get(json.dumps(ctx, default=str), Counter())
    n = sum(cnt.values()); V = len(VOCAB[k]) or 1
    return math.log((cnt[json.dumps(vf(c), default=str)] + 1) / (n + V))

def prior(w, c, comps=COMP):
    return sum(logp(k, w, c) for k in comps)

# ---------------- load DECISION pool
dec = json.load(open(ROOT / "data/sys1/heldout_v2/decision.json")); ids = list(dec["ids"]); idset = set(ids)
P, db = {}, {}
if WHICH == "r1":
    src = ROOT / "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl"
else:
    src = ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl"
for line in open(src):
    r = json.loads(line)
    if r["id"] not in idset:
        continue
    C = [c for c in r["cands"] if c.get("src", "r1") in ("r1", "both")]
    w = cf.wording(r["question"], r.get("context", ""))
    for c in C:
        c["conv"] = cf.sql_conv(c["sql"], r["db_id"])
        c["lp"] = {k: logp(k, w, c["conv"]) for k in COMP}
    r["cands"] = C; r["w"] = w
    P[r["id"]] = r
dbmap = {}
tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
for i in ids:
    dbmap[i] = tr[int(i.rsplit("_", 1)[1])]["db_id"]

def live(C):
    L = [c for c in C if c["runs"] and c["nonnull"]]
    return L or C

def pick(q, delta, comps):
    L = live(q["cands"])
    if not L: return None
    return max(L, key=lambda c: (c["rr"] + delta * sum(c["lp"][k] for k in comps), -c["rank"]))

def okv(i, delta, comps):
    q = P.get(i)
    if not q: return False
    c = pick(q, delta, comps)
    return bool(c and c["ok"])

base = {i: okv(i, 0, []) for i in ids}
print(f"[{WHICH}] base pick {100*sum(base.values())/len(ids):.2f}  (covered {len(P)}/{len(ids)})")
GRID = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

def lodo(comps):
    out, ch = {}, []
    dbs = sorted(set(dbmap.values()))
    for d in dbs:
        trn = [i for i in ids if dbmap[i] != d]
        best = max(GRID, key=lambda g: (sum(okv(i, g, comps) for i in trn), -g))
        ch.append(best)
        for i in ids:
            if dbmap[i] == d: out[i] = okv(i, best, comps)
    return out, ch

SETS = [["tie"], ["cnt"], ["x100"], ["cast"], ["ordsel"], ["seldist"], ["ncol"], list(COMP), ["tie", "x100", "ordsel", "ncol"]]
print("\n| components | in-sample best (delta) | LODO pick | delta pts | DB sign-flip p | McNemar +/- | deltas chosen | conv-misses fixed / misses fixed / breaks |")
print("|---|---|---|---|---|---|---|---|")
for comps in SETS:
    ins = {g: sum(okv(i, g, comps) for i in ids) for g in GRID}
    bg = max(GRID, key=lambda g: (ins[g], -g))
    res, ch = lodo(comps)
    cs = cluster_signflip(base, res, dbmap); qs = question_signflip(base, res)
    # attribution of LODO result vs manual labels
    fixed = [i for i in ids if res[i] and not base[i]]; broke = [i for i in ids if base[i] and not res[i]]
    fconv = sum(1 for i in fixed if manual.get(i.rsplit("_", 1)[1], ["", ""])[1] == "conv")
    print(f"| {'+'.join(comps)} | {100*ins[bg]/len(ids):.2f} ({bg}) | {100*sum(res.values())/len(ids):.2f} | {cs['delta']:+.2f} | {cs['p']:.3f} | +{qs['wins']}/-{qs['losses']} | {Counter(ch).most_common(3)} | {fconv}/{len(fixed)}/{len(broke)} |")

# ---------------- oracle-style sizing on the 84 misses: does prior prefer the best right class over the pick?
print("\n## Prior direction on misses (pick wrong, some live right): f(best right) - f(pick), all components")
by = defaultdict(Counter)
for i in ids:
    q = P.get(i)
    if not q or base[i]: continue
    L = live(q["cands"]); R = [c for c in L if c["ok"]]
    if not R: continue
    pk = pick(q, 0, []); rt = max(R, key=lambda c: c["rr"])
    m = manual.get(i.rsplit("_", 1)[1], ["?", "?"])
    for comps, nm in ((list(COMP), "all"), (["tie"], "tie"), (["cnt"], "cnt"), (["ncol"], "ncol"), (["x100"], "x100"), (["seldist"], "seldist"), (["ordsel"], "ordsel")):
        d = sum(rt["lp"][k] for k in comps) - sum(pk["lp"][k] for k in comps)
        s = "prefers_right" if d > 1e-9 else ("prefers_pick" if d < -1e-9 else "tie")
        by[(nm, m[0] + "/" + m[1])][s] += 1
        by[(nm, "ALL")][s] += 1
for k in sorted(by):
    print(k, dict(by[k]))

# guard: among questions the validator gets right, how often does prior prefer some other live wrong cand within rr margin 1?
print("\n## Risk side: base-right questions where a wrong live cand within rr margin m has higher prior (all comps)")
for m in (0.5, 1.0, 2.0):
    n = 0
    for i in ids:
        q = P.get(i)
        if not q or not base[i]: continue
        pk = pick(q, 0, []); f0 = sum(pk["lp"].values())
        if any((not c["ok"]) and pk["rr"] - c["rr"] < m and sum(c["lp"].values()) > f0 + 1e-9 for c in live(q["cands"])):
            n += 1
    print(f" margin<{m}: {n} at risk")

# ---------------- convention twin pairs: per component, questions where the best-rr right live cand and the best-rr
# wrong live cand differ in that component's value (both applicable). Gold-vs-train-majority, prior acc vs validator acc.
print("\n## Convention twins on DECISION (best right vs best wrong live cand differ in component value)")
print("| comp | n twins | prior prefers right | prior prefers wrong | prior tie | validator prefers right | both prior&val wrong | prior right & val wrong |")
print("|---|---|---|---|---|---|---|---|")
for k in COMP:
    n = pr = pw = pt = vr = bw = fix = 0
    for i in ids:
        q = P.get(i)
        if not q: continue
        L = live(q["cands"])
        R = [c for c in L if c["ok"]]; Wr = [c for c in L if not c["ok"]]
        if not R or not Wr: continue
        a = max(R, key=lambda c: c["rr"]); b = max(Wr, key=lambda c: c["rr"])
        kf, vf = COMP[k]
        if kf(q["w"], a["conv"]) is None or kf(q["w"], b["conv"]) is None: continue
        if vf(a["conv"]) == vf(b["conv"]): continue
        n += 1
        d = a["lp"][k] - b["lp"][k]
        pr += d > 1e-9; pw += d < -1e-9; pt += abs(d) <= 1e-9
        v = a["rr"] > b["rr"]; vr += v
        bw += (d < -1e-9) and not v; fix += (d > 1e-9) and not v
    print(f"| {k} | {n} | {pr} | {pw} | {pt} | {vr} | {bw} | {fix} |")
