"""Validator research 06 (feature fusion): validator score + cheap structured features in a small learned model.

  .venv/bin/python scripts/analysis/validator/fusion06.py            # all experiments, writes runs/analysis/validator/06_*

Model: conditional logit (listwise softmax over a question's live candidates; loss = -log sum_{ok} p), linear or a
1-hidden-layer MLP, L2. Torch on CPU, no sklearn.

Data (read-only):
  train prior   runs/phase1/p1w/p1w/r1/cands_train.jsonl  (BIRD-train lists, 46 train DBs, NO validator scores),
                minus the 179 VALHARD-val ids
  fit weight    runs/phase1/valhard/valhard/rr/val_scored.jsonl (179 val lists, 4 train DBs, VALHARD scores)
  test          DECISION ids only: R1 scores from heldout_union_by_r1.jsonl (src in {r1,both}), VALHARD scores from
                runs/phase1/valhard/valhard/heldout_by_new.jsonl. 126 DECISION ids absent from the files count wrong.
Rule (as pick_gap): live = runs and nonnull (else all); pick = argmax score over live.
Experiments
  E0  baselines (R1 57.29, VALHARD 59.05) and feature-only prior on DECISION
  E1  z(rr) + w * prior, prior trained on BIRD-train (no DECISION), w fit on the 179 val lists (VALHARD scores)
  E1L same, but w chosen leave-one-DB-out on DECISION (diagnostic)
  E2  full fusion (rr features + structural) trained on the 179 val lists, applied to DECISION
  E3  full fusion trained leave-one-DB-out on DECISION (diagnostic): linear, MLP, and with the train prior as feature
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402

OUT = ROOT / "runs/analysis/validator"
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)

DEC = json.load(open(ROOT / "data/sys1/heldout_v2/decision.json"))["ids"]
DECS = set(DEC)

RES = re.compile(r"^(\d+) rows?(?: x (\d+) cols? \(([^)]*)\))?(?:; first: (.*))?$")
IDENT = re.compile(r'"([^"]+)"')
STRLIT = re.compile(r"'((?:[^']|'')*)'")


def parse_res(s):
    if not s:
        return dict(rows=-1, cols=0, types=[], first="")
    m = RES.match(s)
    if not m:
        return dict(rows=-1, cols=0, types=[], first="")
    return dict(rows=int(m.group(1)), cols=int(m.group(2) or 0),
                types=[t.strip() for t in (m.group(3) or "").split(",") if t.strip()], first=m.group(4) or "")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


CUES = {
    "howmany": r"\bhow many\b|\bnumber of\b|\bcount\b|\btotal number\b",
    "list": r"\blist\b|\bname (all|the)\b|\bwhich\b|\bwhat are\b|\bgive (all|the)\b|\bindicate\b|\bstate\b",
    "pct": r"\bpercent|\bpercentage\b|\bratio\b|\bproportion\b|\brate\b|\bhow many times\b",
    "superl": r"\bmost\b|\bhighest\b|\blowest\b|\bleast\b|\btop\b|\blargest\b|\bsmallest\b|\bbest\b|\bworst\b|\best\b|\bmax|\bmin",
    "avg": r"\baverage\b|\bmean\b|\bavg\b",
    "diff": r"\bdifference\b|\bhow (much|many) more\b|\bmore than\b",
    "distinct": r"\bdistinct\b|\bdifferent\b|\bunique\b",
    "sum": r"\btotal\b|\bsum\b|\baltogether\b",
    "both": r"\band\b.*\b(their|its|with)\b|\balong with\b|\btogether with\b|\bas well as\b",
}
CUES = {k: re.compile(v, re.I) for k, v in CUES.items()}
EVOPS = {
    "div": re.compile(r"DIVIDE|\bdivide|/", re.I),
    "x100": re.compile(r"\*\s*100|MULTIPLY\(.*100|percent", re.I),
    "count": re.compile(r"COUNT", re.I),
    "sum": re.compile(r"\bSUM\b", re.I),
    "avg": re.compile(r"\bAVG\b|average", re.I),
    "max": re.compile(r"\bMAX\b", re.I),
    "min": re.compile(r"\bMIN\b", re.I),
    "distinct": re.compile(r"DISTINCT", re.I),
    "sub": re.compile(r"SUBTRACT|\s-\s", re.I),
}


def sql_feats(sql):
    u = sql.upper()
    return dict(
        limit=float(" LIMIT " in u), distinct=float("DISTINCT" in u), count=float("COUNT(" in u),
        countd=float("COUNT(DISTINCT" in u), orderby=float("ORDER BY" in u), groupby=float("GROUP BY" in u),
        joins=float(u.count(" JOIN ")), subq=float(u.count("(SELECT")), cast=float("CAST(" in u),
        div=float("/" in u), x100=float("100" in u), case=float("CASE " in u or "IIF(" in u),
        sumf=float("SUM(" in u), avgf=float("AVG(" in u), maxf=float("MAX(" in u), minf=float("MIN(" in u),
        minus=float(" - " in u), ands=float(u.count(" AND ")), ors=float(u.count(" OR ")),
        like=float(" LIKE " in u), having=float(" HAVING " in u), notnull=float("IS NOT NULL" in u),
        ntok=len(sql.split()), nsel=float(_nsel(u)),
    )


def _nsel(u):
    m = re.match(r"SELECT (.*?) FROM ", u)
    if not m:
        return 1
    depth, n = 0, 1
    for ch in m.group(1):
        depth += ch == "("
        depth -= ch == ")"
        n += ch == "," and depth == 0
    return n


def list_feats(r, cands, score_key):
    """Feature matrix for one question's candidate list. score_key None -> no validator features."""
    q, ev = r.get("question") or "", r.get("context") or ""
    cue = {k: float(bool(p.search(q))) for k, p in CUES.items()}
    evop = {k: float(bool(p.search(ev))) for k, p in EVOPS.items()}
    # evidence terms: quoted literals + identifiers used by any candidate that the evidence names
    ev_lits = {norm(x.replace("''", "'")) for x in STRLIT.findall(ev)} - {""}
    ev_n = norm(ev)
    all_ids = set()
    for c in cands:
        all_ids |= set(IDENT.findall(c["sql"]))
    ev_ids = {norm(x) for x in all_ids if len(norm(x)) >= 3 and norm(x) in ev_n}
    n = len(cands)
    sig_ct = Counter(c.get("sig") for c in cands)
    sig_best_rank = {}
    for c in cands:
        s = c.get("sig")
        sig_best_rank[s] = min(sig_best_rank.get(s, 1e9), c.get("rank", 0))
    ranks = [c.get("rank", 0) for c in cands]
    costs = [c.get("cost", 0.0) for c in cands]
    rs = [parse_res(c.get("res")) for c in cands]
    sfs = [sql_feats(c["sql"]) for c in cands]
    ncols = [x["cols"] for x in rs]
    mean = {k: np.mean([s[k] for s in sfs]) for k in sfs[0]} if sfs else {}
    rank0_sig = cands[ranks.index(min(ranks))].get("sig") if cands else None
    nsig = len(sig_ct)
    rows = []
    for c, rp, sf in zip(cands, rs, sfs):
        sql_n = norm(c["sql"].replace("''", "'"))
        ids_c = {norm(x) for x in IDENT.findall(c["sql"])}
        lit_cov = np.mean([lit in sql_n for lit in ev_lits]) if ev_lits else 1.0
        id_cov = np.mean([i in ids_c for i in ev_ids]) if ev_ids else 1.0
        rows_ = rp["rows"]
        scalar = float(rows_ == 1 and rp["cols"] == 1)
        nulltype = float("null" in rp["types"])
        firstnone = float("None" in rp["first"])
        zero = float(rp["first"].strip("(),' ") in ("0", "0.0"))
        f = dict(
            rank=c.get("rank", 0) / 32.0, lrank=math.log1p(c.get("rank", 0)), rank0=float(c.get("rank", 0) == min(ranks)),
            cost=c.get("cost", 0.0) / 10.0, dcost=(c.get("cost", 0.0) - min(costs)) / 10.0,
            runs=float(bool(c.get("runs"))), nonnull=float(bool(c.get("nonnull"))),
            empty=float(rows_ == 0), lrows=math.log1p(max(rows_, 0)), ncols=float(rp["cols"]),
            dcols=float(rp["cols"] - min([x for x in ncols if x > 0] or [0])), scalar=scalar, nulltype=nulltype,
            firstnone=firstnone, zero=zero,
            cls=sig_ct[c.get("sig")] / n, lcls=math.log(sig_ct[c.get("sig")]), cls_rank=sig_best_rank[c.get("sig")] / 32.0,
            cls_has_rank0=float(c.get("sig") == rank0_sig), lnsig=math.log(nsig),
            lit_cov=lit_cov, id_cov=id_cov, lit_miss=float(lit_cov < 1), id_miss=float(id_cov < 1),
            # question-cue x SQL/result interactions
            hm_count=cue["howmany"] * sf["count"], hm_scalar=cue["howmany"] * scalar, hm_countd=cue["howmany"] * sf["countd"],
            hm_sum=cue["howmany"] * sf["sumf"], list_multi=cue["list"] * float(rows_ > 1), list_scalar=cue["list"] * scalar,
            list_cols=cue["list"] * rp["cols"], pct_div=cue["pct"] * sf["div"], pct_100=cue["pct"] * sf["x100"],
            pct_cast=cue["pct"] * sf["cast"], sup_limit=cue["superl"] * sf["limit"], sup_order=cue["superl"] * sf["orderby"],
            sup_maxsub=cue["superl"] * sf["subq"], avg_avg=cue["avg"] * sf["avgf"], diff_minus=cue["diff"] * sf["minus"],
            dist_distinct=cue["distinct"] * sf["distinct"], sum_sum=cue["sum"] * sf["sumf"],
            both_cols=cue["both"] * float(rp["cols"] > 1),
            # evidence-op coverage
            ev_div=evop["div"] * sf["div"], ev_div_miss=evop["div"] * (1 - sf["div"]), ev_100=evop["x100"] * sf["x100"],
            ev_100_miss=evop["x100"] * (1 - sf["x100"]), ev_count_miss=evop["count"] * (1 - sf["count"]),
            ev_sum_miss=evop["sum"] * (1 - sf["sumf"]), ev_avg_miss=evop["avg"] * (1 - sf["avgf"]),
            ev_max_miss=evop["max"] * (1 - max(sf["maxf"], sf["orderby"])), ev_min_miss=evop["min"] * (1 - max(sf["minf"], sf["orderby"])),
            ev_dist=evop["distinct"] * sf["distinct"], ev_sub_miss=evop["sub"] * (1 - sf["minus"]),
        )
        for k in ("limit", "distinct", "count", "countd", "orderby", "groupby", "subq", "cast", "case", "like",
                  "having", "notnull"):
            f["s_" + k] = sf[k]
        for k in ("joins", "ands", "ntok", "nsel", "subq"):
            f["r_" + k] = (sf[k] - mean[k]) / (10.0 if k == "ntok" else 1.0)
        rows.append(f)
    names = sorted(rows[0]) if rows else []
    X = np.array([[f[k] for k in names] for f in rows], dtype=np.float32)
    if score_key is not None:
        s = np.array([c.get(score_key, -50.0) for c in cands], dtype=np.float32)
        live = np.array([bool(c.get("runs") and c.get("nonnull")) for c in cands])
        ref = s[live] if live.any() else s
        z = (s - ref.mean()) / (ref.std() + 1e-6)
        gap = s - ref.max()
        rrrank = np.argsort(np.argsort(-s)).astype(np.float32)
        # class-level validator aggregates
        sig = [c.get("sig") for c in cands]
        cmax = defaultdict(lambda: -1e9)
        e = np.exp(z - z.max())
        cmass = defaultdict(float)
        for g, zz, ee in zip(sig, z, e):
            cmax[g] = max(cmax[g], zz)
            cmass[g] += ee
        tot = sum(cmass.values())
        V = np.stack([z, gap, np.log1p(rrrank), np.array([cmax[g] for g in sig]), np.array([cmass[g] / tot for g in sig]),
                      (rrrank == 0).astype(np.float32)], 1).astype(np.float32)
        X = np.concatenate([V, X], 1)
        names = ["v_z", "v_gap", "v_lrank", "v_clsmax", "v_clsmass", "v_top"] + names
    return X, names


def load_lists(path, ids=None, score_key=None, src_filter=None, drop_ids=()):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if ids is not None and r["id"] not in ids:
            continue
        if r["id"] in drop_ids:
            continue
        cands = r["cands"]
        if src_filter:
            cands = [c for c in cands if c.get("src") in src_filter]
        if not cands:
            continue
        for c in cands:
            c["sql"] = c.get("sql") or ""
        X, names = list_feats(r, cands, score_key)
        live = np.array([bool(c.get("runs") and c.get("nonnull")) for c in cands])
        if not live.any():
            live[:] = True
        y = np.array([bool(c.get("ok")) for c in cands])
        rr = np.array([c.get(score_key, -50.0) for c in cands], dtype=np.float32) if score_key else None
        out[r["id"]] = dict(X=X, y=y, live=live, db=r["db_id"], rr=rr, names=names, sql=[c["sql"] for c in cands],
                            sig=[c.get("sig") or c["sql"] for c in cands])
    return out


# ---------------------------------------------------------------------------------------------------------- model
class CLogit:
    def __init__(self, hidden=0, l2=1e-3, epochs=200):
        self.hidden, self.l2, self.epochs = hidden, l2, epochs

    def _net(self, d):
        if self.hidden:
            return torch.nn.Sequential(torch.nn.Linear(d, self.hidden), torch.nn.Tanh(), torch.nn.Linear(self.hidden, 1))
        return torch.nn.Linear(d, 1, bias=False)

    def fit(self, lists, cols=None):
        L = [x for x in lists if x["y"][x["live"]].any() and not x["y"][x["live"]].all()]
        Xs = [x["X"][x["live"]] for x in L]
        if cols is not None:
            Xs = [X[:, cols] for X in Xs]
        self.cols = cols
        A = np.concatenate(Xs)
        self.mu, self.sd = A.mean(0), A.std(0) + 1e-6
        X = torch.tensor((A - self.mu) / self.sd)
        Y = torch.tensor(np.concatenate([x["y"][x["live"]] for x in L]))
        seg = torch.tensor(np.repeat(np.arange(len(L)), [len(v) for v in Xs]))
        G = len(L)
        torch.manual_seed(0)
        self.net = self._net(X.shape[1])
        opt = torch.optim.LBFGS(self.net.parameters(), lr=0.5, max_iter=self.epochs, line_search_fn="strong_wolfe")

        def lse(v):  # segment logsumexp
            m = torch.full((G,), -1e30).scatter_reduce(0, seg, v, "amax")
            return m + torch.log(torch.zeros(G).index_add(0, seg, torch.exp(v - m[seg])))

        def closure():
            opt.zero_grad()
            s = self.net(X).squeeze(1)
            loss = (lse(s) - lse(torch.where(Y, s, torch.tensor(-1e30)))).mean()
            loss = loss + self.l2 * sum((p ** 2).sum() for n_, p in self.net.named_parameters() if "weight" in n_)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def score(self, X):
        if self.cols is not None:
            X = X[:, self.cols]
        with torch.no_grad():
            return self.net(torch.tensor((X - self.mu) / self.sd)).squeeze(1).numpy()

    def weights(self, names):
        if self.hidden:
            return {}
        w = self.net.weight.detach().numpy()[0]
        nm = names if self.cols is None else [names[i] for i in self.cols]
        return dict(sorted(zip(nm, w.tolist()), key=lambda t: -abs(t[1])))


def pick_acc(D, scorer, ids=DEC):
    """scorer(rec) -> scores; returns {id: bool} over all DECISION ids (absent ids are wrong)."""
    res = {}
    for i in ids:
        r = D.get(i)
        if r is None:
            res[i] = False
            continue
        s = scorer(r)
        idx = np.where(r["live"])[0]
        j = idx[np.argmax(s[idx])]
        res[i] = bool(r["y"][j])
    return res


def pct(d):
    return 100 * sum(d.values()) / len(d)


DBOF = {}


def cmp(base, arm):
    c = cluster_signflip(base, arm, DBOF)
    q = question_signflip(base, arm)
    return f"{pct(arm):6.2f}  d {c['delta']:+5.2f}  DB p={c['p']:.3f}  (+{q['wins']}/-{q['losses']}, McN p={q['p']:.3f})"


def zs(r):
    s = r["rr"]
    ref = s[r["live"]]
    return (s - ref.mean()) / (ref.std() + 1e-6)


def main():
    log = open(OUT / "06_fusion.out", "w")

    def P(*a):
        print(*a)
        print(*a, file=log, flush=True)

    val_ids = {json.loads(l)["id"] for l in open(ROOT / "runs/phase1/valhard/valhard/rr/val_scored.jsonl")}
    P("loading ...")
    TR = load_lists(ROOT / "runs/phase1/p1w/p1w/r1/cands_train.jsonl", drop_ids=val_ids | DECS)
    VA = load_lists(ROOT / "runs/phase1/valhard/valhard/rr/val_scored.jsonl", score_key="rr")
    R1 = load_lists(ROOT / "runs/phase1/p1w/p1w/heldout_union_by_r1.jsonl", ids=DECS, score_key="rr", src_filter=("r1", "both"))
    VH = load_lists(ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl", ids=DECS, score_key="rr")
    assert all(R1[i]["sql"] == VH[i]["sql"] for i in R1)
    for i, r in VH.items():
        DBOF[i] = r["db"]
    for i in DEC:
        DBOF.setdefault(i, "absent")
    # absent ids are wrong in every arm; sign-flip over DBs needs their real DB -> take from decision.json database_counts?
    # they don't change any delta, so group them in their own 'absent' cluster (contributes 0 to D).
    trdbs = {r["db"] for r in TR.values()}
    P(f"train lists {len(TR)} ({len(trdbs)} DBs), val lists {len(VA)}, DECISION lists {len(VH)}/{len(DEC)}; "
      f"DB overlap train/decision: {trdbs & {r['db'] for r in VH.values()}}")
    names_s = next(iter(TR.values()))["names"]
    names_v = next(iter(VH.values()))["names"]
    nv = 6  # validator columns prefix in score_key lists
    assert names_v[nv:] == names_s

    base_r1 = pick_acc(R1, lambda r: r["rr"])
    base_vh = pick_acc(VH, lambda r: r["rr"])
    orc = {i: bool(i in VH and VH[i]["y"][VH[i]["live"]].any()) for i in DEC}
    P(f"\nE0 baselines: R1 {pct(base_r1):.2f}  VALHARD {pct(base_vh):.2f} ({cmp(base_r1, base_vh)})  live-oracle {pct(orc):.2f}")
    P(f"   greedy (rank0 among live): {pct(pick_acc(VH, lambda r: -r['X'][:, nv + names_s.index('rank')])):.2f}")

    # ---------------- E0/E1: prior trained on BIRD-train only
    prior = {}
    for tag, hid in (("lin", 0), ("mlp", 16)):
        m = CLogit(hidden=hid, l2=1e-3 if hid == 0 else 1e-4).fit(list(TR.values()))
        prior[tag] = m
        p = pick_acc(VH, lambda r: m.score(r["X"][:, nv:]))
        # in-domain sanity: train-DB val lists
        pv = {i: v for i, v in pick_acc(VA, lambda r: m.score(r["X"][:, nv:]), ids=list(VA)).items()}
        pvr = pick_acc(VA, lambda r: r["rr"], ids=list(VA))
        P(f"\nE0 prior-only ({tag}) on DECISION: {cmp(base_r1, p)} vs R1 | on val lists {pct(pv):.2f} vs VALHARD {pct(pvr):.2f}")
        if tag == "lin":
            w = m.weights(names_s)
            P("   top prior weights: " + ", ".join(f"{k} {v:+.2f}" for k, v in list(w.items())[:25]))
    for tag, m in prior.items():
        for D in (VA, R1, VH):
            for r in D.values():
                pr = m.score(r["X"][:, nv:])
                ref = pr[r["live"]]
                r["prior_" + tag] = (pr - ref.mean()) / (ref.std() + 1e-6)
    WGRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    for tag in prior:
        # E1: w from the 179 val lists (VALHARD scores)
        vacc = {w: pct(pick_acc(VA, lambda r, w=w: zs(r) + w * r["prior_" + tag], ids=list(VA))) for w in WGRID}
        wbest = max(WGRID, key=lambda w: (vacc[w], -w))
        P(f"\nE1 z(rr)+w*prior_{tag}: val-list acc by w " + " ".join(f"{w}:{a:.1f}" for w, a in vacc.items()) + f" -> w={wbest}")
        for nm, D, base in (("R1", R1, base_r1), ("VALHARD", VH, base_vh)):
            arm = pick_acc(D, lambda r: zs(r) + wbest * r["prior_" + tag])
            P(f"   {nm:7s} + prior w={wbest}: {cmp(base, arm)}" + (f"   [vs R1 {cmp(base_r1, arm)}]" if nm == "VALHARD" else ""))
        # sensitivity (not a fit): every w on DECISION
        for nm, D, base in (("R1", R1, base_r1), ("VALHARD", VH, base_vh)):
            P(f"   {nm} w-sweep (diagnostic, unfitted): " + " ".join(
                f"{w}:{pct(pick_acc(D, lambda r, w=w: zs(r) + w * r['prior_' + tag])):.2f}" for w in WGRID))
        # E1L: w by leave-one-DB-out on DECISION
        for nm, D, base in (("R1", R1, base_r1), ("VALHARD", VH, base_vh)):
            arm, chosen = {}, {}
            dbs = sorted({r["db"] for r in D.values()})
            for db in dbs:
                trn = [i for i in DEC if DBOF[i] != db]
                acc = {w: pct(pick_acc(D, lambda r, w=w: zs(r) + w * r["prior_" + tag], ids=trn)) for w in WGRID}
                wb = max(WGRID, key=lambda w: (acc[w], -w))
                chosen[db] = wb
                tst = [i for i in DEC if DBOF[i] == db]
                arm.update(pick_acc(D, lambda r: zs(r) + wb * r["prior_" + tag], ids=tst))
            arm.update({i: False for i in DEC if DBOF[i] == "absent"})
            P(f"E1L {nm:7s} + prior_{tag} (w LODO {sorted(Counter(chosen.values()).items())}): {cmp(base, arm)}")

    # ---------------- E2: full fusion trained on the 179 val lists (VALHARD scores)
    for tag, hid, l2 in (("lin", 0, 1e-2), ("lin_l2hi", 0, 1e-1)):
        m = CLogit(hidden=hid, l2=l2).fit(list(VA.values()))
        for nm, D, base in (("R1", R1, base_r1), ("VALHARD", VH, base_vh)):
            arm = pick_acc(D, lambda r: m.score(r["X"]))
            P(f"\nE2 full fusion on val lists ({tag}) -> {nm}: {cmp(base, arm)}" if nm == "R1" else
              f"   -> {nm}: {cmp(base, arm)}")

    # ---------------- E3: LODO on DECISION (diagnostic)
    def add_prior(D):
        for r in D.values():
            r["Xp"] = np.concatenate([r["X"], r["prior_lin"][:, None], r["prior_mlp"][:, None]], 1).astype(np.float32)

    add_prior(R1)
    add_prior(VH)
    vcols = list(range(nv))
    configs = [
        ("rr-only feats (6)", 0, 1e-2, "X", vcols),
        ("rr + all feats lin", 0, 1e-2, "X", None),
        ("rr + all feats lin l2=0.1", 0, 1e-1, "X", None),
        ("rr + all feats mlp16", 16, 1e-3, "X", None),
        ("z + prior_lin + prior_mlp", 0, 1e-2, "Xp", [0, len(names_v), len(names_v) + 1]),
        ("rr feats + priors", 0, 1e-2, "Xp", vcols + [len(names_v), len(names_v) + 1]),
        ("all + priors lin", 0, 1e-2, "Xp", None),
    ]
    P("\nE3 LODO on DECISION (fit on 7 DECISION DBs, test the 8th; diagnostic only)")
    for nm, D, base in (("R1", R1, base_r1), ("VALHARD", VH, base_vh)):
        dbs = sorted({r["db"] for r in D.values()})
        for cname, hid, l2, key, cols in configs:
            arm = {i: False for i in DEC if DBOF[i] == "absent"}
            for db in dbs:
                trn = [dict(r, X=r[key]) for r in D.values() if r["db"] != db]
                m = CLogit(hidden=hid, l2=l2).fit(trn, cols=cols)
                tst = [i for i in DEC if DBOF[i] == db]
                arm.update(pick_acc(D, lambda r: m.score(r[key]), ids=tst))
            P(f"  {nm:7s} {cname:28s} {cmp(base, arm)}" + (f"   [vs R1 {cmp(base_r1, arm)}]" if nm == "VALHARD" else ""))

    # ---------------- E4: gated top-2 duel. If the z-margin between the best two distinct live results < t, let
    # z + w*prior decide between those two results only. (t, w) chosen LODO on DECISION (diagnostic).
    def duel(r, t, w, tag):
        z = zs(r)
        live = np.where(r["live"])[0]
        sig = r["sig"]
        best = {}
        for j in live[np.argsort(-z[live])]:
            best.setdefault(sig[j], j)
        top = sorted(best.values(), key=lambda j: -z[j])[:2]
        s = np.full(len(z), -1e9, dtype=np.float32)
        if len(top) < 2 or z[top[0]] - z[top[1]] >= t:
            s[top[0]] = 1
            return s
        for j in top:
            s[j] = z[j] + w * r["prior_" + tag][j]
        return s

    P("\nE4 gated top-2 duel by prior (diagnostic)")
    TG, WG = [0.1, 0.25, 0.5, 0.75, 1.0], [0.25, 0.5, 1.0, 2.0, 5.0]
    for tag in prior:
        for nm, D, base in (("R1", R1, base_r1), ("VALHARD", VH, base_vh)):
            # pairwise diagnostic: among top-2 duels where exactly one side is right, how often does the prior prefer it?
            agree = tot = 0
            fix = [0, 0]  # validator wrong (#1 bad, #2 good): prior prefers #2 / total
            brk = [0, 0]  # validator right: prior prefers the wrong #2 / total
            for i, r in D.items():
                z = zs(r)
                live = np.where(r["live"])[0]
                best = {}
                for j in live[np.argsort(-z[live])]:
                    best.setdefault(r["sig"][j], j)
                top = sorted(best.values(), key=lambda j: -z[j])[:2]
                if len(top) == 2 and r["y"][top[0]] != r["y"][top[1]]:
                    tot += 1
                    good, bad = (top[0], top[1]) if r["y"][top[0]] else (top[1], top[0])
                    pg = r["prior_" + tag][good] > r["prior_" + tag][bad]
                    agree += pg
                    if r["y"][top[0]]:
                        brk[0] += not pg
                        brk[1] += 1
                    else:
                        fix[0] += pg
                        fix[1] += 1
            vacc = pct(pick_acc(VA, lambda r: duel(r, 0.5, 1.0, tag), ids=list(VA))) if all("sig" in r for r in VA.values()) else float("nan")
            arm, ch = {i: False for i in DEC if DBOF[i] == "absent"}, []
            for db in sorted({r["db"] for r in D.values()}):
                trn = [i for i in DEC if DBOF[i] != db]
                g = max(((t, w) for t in TG for w in WG),
                        key=lambda tw: (pct(pick_acc(D, lambda r: duel(r, tw[0], tw[1], tag), ids=trn)), -tw[0]))
                ch.append(g)
                arm.update(pick_acc(D, lambda r: duel(r, g[0], g[1], tag), ids=[i for i in DEC if DBOF[i] == db]))
            P(f"  {nm:7s} prior_{tag}: prior picks right side in {agree}/{tot} top-2 split duels; "
              f"[validator-wrong duels: prior right {fix[0]}/{fix[1]}; validator-right duels: prior flips {brk[0]}/{brk[1]}] "
              f"LODO duel {cmp(base, arm)}  (t,w) {Counter(ch).most_common(3)}")
    log.close()


if __name__ == "__main__":
    main()
