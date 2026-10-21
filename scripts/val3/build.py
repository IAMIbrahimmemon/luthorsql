"""VAL3 data v3, step 3 (Mac, CPU): validator training + checkpoint-selection lists from the box's R1 k=32 lists.

  .venv/bin/python scripts/val3/build.py [--box runs/phase1/val3_lists/val3] [--out data/val3/lists] [--workers 6]
  .venv/bin/python scripts/val3/difficulty.py                  # Mac gate before any training (scripts/val3/difficulty.py)

Inputs: the old R1 lists (VALHARD: data/valhard/r1/train_lists.jsonl, mixed with hard negatives; their raw pools
runs/phase1/p1w/p1w/{r1,r2}/cands_train.jsonl), the box outputs of scripts/chains/val3_lists_box.sh
(<box>/cands_{bird_rr,bird_big,synsql}_{r1,r2}.jsonl; r1 = k=32 distinct, r2 = R2 greedy) and data/val3/q/meta.jsonl
(scripts/val3/sources.py: src, kind a = R1 trained on the question / b = never trained, split).

Lists (one live representative per result class, cap 12, near-miss mutants, then VALHARD's hard negatives):
  old      VALHARD's 3,830 train lists unchanged in content (in-sample BIRD-train).
  bird     new BIRD-train lists: scripts/val2/build_lists.py --pool (BIRD-Verified relabel, gold-noise drop),
           scripts/valhard/mine.py, scripts/valhard/mix.py, exactly VALHARD's code and defaults, into <work>/bird.
  synsql   new SynSQL lists: the same steps with the same functions (bl.build_list, mutate, mine.mine_list,
           mine.keep_edits, mix.mix_one; VALHARD's defaults) on the SynSQL DBs (no relabel: no Verified set exists).
  Train with `rerank_sq2.py train --loss group` (pair members are extra members of a result class).

Label-noise mask (rule M, train lists only; val lists are never masked):
  g1 = result signature of R1's rank-0 candidate in its raw k=32 list, g2 = that of R2's rank-0 candidate (R2 k=1 on
  the box for new questions; p1w R2 lists for old ones). The list's greedy class is MASKED when ALL hold:
    (1) g1 is live (runs and returns a non-NULL value) and g2 == g1 (R1 and R2 return the same result);
    (2) that class is labelled wrong in the list (the list's gold, after BIRD-Verified relabelling, disagrees);
    (3) the question has no BIRD-Verified entry (neither corrected nor confirmed): every new SynSQL question and
        every BIRD question outside BIRD-Verified;
    (4) the question's gold was not corrected (implied by 3).
  Masked = every candidate whose sig == g1 is removed from the list (it is neither positive nor negative), the list
  loses any x3 (weight rule below), and the list is dropped if it then lacks a right or a wrong candidate.
Weights (list "weight", normalised by its mean in the trainer):
  hard     = R1's rank-0 class is not right under the list's labels (dead or missing counts as not right).
  oos      = the question is out-of-sample for R1 (meta kind b: BIRD never trained, every SynSQL question).
  verified = BIRD question with a BIRD-Verified entry.
  w_hard   = 3 if hard and not masked and (verified or oos), else 1   (x3 dropped on unverified in-sample hard lists).
  in-sample lists (old + bird kind a): w = clip(w_hard * r_bin, 1/3, 3) (a masked list: r_bin = 1), r_bin = T(bin) / S(bin) where bin =
           (position of the first right result class in R1's rank order: 0 | 1 | 2-3 | 4+) x (#live result classes:
           1 | 2-4 | 5-9 | 10+) (04_list_difficulty.py's bins), S = the in-sample train lists' bin shares and T = the
           OUT-OF-SAMPLE target: the bin shares of the new BIRD kind-b train lists (--target bird_oos; 'oos' adds
           SynSQL), add-alpha smoothed, r clipped to [1/3, 3]. Nothing here reads DECISION.
  out-of-sample lists: w = w_hard (SynSQL times --synsql-weight).
Val (checkpoint selection, --val-max lists): out-of-sample questions on DBs no train list is on: BIRD kind-b lists on
  the 4 pinned val DBs (citeseer, disney, professional_basketball, trains) first, then SynSQL lists on the SynSQL val
  DBs; one representative per class, cap 12, no mutants and no hard negatives, weight 1, never masked.
Writes <out>/{train_lists,val_lists}.jsonl, <out>/build_stats.json, <out>/masked.jsonl (ids + sigs, for audit).
Extra list field "v3" = {src, kind, oos, verified, hard, masked, bin, r_bin, w_hard} (analysis only; the trainer reads
id/db_id/question/context/cands/weight). Prints counts only.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/val2"))
sys.path.insert(0, str(ROOT / "scripts/valhard"))
import build_lists as bl  # noqa: E402
import edits as ed  # noqa: E402
import execq  # noqa: E402
import mine  # noqa: E402
import mix  # noqa: E402
import mutate  # noqa: E402
import sqldiff  # noqa: E402

PY = sys.executable
VAL_DBS = set(bl.VAL_DBS)
W_MIN, W_MAX = 1.0 / 3.0, 3.0
SETS = ("bird_rr", "bird_big", "synsql")
MIX_DEFAULTS = dict(max_edits=3, max_mined_add=2, max_pair_members=2, pair_members=True, drop_mutations=["limit"],
                    use_hard=True)  # scripts/valhard/mix.py defaults (VALHARD)


# ---- pure rules (tests/test_val3.py) -----------------------------------------------------------------------------------

def live(c: dict) -> bool:
    return bool(c.get("runs")) and bool(c.get("nonnull"))


def greedy(raw: dict | None) -> dict | None:
    """The rank-0 candidate of a raw generator list (None when missing or SQL-less)."""
    if not raw:
        return None
    g = min((c for c in raw.get("cands", []) if c.get("sql")), key=lambda c: c["rank"], default=None)
    return g if g is not None and g["rank"] == 0 else None


def list_labels(lst: dict) -> dict[str, bool]:
    """sig -> ok over the list's candidates (a sig has one label per list)."""
    return {c["sig"]: bool(c.get("ok")) for c in lst["cands"] if c.get("sig")}


def is_hard(lst: dict, raw: dict | None) -> bool:
    """R1's rank-0 class is not right under the list's labels (dead, missing or wrong)."""
    g = greedy(raw)
    if g is None or not live(g) or not g.get("sig"):
        return True
    return not list_labels(lst).get(g["sig"], False)


def mask_sig(lst: dict, raw1: dict | None, raw2: dict | None, verified: bool) -> str | None:
    """Rule M: the greedy sig to mask, or None."""
    if verified or lst.get("gold_corrected"):
        return None
    g1, g2 = greedy(raw1), greedy(raw2)
    if g1 is None or g2 is None or not live(g1) or not live(g2):
        return None
    s = g1.get("sig")
    if not s or g2.get("sig") != s:
        return None
    lab = list_labels(lst)
    if s not in lab or lab[s]:
        return None  # not in the list (nothing to mask) or labelled right (gold agrees)
    return s


def apply_mask(lst: dict, sig: str) -> dict | None:
    """The list without the masked class; None if it no longer has a right and a wrong candidate."""
    cands = [c for c in lst["cands"] if c.get("sig") != sig]
    oks = [bool(c.get("ok")) for c in cands]
    if not any(oks) or all(oks):
        return None
    return dict(lst, cands=cands)


def class_order(lst: dict, raw: dict | None) -> list[bool]:
    """Live result classes of R1's raw list in rank order, labelled by the list (a raw class the list does not hold:
    its gen-time label unless the gold was corrected, then wrong). Falls back to the list's own real classes."""
    lab = list_labels(lst)
    if raw:
        seen, out = set(), []
        for c in sorted((c for c in raw["cands"] if c.get("sql")), key=lambda c: c["rank"]):
            s = c.get("sig")
            if not live(c) or not s or s in seen:
                continue
            seen.add(s)
            out.append(lab[s] if s in lab else (False if lst.get("gold_corrected") else bool(c.get("ok"))))
        if out:
            return out
    reals = sorted((c for c in lst["cands"] if not c.get("mutant") and c.get("prov", "base") == "base"),
                   key=lambda c: c["rank"])
    return [bool(c.get("ok")) for c in reals]


def diff_bin(classes: list[bool]) -> tuple[str, str]:
    fr = next((i for i, ok in enumerate(classes) if ok), None)
    frb = "none" if fr is None else "0" if fr == 0 else "1" if fr == 1 else "2-3" if fr < 4 else "4+"
    n = len(classes)
    nb = "1" if n <= 1 else "2-4" if n <= 4 else "5-9" if n <= 9 else "10+"
    return frb, nb


def bin_ratios(source: list, target: list, alpha: float = 1.0, lo: float = W_MIN, hi: float = W_MAX) -> dict:
    """r(bin) = target share / source share (add-alpha smoothed over the bins either side has), clipped to [lo, hi]."""
    cs, ct = collections.Counter(source), collections.Counter(target)
    bins = sorted(set(cs) | set(ct))
    ns, nt = len(source) + alpha * len(bins), len(target) + alpha * len(bins)
    return {b: min(hi, max(lo, ((ct[b] + alpha) / nt) / ((cs[b] + alpha) / ns))) for b in bins}


def list_weight(hard: bool, verified: bool, oos: bool, masked: bool, r_bin: float | None) -> tuple[float, float]:
    """(w_hard, final weight) by the rule in the module doc; r_bin None for out-of-sample lists."""
    w_hard = 3.0 if hard and not masked and (verified or oos) else 1.0
    if r_bin is None:
        return w_hard, w_hard
    return w_hard, min(W_MAX, max(W_MIN, w_hard * r_bin))


def ess(ws: list[float]) -> float:
    return sum(ws) ** 2 / sum(w * w for w in ws) if ws else 0.0


def tv(a: dict, b: dict) -> float:
    """Total-variation distance of two {bin: weight} mixes."""
    sa, sb = sum(a.values()) or 1.0, sum(b.values()) or 1.0
    return 0.5 * sum(abs(a.get(k, 0) / sa - b.get(k, 0) / sb) for k in set(a) | set(b))


def quantiles(xs: list[float], qs=(0.0, 0.1, 0.5, 0.9, 1.0)) -> list[float]:
    ys = sorted(xs)
    return [ys[min(len(ys) - 1, int(q * (len(ys) - 1) + 0.5))] for q in qs] if ys else []


def strip_for_val(lst: dict) -> dict | None:
    """Val list: real result classes only (no mutants, no mined / pair / edit entries), weight 1."""
    cands = [c for c in lst["cands"] if not c.get("mutant") and c.get("prov", "base") in ("base", "g")]
    oks = [bool(c.get("ok")) for c in cands]
    if not any(oks) or all(oks):
        return None
    return dict(lst, cands=cands, weight=1.0)


# ---- generic list building (non-BIRD pools: SynSQL) ------------------------------------------------------------------

def build_generic(pool: dict[str, dict], paths: dict, split_of, seed: int, workers: int,
                  st: collections.Counter) -> tuple[list, list, dict]:
    """bl.build_list + near-miss mutants (build_lists step 2) and VALHARD's mined pairs + synthetic edits (mine.py) for
    a pool whose labels are final (no relabel). Returns (train base lists, val base lists, {id: hard record})."""
    built, mut_jobs, cols = {}, collections.defaultdict(list), {}
    for i, rec in pool.items():
        out, info = bl.build_list(rec, [("r1", [dict(c) for c in rec["cands"]])], seed)
        out["gold_corrected"] = False
        right = [c for c in out["cands"] if c["ok"]]
        if not right:
            st["synsql_no_live_right"] += 1
            continue
        built[i] = (out, info)
        if split_of(i) == "val":
            continue  # val lists carry no mutants
        db = rec["db_id"]
        if db not in cols:
            cols[db] = bl.db_columns(paths[db])
        sqls = [right[0]["sql"]] + [m["sql"] for m in info["right_members"] if m["sql"] != right[0]["sql"]]
        muts = []
        for s in sqls:
            muts = mutate.mutants(s, cols[db], seed=f"{seed}:{i}")
            if muts:
                break
        if muts:
            out["_muts"] = muts
            mut_jobs[db].append((i, rec["gold_sql"], [m for _, m in muts]))
    print(f"synsql: {len(built)} lists; executing {sum(len(s) for v in mut_jobs.values() for _, _, s in v)} mutants",
          flush=True)
    mres = execq.run_tasks(mut_jobs, paths, bl.GOLD_SECONDS, bl.GEN_SECONDS, workers=workers)
    train, val = [], []
    for i, (out, _info) in built.items():
        muts = out.pop("_muts", None)
        if muts:
            status, _, r = mres[i]
            if status == "ok":
                out["cands"] += bl.pick_mutants([(k, s, x) for (k, s), x in zip(muts, r)],
                                                {c["sig"] for c in out["cands"]})
            else:
                st[f"synsql_mutant_gold_{status}"] += 1
        if not any(not c["ok"] for c in out["cands"]):
            st["synsql_drop_no_wrong_class"] += 1
            continue
        (val if split_of(i) == "val" else train).append(out)
    # mined pairs + synthetic edits (train lists only), as scripts/valhard/mine.py
    hard, jobs, schemas = {}, collections.defaultdict(list), {}
    for r in train:
        i, db = r["id"], r["db_id"]
        union = bl.union_cands([("r1", [dict(c) for c in pool[i]["cands"]])])
        rec = {"id": i, "db_id": db, "split": "train", "mined": mine.mine_list(r, union, {}), "edits": [],
               "edit_source": mine.edit_source(r, union)}
        hard[i] = rec
        if rec["edit_source"] is None:
            continue
        if db not in schemas:
            schemas[db] = ed.schema_of(paths[db])
        es = ed.edits(rec["edit_source"], schemas[db], seed=f"{seed}:{i}")
        if es:
            rec["_trials"] = es
            jobs[db].append((i, r["gold_sql"], [s for _, s in es]))
    print(f"synsql: executing {sum(len(s) for v in jobs.values() for _, _, s in v)} synthetic edits", flush=True)
    res = execq.run_tasks(jobs, paths, bl.GOLD_SECONDS, bl.GEN_SECONDS, workers=workers)
    by_id = {r["id"]: r for r in train}
    for i, rec in hard.items():
        trials = rec.pop("_trials", None)
        if not trials:
            continue
        status, _, rr = res[i]
        if status != "ok":
            continue
        kept, _ = mine.keep_edits([(k, s, x) for (k, s), x in zip(trials, rr)],
                                  {c["sig"] for c in by_id[i]["cands"] if c.get("sig")})
        for e in kept:
            kind, _, _ = sqldiff.classify(rec["edit_source"], e["sql"])
            e["diff_type"] = kind
            if sqldiff.is_hard(kind):
                e["bucket"] = sqldiff.BUCKET[kind]
        rec["edits"] = kept
    return train, val, hard


def mix_lists(rows: list[dict], hard: dict) -> list[dict]:
    a = SimpleNamespace(**MIX_DEFAULTS)
    out = []
    for r in rows:
        m, _ = mix.mix_one(r, hard.get(r["id"]), a)
        oks = [bool(c.get("ok")) for c in m["cands"]]
        if any(oks) and not all(oks):
            out.append(m)
    return out


# ---- data -------------------------------------------------------------------------------------------------------------

def jl(p: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, open(p))} if p.exists() else {}


def run(cmd: list[str], logf: Path) -> None:
    print("$ " + " ".join(map(str, cmd)), flush=True)
    with open(logf, "w") as f:
        rc = subprocess.call(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if rc:
        sys.exit(f"failed ({rc}): see {logf}")


def bird_new(a, pools1: dict, st: collections.Counter) -> tuple[list, list, dict]:
    """New BIRD lists through VALHARD's exact scripts. Returns (train mixed lists, val base lists, raw pool)."""
    w = a.work / "bird"
    pool = {i: r for s in ("bird_rr", "bird_big") for i, r in pools1.get(s, {}).items()}
    st["bird_pool_questions"] = len(pool)
    if not pool:
        return [], [], {}
    w.mkdir(parents=True, exist_ok=True)
    pf = w / "pool_r1.jsonl"
    with open(pf, "w") as f:
        for i in sorted(pool, key=bl.qindex):
            f.write(json.dumps(pool[i]) + "\n")
    if not (a.reuse and (w / "train_lists.jsonl").exists()):
        run([PY, "scripts/val2/build_lists.py", "--pool", f"r1={pf}", "--out", w / "base", "--workers", a.workers,
             "--seed", a.seed], w / "build_lists.log")
        run([PY, "scripts/valhard/mine.py", "--base", w / "base", "--pool", f"r1={pf}", "--out", w, "--workers",
             a.workers, "--seed", a.seed], w / "mine.log")
        run([PY, "scripts/valhard/mix.py", "--base", w / "base", "--hard", w / "hard.jsonl", "--out", w], w / "mix.log")
    train = list(jl(w / "train_lists.jsonl").values())
    val = list(jl(w / "base/val_lists.jsonl").values())
    return train, val, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, default=ROOT / "runs/phase1/val3_lists/val3")
    ap.add_argument("--meta", type=Path, default=ROOT / "data/val3/q/meta.jsonl")
    ap.add_argument("--old", type=Path, default=ROOT / "data/valhard/r1/train_lists.jsonl")
    ap.add_argument("--old-r1", type=Path, default=ROOT / "runs/phase1/p1w/p1w/r1/cands_train.jsonl")
    ap.add_argument("--old-r2", type=Path, default=ROOT / "runs/phase1/p1w/p1w/r2/cands_train.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data/val3/lists")
    ap.add_argument("--work", type=Path, default=ROOT / "data/val3/work")
    ap.add_argument("--synsql-db-root", type=Path, default=ROOT / "data/val3/dbs/synsql")
    ap.add_argument("--target", choices=["bird_oos", "oos"], default="bird_oos")
    ap.add_argument("--alpha", type=float, default=1.0, help="bin-share smoothing")
    ap.add_argument("--synsql-max", type=int, default=4000, help="SynSQL train lists kept (seeded sample; 0 = all)")
    ap.add_argument("--synsql-weight", type=float, default=1.0)
    ap.add_argument("--val-max", type=int, default=400)
    ap.add_argument("--no-mask", action="store_true", help="ablation: rule M off")
    ap.add_argument("--reuse", action="store_true", help="reuse <work>/bird outputs of an earlier run")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    st = collections.Counter()
    meta = jl(a.meta)
    held = set(json.load(open(ROOT / "data/sys1/dev_dbs_bird.json")))
    fx = set(bl.verified())
    pools1 = {s: {i: r for i, r in jl(a.box / f"cands_{s}_r1.jsonl").items() if not r.get("gold_error")} for s in SETS}
    pools2 = {s: jl(a.box / f"cands_{s}_r2.jsonl") for s in SETS}
    for s in SETS:
        st[f"box_{s}_r1"] = len(pools1[s])
        st[f"box_{s}_r2"] = len(pools2[s])
    raw2 = {i: r for p in pools2.values() for i, r in p.items()}
    for i in raw2:
        assert i in meta, i  # nothing on the box that sources.py did not stage

    # ---- new BIRD (VALHARD scripts) + SynSQL (same functions) -----------------------------------------------------
    bird_train, bird_val, bird_pool = bird_new(a, pools1, st)
    syn_pool = pools1["synsql"]
    paths = {p.stem: str(p) for p in a.synsql_db_root.glob("*/*.sqlite")}
    split_of = lambda i: meta[i]["split"]  # noqa: E731
    syn_train, syn_val, syn_hard = build_generic(syn_pool, paths, split_of, a.seed, a.workers, st) if syn_pool else ([], [], {})
    syn_train = mix_lists(syn_train, syn_hard)
    st["synsql_train_lists_all"] = len(syn_train)
    if a.synsql_max and len(syn_train) > a.synsql_max:
        syn_train = sorted(syn_train, key=lambda r: bl.unit(a.seed, "synkeep", r["id"]))[:a.synsql_max]

    # ---- assemble train with provenance -----------------------------------------------------------------------------
    old = list(jl(a.old).values())
    old_r1, old_r2 = jl(a.old_r1), jl(a.old_r2)
    rows = []  # (list, src, kind, raw1, raw2)
    for r in old:
        rows.append((r, "old", "a", old_r1.get(r["id"]), old_r2.get(r["id"])))
    for r in bird_train:
        m = meta.get(r["id"])
        assert m is not None and m["split"] == "train", r["id"]
        rows.append((r, "bird", m["kind"], bird_pool.get(r["id"]), raw2.get(r["id"])))
    for r in syn_train:
        rows.append((r, "synsql", "b", syn_pool.get(r["id"]), raw2.get(r["id"])))
    recs, masked = [], []
    for r, src, kind, raw1, rr2 in rows:
        assert r["db_id"] not in held, r["id"]
        verified = src != "synsql" and r["id"].startswith("bird_train_train_") and bl.qindex(r["id"]) in fx
        oos = kind == "b"
        hard = is_hard(r, raw1)
        sig = None if a.no_mask else mask_sig(r, raw1, rr2, verified)
        st[f"lists_{src}_{kind}"] += 1
        st[f"hard_{src}_{kind}"] += hard
        if rr2 is None:
            st[f"no_r2_{src}"] += 1
        if sig is not None:
            st[f"masked_{src}_{kind}"] += 1
            masked.append({"id": r["id"], "src": src, "kind": kind, "sig": sig, "hard": hard})
            r = apply_mask(r, sig)
            if r is None:
                st[f"masked_dropped_{src}_{kind}"] += 1
                continue
        b = diff_bin(class_order(r, raw1))
        recs.append({"lst": r, "src": src, "kind": kind, "oos": oos, "verified": verified, "hard": hard,
                     "masked": sig is not None, "bin": b})
    # ---- weights: x3 rule, then in-sample lists reweighted toward the out-of-sample bin mix ---------------------------
    ins = [x for x in recs if not x["oos"]]
    tgt = [x for x in recs if x["oos"] and (x["src"] == "bird" or a.target == "oos")]
    if len(tgt) < 200:
        st["target_fallback_all_oos"] = 1
        tgt = [x for x in recs if x["oos"]]
    ratios = bin_ratios([x["bin"] for x in ins], [x["bin"] for x in tgt], a.alpha) if tgt else {}
    for x in recs:
        rb = None if x["oos"] else (1.0 if x["masked"] else ratios.get(x["bin"], 1.0))
        w_hard, w = list_weight(x["hard"], x["verified"], x["oos"], x["masked"], rb)
        if x["src"] == "synsql":
            w *= a.synsql_weight
        x["lst"]["weight"] = round(w, 4)
        x["lst"]["v3"] = {"src": x["src"], "kind": x["kind"], "oos": x["oos"], "verified": x["verified"],
                          "hard": x["hard"], "masked": x["masked"], "bin": list(x["bin"]),
                          "r_bin": None if rb is None else round(rb, 4), "w_hard": w_hard}
    # ---- val: out-of-sample, DBs disjoint from train ------------------------------------------------------------------
    train_dbs = {x["lst"]["db_id"] for x in recs}
    vb = [v for v in (strip_for_val(r) for r in bird_val) if v]
    vs = [v for v in (strip_for_val(r) for r in syn_val) if v]
    for v in vb:
        assert meta[v["id"]]["kind"] == "b" and v["db_id"] in VAL_DBS, v["id"]
    vs = sorted(vs, key=lambda r: bl.unit(a.seed, "synval", r["id"]))
    val = vb[:a.val_max] + vs[:max(0, a.val_max - len(vb))]
    assert not ({r["db_id"] for r in val} & train_dbs), "val DBs overlap train DBs"
    for r in val:
        r["v3"] = {"src": "bird" if r["id"].startswith("bird_") else "synsql", "kind": "b", "oos": True}
    # ---- write --------------------------------------------------------------------------------------------------------
    a.out.mkdir(parents=True, exist_ok=True)
    order = {"old": 0, "bird": 1, "synsql": 2}
    train = [x["lst"] for x in sorted(recs, key=lambda x: (order[x["src"]], x["lst"]["id"]))]
    with open(a.out / "train_lists.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(a.out / "val_lists.jsonl", "w") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")
    with open(a.out / "masked.jsonl", "w") as f:
        for m in masked:
            f.write(json.dumps(m) + "\n")
    # ---- stats --------------------------------------------------------------------------------------------------------
    def group(xs):
        n = max(len(xs), 1)
        return {"lists": len(xs), "hard": sum(x["hard"] for x in xs),
                "x3": sum(x["lst"]["v3"]["w_hard"] == 3.0 for x in xs),
                "masked_kept": sum(x["masked"] for x in xs),
                "mean_weight": round(sum(x["lst"]["weight"] for x in xs) / n, 3),
                "weight_share": round(sum(x["lst"]["weight"] for x in xs) / max(sum(y["lst"]["weight"] for y in recs), 1e-9), 4),
                "mean_entries": round(sum(len(x["lst"]["cands"]) for x in xs) / n, 2),
                "dbs": len({x["lst"]["db_id"] for x in xs})}
    by = collections.defaultdict(list)
    for x in recs:
        by[f"{x['src']}_{x['kind']}"].append(x)
    w_ins = collections.Counter()
    u_ins = collections.Counter()
    for x in ins:  # the bin reweighting alone (weight / w_hard)
        w_ins[x["bin"]] += x["lst"]["weight"] / x["lst"]["v3"]["w_hard"]
        u_ins[x["bin"]] += 1
    t_mix = collections.Counter(x["bin"] for x in tgt)
    entries = sum(len(r["cands"]) for r in train)
    stats = {
        "args": {k: str(v) for k, v in vars(a).items()},
        "counts": dict(sorted(st.items())),
        "train": {"lists": len(train), "entries": entries, "dbs": len(train_dbs),
                  "hard_lists": sum(x["hard"] for x in recs),
                  "hard_lists_oos_or_verified": sum(x["hard"] and (x["oos"] or x["verified"]) for x in recs),
                  "x3_lists": sum(x["lst"]["v3"]["w_hard"] == 3.0 for x in recs),
                  "weight_quantiles": [round(q, 3) for q in quantiles([r["weight"] for r in train])],
                  "by_source": {k: group(v) for k, v in sorted(by.items())}},
        "val": {"lists": len(val), "bird": len(vb[:a.val_max]), "synsql": len(val) - len(vb[:a.val_max]),
                "dbs": len({r["db_id"] for r in val}), "mean_entries": round(sum(len(r["cands"]) for r in val) / max(len(val), 1), 2),
                "bird_available": len(vb), "synsql_available": len(vs)},
        "reweight": {"target": a.target, "target_lists": len(tgt), "ratios": {"|".join(k): round(v, 3) for k, v in ratios.items()},
                     "insample_ess": round(ess([x["lst"]["weight"] for x in ins]), 1), "insample_lists": len(ins),
                     "tv_insample_vs_target_before": round(tv(u_ins, t_mix), 4),
                     "tv_insample_vs_target_after": round(tv(w_ins, t_mix), 4)},
        "masked": len(masked),
        # VALHARD: 44.7k entries -> 1.92 h for 2 epochs on one H100 (runs/phase1/valhard/PREREG.md)
        "est_train_hours_per_epoch_h100": round(entries / 44700 * 0.96, 2),
        "minutes": round((time.time() - t0) / 60, 1)}
    (a.out / "build_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    main()
