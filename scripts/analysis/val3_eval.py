"""VAL3 (runs/phase1/val3/PREREG.md): the data-v3 validators (arm A seeds 1 and 2, arm B seed 1) vs VALHARD, System 1
only (R1's candidates, ONE validator picks), on DECISION (796 q, 8 DBs, original BIRD-train gold). DECISION ONLY:
RESERVED rows are dropped at load and never scored here (blind; a later, separate claim).

  .venv/bin/python scripts/analysis/val3_eval.py                          # after the boxes: runs/phase1/val3/<run>/val3/
  .venv/bin/python scripts/analysis/val3_eval.py --json runs/phase1/val3/eval.json
  .venv/bin/python scripts/analysis/val3_eval.py --dry-run                # code path: VALHARD's own files stand in

Box outputs are label-free (scripts/chains/val3_stage.py); labels are joined here, per question and candidate position,
from the labelled pools (data/valhard/r1/heldout_pool.jsonl; runs/phase1/repair/repair/k64_repair_pool.jsonl), and the
SQL at every position must match. Pick = max rr over live candidates (runs and nonnull; all if none live), ties to the
earlier candidate; uncovered DECISION questions count wrong in every arm. k32 files: src in {r1, both} (as
valhard_eval.py / pick_rules_eval.py; the R1 pool has no other src).

PRIMARY (per arm; the averaged model's pick on R1's k=32 pool vs VALHARD on the same pool, 59.05):
  single-seed arm (B): delta >= +1.0, one-sided exact McNemar p < 0.05, exact DB sign-flip p < 0.25 -> PASS;
                       delta < +0.3 -> KILL; otherwise NO PASS.
  two-seed arm (A):    the same three bars on the SEED-MEAN (per-question hit = mean over the two seeds): delta =
                       mean of the seed deltas; question-level test = exact one-sided paired sign-flip on the
                       per-question differences (values in {+-0.5, +-1}; with one seed it is exactly McNemar);
                       DB test = exact sign-flip over the 8 per-DB sums (two-sided). PASS also needs EVERY seed's
                       delta >= +0.3. KILL if the mean delta < +0.3.
Measurement: validator seed gap (A s1 vs A s2): accuracy gap, discordant questions, two-sided McNemar.
SECONDARY (info, no decision): k64+repair pool picks vs VALHARD on that pool (59.55); close-pair top-2 accuracy
(top-2 live result classes, one right, margin < 1: own gate, and on VALHARD's gated pairs); hard-list accuracy
("greedy wrong, right lower": first live candidate in generator order wrong, a right live candidate exists).
Info: val history, averaged steps, schedule-shortened flag, SKIPPED_FOR_MEMORY from the pulled run dirs.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402

DECISION = ROOT / "data/sys1/heldout_v2/decision.json"
RUNS_DIR = ROOT / "runs/phase1/val3"
RUNS = {"A_s1": "A", "A_s2": "A", "B_s1": "B"}
LABELS = {"k32": ROOT / "data/valhard/r1/heldout_pool.jsonl",
          "k64r": ROOT / "runs/phase1/repair/repair/k64_repair_pool.jsonl"}
BASE = {"k32": ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl",
        "k64r": ROOT / "runs/phase1/repair/repair/k64_repair_pool_by_valhard.jsonl"}
BASE_ACC = {"k32": 59.05, "k64r": 59.55}  # VALHARD, as frozen in runs/phase1/pickrules/PREREG.md
PASS_DELTA, PASS_MCN, PASS_DBP, KILL_DELTA, SEED_FLOOR = 1.0, 0.05, 0.25, 0.3, 0.3
GATE = 1.0
FORBIDDEN = ("bird_dev", "gate.json", "sealed", "/score_", "reserved")


def guard(p: Path) -> Path:
    if any(f in str(p) for f in FORBIDDEN):
        raise SystemExit(f"refusing to read {p}: BIRD dev / gate / sealed / RESERVED files are off limits here")
    return p


def load(path: Path, ids: set) -> dict:
    out = {}
    for line in open(guard(path)):
        r = json.loads(line)
        if r["id"] not in ids:
            continue  # never look at non-DECISION rows
        if any("src" in c for c in r["cands"]) and all(c.get("src") in ("r1", "both", None) for c in r["cands"]):
            r["cands"] = [c for c in r["cands"] if c.get("src") in ("r1", "both")]
        out[r["id"]] = r
    return out


def join_labels(scored: dict, labelled: dict) -> dict:
    """Copy ok from the labelled pool onto the label-free scored rows (same question, same candidate positions)."""
    out = {}
    for i, r in scored.items():
        lab = labelled.get(i)
        if lab is None or len(lab["cands"]) != len(r["cands"]):
            raise SystemExit(f"label join: {i} missing or candidate count differs")
        for c, lc in zip(r["cands"], lab["cands"]):
            if (c.get("sql") or "") != (lc.get("sql") or ""):
                raise SystemExit(f"label join: {i} SQL differs at a candidate position")
        out[i] = {**r, "cands": [{**c, "ok": bool(lc.get("ok"))} for c, lc in zip(r["cands"], lab["cands"])]}
    return out


def live(cands):
    L = [c for c in cands if c.get("runs") and c.get("nonnull")]
    return L or cands


def pick(cands):
    u = [c for c in cands if c.get("sql")]
    if not u:
        return None
    return max(live(u), key=lambda c: c.get("rr", -1e9))


def classes(cands):
    """[(best rr, ok, sig)] over live result classes, best first."""
    d = {}
    for c in live([c for c in cands if c.get("sql")]):
        s = c.get("sig") or c["sql"]
        if s not in d or c.get("rr", -1e9) > d[s][0]:
            d[s] = (c.get("rr", -1e9), bool(c.get("ok")), s)
    return sorted(d.values(), key=lambda x: -x[0])


def hits(S: dict, ids: list) -> dict:
    return {i: bool(i in S and (p := pick(S[i]["cands"])) and p.get("ok")) for i in ids}


def mcnemar_one_sided(w: int, l_: int) -> float:
    m = w + l_
    return 1.0 if m == 0 else sum(math.comb(m, j) for j in range(w, m + 1)) / 2 ** m


def paired_signflip_one_sided(d: list) -> float:
    """Exact P[sum of randomly signed |d_i| >= observed sum], d_i in multiples of 0.5 (seed-mean hit differences)."""
    k = [int(round(2 * x)) for x in d if abs(x) > 1e-12]
    obs = sum(k)
    dist = {0: 1}
    for v in k:
        nd = collections.defaultdict(int)
        for s, c in dist.items():
            nd[s + v] += c
            nd[s - v] += c
        dist = nd
    tot = 2 ** len(k)
    return sum(c for s, c in dist.items() if s >= obs) / tot


def db_signflip(d: dict, db: dict, alpha: float = 0.05) -> dict:
    """Exact two-sided DB-level sign flip on real-valued per-question differences (luthor.stats.cluster_signflip for
    non-binary hits), with its inverted CI (points)."""
    keys = sorted(set(db.values()))
    D = np.array([sum(v for i, v in d.items() if db[i] == g) for g in keys], float)
    n = np.array([sum(1 for i in d if db[i] == g) for g in keys], float)
    S = np.array(list(itertools.product((-1.0, 1.0), repeat=len(D))))
    p = float(np.mean(np.abs(S @ D) >= abs(D.sum()) - 1e-9))
    est = D.sum() / n.sum()

    def pval(d0):
        x = D - n * d0
        return np.mean(np.abs(S @ x) >= abs(x.sum()) - 1e-9)
    grid = np.arange(-1.0, 1.0 + 1e-4 / 2, 1e-4)
    i0 = int(np.searchsorted(grid, est))
    inside = []
    for rg in (range(i0, len(grid)), range(i0 - 1, -1, -1)):
        for j in rg:
            if pval(grid[j]) >= alpha:
                inside.append(grid[j])
            else:
                break
    lo, hi = (min(inside), max(inside)) if inside else (est, est)
    return {"delta": 100 * est, "p": p, "ci": (100 * lo, 100 * hi), "per_db": dict(zip(keys, D.tolist()))}


def compare(base: dict, new_hits: list, ids: list, db: dict) -> dict:
    """new_hits: one hit dict per seed. Seed-mean statistics (with one seed: McNemar + cluster_signflip)."""
    n = len(ids)
    deltas = [100 * (sum(h.values()) - sum(base.values())) / n for h in new_hits]
    d = {i: sum(h[i] for h in new_hits) / len(new_hits) - base[i] for i in ids}
    out = {"seeds": len(new_hits), "acc": [round(100 * sum(h.values()) / n, 2) for h in new_hits],
           "base": round(100 * sum(base.values()) / n, 2), "seed_deltas": [round(x, 2) for x in deltas],
           "delta": round(sum(deltas) / len(deltas), 2)}
    if len(new_hits) == 1:
        h = new_hits[0]
        c, q = cluster_signflip(base, h, db), question_signflip(base, h)
        out.update(wins=q["wins"], losses=q["losses"], q_p_one_sided=mcnemar_one_sided(q["wins"], q["losses"]),
                   mcnemar_2sided_p=q["p"], db_p=c["p"], db_ci=[round(float(x), 2) for x in c["ci"]],
                   per_db={k: int(v) for k, v in c["per_db"].items()})
    else:
        c = db_signflip(d, db)
        out.update(wins=round(sum(v for v in d.values() if v > 0), 1), losses=round(-sum(v for v in d.values() if v < 0), 1),
                   q_p_one_sided=paired_signflip_one_sided(list(d.values())), db_p=c["p"],
                   db_ci=[round(float(x), 2) for x in c["ci"]], per_db={k: round(v, 1) for k, v in c["per_db"].items()})
    out["q_p_one_sided"] = round(out["q_p_one_sided"], 4)
    out["db_p"] = round(out["db_p"], 4)
    return out


def verdict(c: dict) -> str:
    ok = (c["delta"] >= PASS_DELTA and c["q_p_one_sided"] < PASS_MCN and c["db_p"] < PASS_DBP
          and min(c["seed_deltas"]) >= SEED_FLOOR)
    return "PASS" if ok else "KILL" if c["delta"] < KILL_DELTA else "NO PASS"


def close_pairs(S: dict, ids: list, ref: dict | None = None) -> dict:
    """Own gate: questions whose top-2 live classes (margin < GATE, in this validator's rr) have exactly one right;
    accuracy = top-1 right. Matched (ref given): VALHARD's gated one-right pairs; does S rank the right class above?"""
    k = r = 0
    for i in ids:
        if i not in S:
            continue
        cs = classes(S[i]["cands"])
        if len(cs) >= 2 and cs[0][0] - cs[1][0] < GATE and cs[0][1] != cs[1][1]:
            k += 1
            r += cs[0][1]
    out = {"own_n": k, "own_acc": round(r / max(k, 1), 4)}
    if ref is not None:
        m = mr = 0
        for i in ids:
            if i not in S or i not in ref:
                continue
            cr = classes(ref[i]["cands"])
            if len(cr) < 2 or cr[0][0] - cr[1][0] >= GATE or cr[0][1] == cr[1][1]:
                continue
            right, wrong = (cr[0][2], cr[1][2]) if cr[0][1] else (cr[1][2], cr[0][2])
            mine = {s: v for v, _, s in classes(S[i]["cands"])}
            if right in mine and wrong in mine:
                m += 1
                mr += mine[right] > mine[wrong]
        out.update(matched_n=m, matched_acc=round(mr / max(m, 1), 4))
    return out


def hard_lists(S: dict, ids: list, labelled: dict) -> list:
    """'greedy wrong, right lower': first live candidate in generator order wrong, some live candidate right
    (class structure from the labelled pool, which every validator file shares)."""
    out = []
    for i in ids:
        if i not in labelled:
            continue
        cs = sorted([c for c in labelled[i]["cands"] if c.get("sql")], key=lambda c: c["rank"])
        L = [c for c in cs if c.get("runs") and c.get("nonnull")]
        if L and not L[0].get("ok") and any(c.get("ok") for c in L):
            out.append(i)
    return out


def run_info(d: Path) -> dict:
    info = {}
    rr = next(iter(sorted(d.glob("val3/rr_*"))), None)
    if rr is not None:
        vh = rr / "val_history.json"
        if vh.exists():
            info["val_history"] = [(h["tag"], round(h["pick_acc"], 4), h.get("group_nll")) for h in json.load(open(vh))]
        if (rr / "avg_steps.json").exists():
            info["avg_steps"] = json.load(open(rr / "avg_steps.json"))
        if (rr / "VAL3_TRAINED").exists():
            info["stamp"] = (rr / "VAL3_TRAINED").read_text().strip()
    lg = next((p for p in (d / "logs/val3_train.log", d / "val3/val3_train.log") if p.exists()), None)
    if lg:
        txt = lg.read_text(errors="replace")
        info["schedule_shortened"] = "time limit" in txt
        m = re.search(r"^SKIPPED_FOR_MEMORY (\d+)", txt, re.M)
        info["skipped_for_memory"] = int(m.group(1)) if m else None
        m = re.search(r"^trained .*$", txt, re.M)
        info["trained"] = m.group(0) if m else None
    return info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--dry-run", action="store_true", help="VALHARD's own scored files stand in for every run (delta 0)")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    dec = json.load(open(DECISION))
    ids = list(dec["ids"])
    idset = set(ids)
    tr = json.load(open(ROOT / "data/raw/bird_train/train.json"))
    db = {i: tr[int(i.rsplit("_", 1)[1])]["db_id"] for i in ids}
    assert collections.Counter(db.values()) == collections.Counter(dec["database_counts"])
    n = len(ids)
    LAB = {p: load(f, idset) for p, f in LABELS.items()}
    B = {p: load(f, idset) for p, f in BASE.items()}
    out = {"n": n, "dry_run": a.dry_run, "runs": {}, "controls": {}}
    if a.dry_run:
        print("DRY RUN: VALHARD's scored files stand in for every run (labels stripped and re-joined; every delta 0)")
    S = {}
    for run in RUNS:
        for p in ("k32", "k64r"):
            f = BASE[p] if a.dry_run else a.runs_dir / run / "val3" / f"{p}_by_{run}.jsonl"
            if not f.exists():
                raise SystemExit(f"missing {f}")
            raw = load(f, idset)
            if a.dry_run:  # strip labels as the box files are, so the join is exercised
                raw = {i: {**r, "cands": [{k: v for k, v in c.items() if k != "ok"} for c in r["cands"]]} for i, r in raw.items()}
            S[(run, p)] = join_labels(raw, LAB[p])
    # controls: candidate SQL per question identical to VALHARD's file; VALHARD's frozen accuracies reproduce
    for (run, p), sc in S.items():
        out["controls"][f"{run}/{p}_sql_mismatch_vs_valhard"] = sum(
            1 for i in ids if (i in sc) != (i in B[p]) or
            (i in sc and sorted(c.get("sql") or "" for c in sc[i]["cands"]) != sorted(c.get("sql") or "" for c in B[p][i]["cands"])))
    H = {(r, p): hits(sc, ids) for (r, p), sc in S.items()}
    HB = {p: hits(B[p], ids) for p in B}
    for p in B:
        acc = round(100 * sum(HB[p].values()) / n, 2)
        out["controls"][f"valhard_{p}_acc"] = acc
        out["controls"][f"valhard_{p}_reproduces_{BASE_ACC[p]}"] = abs(acc - BASE_ACC[p]) < 0.006
    print(f"DECISION n={n}, {len(set(db.values()))} DBs; covered " +
          ", ".join(f"{r}/{p} {len(sc)}" for (r, p), sc in S.items()))
    print("controls (mismatch must be 0, reproduces must be True): " + json.dumps(out["controls"]))

    # ---- PRIMARY: per arm on k32 ----------------------------------------------------------------------------------------
    print("\nPRIMARY (R1 k=32 pool; averaged model's pick vs VALHARD 59.05; DECISION)")
    arms = collections.defaultdict(list)
    for run, arm in RUNS.items():
        arms[arm].append(run)
    out["primary"] = {}
    for arm, runs in sorted(arms.items()):
        c = compare(HB["k32"], [H[(r, "k32")] for r in runs], ids, db)
        c["runs"] = runs
        c["verdict"] = verdict(c)
        out["primary"][arm] = c
        print(f"  arm {arm} ({', '.join(runs)}): acc {c['acc']} vs {c['base']}  seed deltas {c['seed_deltas']}  "
              f"delta {c['delta']:+.2f}  +{c['wins']}/-{c['losses']}  q p(1-sided) {c['q_p_one_sided']:.4f}  "
              f"DB p {c['db_p']:.4f}  CI {c['db_ci']}  -> {c['verdict']}")
        print(f"      per-DB {c['per_db']}")
        for r in runs:  # per-seed cells (info for a two-seed arm)
            cs = compare(HB["k32"], [H[(r, "k32")]], ids, db)
            out["runs"].setdefault(r, {})["k32"] = cs
            if len(runs) > 1:
                print(f"      {r}: delta {cs['delta']:+.2f} +{cs['wins']}/-{cs['losses']} McN1 p {cs['q_p_one_sided']:.4f} "
                      f"DB p {cs['db_p']:.4f}")
    print(f"  (PASS: delta >= +{PASS_DELTA}, one-sided q p < {PASS_MCN}, DB p < {PASS_DBP}, every seed >= +{SEED_FLOOR}; "
          f"KILL: delta < +{KILL_DELTA}; two arms tested, no multiplicity correction: info under Bonferroni q p < 0.025: "
          + ", ".join(f"{k} {v['q_p_one_sided'] < 0.025}" for k, v in out["primary"].items()) + ")")

    # ---- measurement: validator seed gap ---------------------------------------------------------------------------------
    if len(arms.get("A", [])) == 2:
        r1_, r2_ = arms["A"]
        out["seed_gap"] = {}
        for p in ("k32", "k64r"):
            q = question_signflip(H[(r1_, p)], H[(r2_, p)])
            g = {"acc": [round(100 * sum(H[(r, p)].values()) / n, 2) for r in (r1_, r2_)], "gap": round(q["delta"], 2),
                 "discordant": q["wins"] + q["losses"], "s2_only": q["wins"], "s1_only": q["losses"],
                 "mcnemar_2sided_p": round(q["p"], 4),
                 "pick_differs": sum(1 for i in ids if i in S[(r1_, p)] and
                                     pick(S[(r1_, p)][i]["cands"]) is not None and pick(S[(r2_, p)][i]["cands"]) is not None
                                     and pick(S[(r1_, p)][i]["cands"])["sql"] != pick(S[(r2_, p)][i]["cands"])["sql"])}
            out["seed_gap"][p] = g
            print(f"\nSEED GAP arm A ({p}): {r1_} {g['acc'][0]} vs {r2_} {g['acc'][1]}  gap {g['gap']:+.2f}  "
                  f"discordant {g['discordant']} (+{g['s2_only']}/-{g['s1_only']})  McNemar p {g['mcnemar_2sided_p']:.3f}  "
                  f"different pick in {g['pick_differs']} questions")

    # ---- SECONDARY ---------------------------------------------------------------------------------------------------------
    print("\nSECONDARY (info, no decision)")
    out["secondary"] = {"k64r": {}, "close_pairs": {}, "hard_lists": {}}
    for arm, runs in sorted(arms.items()):
        c = compare(HB["k64r"], [H[(r, "k64r")] for r in runs], ids, db)
        out["secondary"]["k64r"][arm] = c
        print(f"  k64+repair arm {arm}: acc {c['acc']} vs VALHARD {c['base']}  delta {c['delta']:+.2f}  "
              f"+{c['wins']}/-{c['losses']}  q p {c['q_p_one_sided']:.4f}  DB p {c['db_p']:.4f}")
        c2 = compare(HB["k32"], [H[(r, "k64r")] for r in runs], ids, db)
        out["secondary"]["k64r"][f"{arm}_vs_valhard_k32"] = c2
        print(f"      (vs VALHARD k32 59.05: delta {c2['delta']:+.2f}, q p {c2['q_p_one_sided']:.4f}, DB p {c2['db_p']:.4f})")
    cpv = close_pairs(B["k32"], ids)
    out["secondary"]["close_pairs"]["VALHARD"] = cpv
    print(f"  close pairs (k32, top-2 classes one right, margin < {GATE}): VALHARD own {cpv['own_acc']:.3f} (n={cpv['own_n']})")
    for run in RUNS:
        cp = close_pairs(S[(run, "k32")], ids, ref=B["k32"])
        out["secondary"]["close_pairs"][run] = cp
        print(f"    {run}: own {cp['own_acc']:.3f} (n={cp['own_n']}); on VALHARD's gated pairs {cp['matched_acc']:.3f} "
              f"(n={cp['matched_n']}; VALHARD {cpv['own_acc']:.3f})")
    hl = hard_lists(B["k32"], ids, LAB["k32"])
    hv = sum(HB["k32"][i] for i in hl) / max(len(hl), 1)
    out["secondary"]["hard_lists"] = {"n": len(hl), "VALHARD": round(hv, 4)}
    print(f"  hard lists (greedy wrong, right lower; k32): n={len(hl)}  VALHARD {100 * hv:.1f}%  " + "  ".join(
        f"{r} {100 * sum(H[(r, 'k32')][i] for i in hl) / max(len(hl), 1):.1f}%" for r in RUNS))
    for r in RUNS:
        out["secondary"]["hard_lists"][r] = round(sum(H[(r, "k32")][i] for i in hl) / max(len(hl), 1), 4)

    # ---- info: training runs ---------------------------------------------------------------------------------------------
    if not a.dry_run:
        print("\nINFO (training)")
        for run in RUNS:
            inf = run_info(a.runs_dir / run)
            out["runs"].setdefault(run, {})["train"] = inf
            print(f"  {run}: {json.dumps(inf)[:600]}")
        data = {re.sub(r"^.*?(train=\S+ val=\S+).*$", r"\1", v["train"].get("stamp", "")) for v in out["runs"].values()}
        out["controls"]["same_train_val_data_all_runs"] = len(data) == 1 and "" not in data
        out["controls"]["stamp_matches_run"] = all(
            f"arm={RUNS[r]} seed={r.split('_s')[1]} " in v["train"].get("stamp", "") for r, v in out["runs"].items())
        out["controls"]["any_schedule_shortened"] = any(v["train"].get("schedule_shortened") for v in out["runs"].values())
        print(f"  same train/val data in every run: {out['controls']['same_train_val_data_all_runs']} (must be True); stamps match "
              f"arm/seed: {out['controls']['stamp_matches_run']} (must be True); schedule shortened by the time limit in any run: {out['controls']['any_schedule_shortened']}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=str))
        print("wrote", a.json)
    return out


if __name__ == "__main__":
    main()
