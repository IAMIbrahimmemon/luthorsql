"""Mac side, CPU: evaluate the ff2 box run (scripts/chains/ff2_box.sh; prereg runs/phase1/ff2/PREREG.md; plan
docs/paper_notes/research/freeform_plan_2026-10-10.md §3.3-3.4). DECISION only (796 ids; missing = wrong). The held-out
union file (scored/heldout_by_new.jsonl) also holds RESERVED rows: they are filtered out by id before anything is
computed, and nothing is printed or written about them.

  uv run python scripts/analysis/ff2_eval.py [--ff1 runs/phase1/ff1/ff1] [--ff2 runs/phase1/ff2/ff2]
                                             [--out runs/phase1/ff2/ff2_eval.json]

Scores per candidate: r1, r2 (phase-1 validators, <ff1>/scored/<pool>_by_r{1,2}.jsonl) and new (the mixed-list
validator, <ff2>/scored/<pool>_by_new.jsonl), joined on (id, SQL); per-question z over live candidates. Score variants:
r1, r2, new, sumz (z r1 + z r2: phase 1's deployed combo), sumz3 (z r1 + z r2 + z new), new_r1 (z new + z r1).
Rules (as ff1_eval.py): A max score over live candidates; B z + beta * log1p(ff_sig_n), beta LODO; C ff vote; D grammar
only; E B with the ff-vote fallback (beta, tau LODO). Pools: grammar, +omni, +qwen, +both, +mut, +both+mut.

PRIMARY (prereg): +both+mut, rule A, score `new` vs phase-1 rule B on sumz (LODO, same pool; 70.98 in ff1_eval.json):
delta in points, exact DB sign-flip p (8 DECISION DBs) and McNemar; pass = delta >= +1.0 and p < 0.05, kill = A_new
below 70.98, else hold. Also vs phase-1 rule A sumz on the same pool (64.07; 63.94 on +both).
Calibration (§3.3-3): pick accuracy on "exactly one right" candidate pairs by source pair (g-g, g-ff, ff-ff; g = grammar,
mut = arithmetic mutants reported apart), on +qwen (the clean proxy) and +both+mut; a g-ff vs g-g gap >= 5 points =
source bias.
--fake-new (dry run before the box): use the R1 scores as `new` (then A_new must equal ff1_eval's r1 A).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luthor.stats import cluster_signflip, question_signflip  # noqa: E402

DEC = ROOT / "data/sys1/heldout_v2/decision.json"
ADDED = ("ff", "mut")
BETAS = [round(0.1 * i, 2) for i in range(0, 31)]
TAUS = [round(0.1 * i, 2) for i in range(0, 21)]
POOLS = [("grammar", "m_both"), ("+omni", "m_omni"), ("+qwen", "m_qwen"), ("+both", "m_both"), ("+mut", "m_mut"),
         ("+both+mut", "m_both_mut")]
RAW = ("r1", "r2", "new")
VARIANTS = {"r1": ("r1",), "r2": ("r2",), "new": ("new",), "sumz": ("r1", "r2"), "sumz3": ("r1", "r2", "new"),
            "new_r1": ("new", "r1")}
PRIMARY = ("+both+mut", "A", "new")
REF_B = ("+both+mut", "B", "sumz")
REF_A = ("+both+mut", "A", "sumz")
PASS_DELTA, PASS_P = 1.0, 0.05


def live(c: dict) -> bool:
    return bool(c.get("runs") and c.get("nonnull"))


def pct(x: float, n: int) -> float:
    return round(100.0 * x / n, 2) if n else 0.0


def jl(p: Path, keep: set | None = None) -> dict:
    out = {}
    if p.exists():
        for l in open(p):
            r = json.loads(l)
            if keep is None or r["id"] in keep:
                out[r["id"]] = r
    return out


def src_group(c: dict) -> str:
    s = c.get("src")
    return s if s in ADDED else "g"


def load_pool(ff1: Path, ff2: Path, name: str, ids: set, grammar_only: bool, fake_new: bool) -> dict:
    sc = {"r1": jl(ff1 / "scored" / f"{name}_by_r1.jsonl", ids), "r2": jl(ff1 / "scored" / f"{name}_by_r2.jsonl", ids),
          "new": jl((ff1 if fake_new else ff2) / "scored" / f"{name}_by_{'r1' if fake_new else 'new'}.jsonl", ids)}
    if not (sc["r1"] and sc["r2"] and sc["new"]):
        return {}
    look = {k: {i: {c["sql"]: c.get("rr") for c in r["cands"] if c.get("sql")} for i, r in v.items()} for k, v in sc.items()}
    out = {}
    for i, r in sc["r1"].items():
        cl, seen = [], set()
        for c in r["cands"]:
            s = c.get("sql")
            if not s or s in seen:
                continue
            v = {k: look[k].get(i, {}).get(s) for k in RAW}
            if any(x is None for x in v.values()):
                continue
            if grammar_only and c.get("src") in ADDED:
                continue
            seen.add(s)
            cl.append({"ok": bool(c.get("ok")), "live": live(c), "src": src_group(c), "ffs": c.get("ff_sig_n", 0) or 0,
                       **v})
        ff = r.get("ff") or {}
        has_ff = (ff.get("n") or 0) > 0 and not grammar_only
        rec = {"db": r["db_id"], "cands": cl, "vote_ok": bool(ff.get("vote_ok")) if has_ff else None,
               "vote_share": (ff.get("vote_n", 0) / ff["live"]) if has_ff and ff.get("live") else 0.0}
        base = [c for c in cl if c["live"]] or cl
        for k in RAW:
            x = [c[k] for c in base]
            m = sum(x) / len(x) if x else 0.0
            sd = math.sqrt(sum((y - m) ** 2 for y in x) / len(x)) if x else 0.0
            for c in cl:
                c["z" + k] = (c[k] - m) / sd if sd > 0 else 0.0
        for c in cl:
            for vn, parts in VARIANTS.items():
                c["v_" + vn] = sum(c["z" + p] for p in parts)
        out[i] = rec
    return out


def pick(rec: dict, rule: str, v: str, beta: float = 0.0, tau: float = 0.0) -> dict | None:
    """The picked candidate (None: nothing to pick) for rules A, B, D, E (C: see ok_of)."""
    cl = rec["cands"]
    lv = [c for c in cl if c["live"]] or cl
    if rule == "D":
        g = [c for c in lv if c["src"] not in ADDED] or [c for c in cl if c["src"] not in ADDED]
        return max(g, key=lambda c: c["v_" + v]) if g else None
    if not lv:
        return None
    if rule == "A":
        return max(lv, key=lambda c: c["v_" + v])
    return max(lv, key=lambda c: c["v_" + v] + beta * math.log1p(c["ffs"]))


def ok_of(rec: dict, rule: str, v: str, beta: float = 0.0, tau: float = 0.0) -> bool:
    if rule == "C":
        return bool(rec["vote_ok"])
    if rule == "E":
        lv = [c for c in rec["cands"] if c["live"]] or rec["cands"]
        if not lv:
            return False
        val = sorted((c["v_" + v] + beta * math.log1p(c["ffs"]) for c in lv), reverse=True)
        margin = val[0] - val[1] if len(val) > 1 else float("inf")
        if rec["vote_ok"] is not None and rec["vote_share"] >= 0.5 and margin < tau:
            return bool(rec["vote_ok"])
        rule = "B"
    c = pick(rec, rule, v, beta, tau)
    return bool(c and c["ok"])


def b_rows(pool: dict, ids: list[str], v: str, beta: float) -> dict:
    """{id: (ok of the rule-B pick, B's top-2 margin, vote_ok, vote_share)}."""
    out = {}
    for i in ids:
        r = pool.get(i)
        if r is None:
            continue
        lv = [c for c in r["cands"] if c["live"]] or r["cands"]
        if not lv:
            out[i] = (False, float("inf"), r["vote_ok"], r["vote_share"])
            continue
        val = sorted(((c["v_" + v] + beta * math.log1p(c["ffs"]), j) for j, c in enumerate(lv)), reverse=True)
        m = val[0][0] - val[1][0] if len(val) > 1 else float("inf")
        out[i] = (bool(lv[val[0][1]]["ok"]), m, r["vote_ok"], r["vote_share"])
    return out


def e_ok(row: tuple, tau: float) -> bool:
    ok, margin, vok, share = row
    if vok is not None and share >= 0.5 and margin < tau:
        return bool(vok)
    return ok


def per_question(pool: dict, ids: list[str], rule: str, v: str) -> tuple[dict, list]:
    """{id: ok} for every DECISION id (missing = wrong); B / E fitted leave-one-DB-out (as ff1_eval.lodo)."""
    if rule in ("A", "C", "D"):
        return {i: (i in pool and ok_of(pool[i], rule, v)) for i in ids}, []
    taus = TAUS if rule == "E" else [0.0]
    rows = {b: b_rows(pool, ids, v, b) for b in BETAS}
    dbs = sorted({pool[i]["db"] for i in ids if i in pool})
    per_db = {(b, t): collections.Counter() for b in BETAS for t in taus}
    for b in BETAS:
        for t in taus:
            for i, row in rows[b].items():
                per_db[(b, t)][pool[i]["db"]] += e_ok(row, t)
    tot = {p: sum(c.values()) for p, c in per_db.items()}
    res, fits = {i: False for i in ids}, []
    for d in dbs:
        best = max(per_db, key=lambda p: (tot[p] - per_db[p][d], -p[0], -p[1]))
        fits.append(best)
        res.update({i: e_ok(row, best[1]) for i, row in rows[best[0]].items() if pool[i]["db"] == d})
    return res, fits


def pairs_calibration(pool: dict, ids: list[str], v: str) -> dict:
    """Pick accuracy on exactly-one-right live candidate pairs, by source pair (ties count 0.5)."""
    acc = collections.defaultdict(lambda: [0.0, 0])
    for i in ids:
        if i not in pool:
            continue
        lv = [c for c in pool[i]["cands"] if c["live"]]
        right = [c for c in lv if c["ok"]]
        wrong = [c for c in lv if not c["ok"]]
        for r in right:
            for w in wrong:
                s = 1.0 if r["v_" + v] > w["v_" + v] else (0.5 if r["v_" + v] == w["v_" + v] else 0.0)
                pair = "-".join(sorted((r["src"], w["src"])))
                for key in (pair, f"right_{r['src']}/wrong_{w['src']}"):
                    acc[key][0] += s
                    acc[key][1] += 1
    out = {k: {"acc": round(100.0 * a / n, 2), "pairs": n} for k, (a, n) in sorted(acc.items())}
    if "g-g" in out and "ff-g" in out:
        gap = round(out["ff-g"]["acc"] - out["g-g"]["acc"], 2)
        out["gap_gff_minus_gg"] = gap
        out["source_bias"] = abs(gap) >= 5.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff1", type=Path, default=ROOT / "runs/phase1/ff1/ff1")
    ap.add_argument("--ff2", type=Path, default=ROOT / "runs/phase1/ff2/ff2")
    ap.add_argument("--out", type=Path, default=None, help="default: <ff2>/../ff2_eval.json")
    ap.add_argument("--fake-new", action="store_true", help="dry run: R1 scores stand in for the new validator")
    a = ap.parse_args()
    out_p = a.out or a.ff2.parent / ("ff2_eval_fake.json" if a.fake_new else "ff2_eval.json")
    ids = list(json.loads(DEC.read_text())["ids"])
    keep = set(ids)
    n = len(ids)
    res: dict = {"ff1": str(a.ff1), "ff2": str(a.ff2), "fake_new": a.fake_new, "n_decision": n, "pools": {}}
    p1 = json.loads((a.ff1.parent / "ff1_eval.json").read_text()).get("pools", {}) if (a.ff1.parent / "ff1_eval.json").exists() else {}
    vecs, pools = {}, {}
    for label, name in POOLS:
        pool = load_pool(a.ff1, a.ff2, name, keep, label == "grammar", a.fake_new)
        if not pool:
            continue
        pools[label] = pool
        has_ff = label not in ("grammar", "+mut")
        rep = {"oracle": pct(sum(any(c["ok"] for c in r["cands"]) for r in pool.values()), n)}
        for v in VARIANTS:
            s = {}
            for rule in ("A", "D") + (("C", "B", "E") if has_ff else ()):
                if rule == "C" and v != "new":
                    continue
                vec, fits = per_question(pool, ids, rule, v)
                vecs[(label, rule, v)] = vec
                s[rule] = pct(sum(vec.values()), n)
                if fits:
                    s[rule + "_fits"] = fits
            rep[v] = s
        if label in p1:
            rep["phase1_ff1_eval"] = {k: {r: x for r, x in p1[label][k].items() if not r.endswith("_fits")}
                                      for k in ("r1", "r2", "sumz")}
        res["pools"][label] = rep
        line = " ".join(f"{v}:" + ",".join(f"{r}={x}" for r, x in rep[v].items() if not r.endswith("_fits"))
                        for v in VARIANTS)
        print(f"pool {label:10s} oracle {rep['oracle']:6.2f} | {line}")

    # reproduction check: phase-1 numbers recomputed here (same candidates, same LODO grid)
    if "+both+mut" in res["pools"] and "+both+mut" in p1:
        mine, theirs = res["pools"]["+both+mut"]["sumz"], p1["+both+mut"]["sumz"]
        res["phase1_reproduced"] = {r: (mine.get(r), theirs.get(r)) for r in ("A", "B", "D", "E")}
        print("phase-1 reproduction (+both+mut sumz, here vs ff1_eval):", json.dumps(res["phase1_reproduced"]))

    # primary (prereg)
    if PRIMARY in vecs and REF_B in vecs:
        db_of = {i: r["db"] for p in pools.values() for i, r in p.items()}
        prim, refb, refa = vecs[PRIMARY], vecs[REF_B], vecs[REF_A]
        cb = cluster_signflip({i: refb[i] for i in ids}, {i: prim[i] for i in ids}, {i: db_of.get(i, "?") for i in ids})
        qb = question_signflip({i: refb[i] for i in ids}, {i: prim[i] for i in ids})
        ca = cluster_signflip({i: refa[i] for i in ids}, {i: prim[i] for i in ids}, {i: db_of.get(i, "?") for i in ids})
        a_new, b_ref = pct(sum(prim.values()), n), pct(sum(refb.values()), n)
        verdict = ("PASS" if cb["delta"] >= PASS_DELTA and cb["p"] < PASS_P else
                   "KILL" if a_new < b_ref else "HOLD")
        res["primary"] = {"metric": "DECISION +both+mut rule A, new validator alone", "A_new": a_new,
                          "ref_B_sumz_phase1": b_ref, "delta_vs_B": round(cb["delta"], 2), "signflip_p_vs_B": cb["p"],
                          "ci95_vs_B": cb["ci"], "per_db_vs_B": cb["per_db"], "mcnemar_vs_B": qb,
                          "ref_A_sumz_phase1": pct(sum(refa.values()), n), "delta_vs_A": round(ca["delta"], 2),
                          "signflip_p_vs_A": ca["p"], "verdict": verdict, "fake_new": a.fake_new}
        print(f"PRIMARY A_new {a_new} vs phase-1 B sumz {b_ref}: delta {cb['delta']:+.2f}, DB sign-flip p={cb['p']:.3f}, "
              f"McNemar +{qb['wins']}/-{qb['losses']} p={qb['p']:.3f}; vs phase-1 A sumz {res['primary']['ref_A_sumz_phase1']}: "
              f"{ca['delta']:+.2f} (p={ca['p']:.3f}) -> {verdict}{' (FAKE new = R1)' if a.fake_new else ''}")

    # calibration (§3.3-3)
    res["calibration"] = {}
    for label in ("+qwen", "+both+mut"):
        if label in pools:
            res["calibration"][label] = {v: pairs_calibration(pools[label], ids, v) for v in ("new", "r1", "r2", "sumz")}
            for v, c in res["calibration"][label].items():
                print(f"calib {label:10s} {v:5s} " + " ".join(f"{k}={x['acc']}({x['pairs']})" for k, x in c.items()
                                                           if isinstance(x, dict) and "/" not in k)
                      + f" gap={c.get('gap_gff_minus_gg')} bias={c.get('source_bias')}")

    # what rule A (new) picks, by source
    if "+both+mut" in pools:
        by = collections.Counter()
        for i in ids:
            r = pools["+both+mut"].get(i)
            c = pick(r, "A", "new") if r else None
            if c:
                by[(c["src"], c["ok"])] += 1
        res["A_new_picks_by_source"] = {f"{s}_{'right' if ok else 'wrong'}": k for (s, ok), k in sorted(by.items())}
        print("A_new picks by source:", json.dumps(res["A_new_picks_by_source"]))
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=1, default=str))
    print(f"-> {out_p}")


if __name__ == "__main__":
    main()
