"""Mac side, CPU: evaluate the ff1 box run (scripts/chains/ff1_box.sh; plan docs/paper_notes/research/
freeform_plan_2026-10-10.md, §1.3-1.5, §3.4). DECISION only (796 ids; missing or uncovered = wrong) plus the BIRD-train
bug set (training DBs) for P2. Never reads RESERVED or BIRD dev.

  uv run python scripts/analysis/ff1_eval.py [--dir runs/phase1/ff1/ff1] [--out runs/phase1/ff1/ff1_eval.json]

Inputs under --dir: lab/{omni,qwen}_{dec,noev,bug}.jsonl (every sample labelled, merge.py --samples) and
scored/<pool>_by_r{1,2}.jsonl (merged pools with validator scores).

Reports
- per source (grammar only; +omni; +qwen; +both; +mut; +both+mut): oracle and the pick of
    A  max validator score over live candidates (today's rule, all sources)
    B  z(score) + beta * log(1 + ff_sig_n)                              (beta LODO-fitted)
    C  ff vote alone (most common live ff result; OmniSQL maj@n)
    D  grammar only (max score over grammar candidates)
    E  B, but take the ff vote when B's top-2 margin < tau and the vote holds >= 50% of live ff samples (beta, tau LODO)
  for three scores: R1, R2 and sum_z (per-question z of R1 + z of R2). LODO = fit on 7 DECISION DBs, apply to the 8th.
- ff alone per generator: greedy, vote@8, vote@16, oracle@16 (pass@16), mean sample accuracy.
- P1 gold-text recall (per sample), P2 wrong-gold reproduction (DECISION + training-DB bug set pooled), P3 evidence
  ablation, P4 difference-in-differences vs the published BIRD-dev gaps (greedy +13.0, vote +4.8), and the §1.5
  "material contamination?" call.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/freeform"))
import probe  # noqa: E402

DEC = ROOT / "data/sys1/heldout_v2/decision.json"
TRAIN = ROOT / "data/raw/bird_train/train.json"
BUG = ROOT / "data/ff1/bug_train_ids.json"
DEV_GAP_GREEDY, DEV_GAP_VOTE = 63.9 - 50.9, 66.1 - 61.3  # OmniSQL paper Table 4: OmniSQL-7B vs Qwen2.5-Coder-7B
ADDED = ("ff", "mut")
BETAS = [round(0.1 * i, 2) for i in range(0, 31)]
TAUS = [round(0.1 * i, 2) for i in range(0, 21)]
POOLS = [("grammar", "m_both"), ("+omni", "m_omni"), ("+qwen", "m_qwen"), ("+both", "m_both"), ("+mut", "m_mut"),
         ("+both+mut", "m_both_mut")]


def live(c: dict) -> bool:
    return bool(c.get("runs") and c.get("nonnull"))


def pct(x: float, n: int) -> float:
    return round(100.0 * x / n, 2) if n else 0.0


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


# ---- ff samples -----------------------------------------------------------------------------------------------------

def vote_ok(cands: list[dict], k: int) -> bool:
    """Majority result among the first k live temperature samples (ties -> first seen)."""
    s = [c for c in sorted(cands, key=lambda c: c["rank"]) if c.get("temp") not in (0.0, None)][:k]
    lv = [c for c in s if live(c)]
    if not lv:
        return False
    n = collections.Counter(c["sig"] for c in lv)
    best = max(n.values())
    return bool(next(c for c in lv if n[c["sig"]] == best).get("ok"))


def gen_stats(recs: dict, ids: list[str], db_of: dict, skip_db: str | None = None) -> dict:
    ids = [i for i in ids if db_of.get(i) != skip_db]
    n = len(ids)
    g = v8 = v16 = orc = 0
    acc, ns = 0.0, 0
    for i in ids:
        r = recs.get(i)
        if not r:
            continue
        cl = r["cands"]
        gr = next((c for c in cl if c["rank"] == 0 and c.get("temp") == 0.0), None)
        g += bool(gr and live(gr) and gr.get("ok"))
        v8 += vote_ok(cl, 8)
        v16 += vote_ok(cl, 16)
        smp = [c for c in cl if c.get("temp") not in (0.0, None)]
        orc += any(c.get("ok") for c in smp)
        acc += sum(bool(c.get("ok")) for c in smp)
        ns += len(smp)
    return {"n": n, "greedy": pct(g, n), "vote@8": pct(v8, n), "vote@16": pct(v16, n), "oracle@16": pct(orc, n),
            "sample_acc": round(100.0 * acc / ns, 2) if ns else None}


def text_probe(recs: dict) -> dict:
    tot = em = ok_n = ok_em = 0
    for r in recs.values():
        g = probe.canon(r.get("gold_sql"))
        for c in r["cands"]:
            hit = bool(g) and canon_cached(c.get("sql_raw") or c.get("sql")) == g
            tot += 1
            em += hit
            if c.get("ok"):
                ok_n += 1
                ok_em += hit
    return {"samples": tot, "gold_text_em": round(em / tot, 4) if tot else None, "right": ok_n,
            "gold_text_em_among_right": round(ok_em / ok_n, 4) if ok_n else None}


_CANON: dict = {}


def canon_cached(s):
    if s not in _CANON:
        _CANON[s] = probe.canon(s)
    return _CANON[s]


def bug_probe(recs: dict, bugs: dict) -> dict:
    po = pv = go = gv = ao = av = 0
    for qid, b in bugs.items():
        r = recs.get(qid)
        if not r:
            continue
        cl = sorted(r["cands"], key=lambda c: c["rank"])
        lv = [c for c in cl if c.get("temp") not in (0.0, None) and live(c)]
        if lv:
            n = collections.Counter(c["sig"] for c in lv)
            vs = next(c["sig"] for c in lv if n[c["sig"]] == max(n.values()))
            po += vs == b["orig_sig"]
            pv += vs == b["ver_sig"]
        gr = next((c for c in cl if c["rank"] == 0 and c.get("temp") == 0.0 and live(c)), None)
        if gr:
            go += gr["sig"] == b["orig_sig"]
            gv += gr["sig"] == b["ver_sig"]
        sigs = {c["sig"] for c in cl if live(c)}
        ao += b["orig_sig"] in sigs
        av += b["ver_sig"] in sigs

    def ratio(o, v):
        return round(o / (o + v), 3) if o + v else None
    return {"n_bug": sum(q in recs for q in bugs), "vote_orig": po, "vote_verified": pv, "repro_vote": ratio(po, pv),
            "greedy_orig": go, "greedy_verified": gv, "repro_greedy": ratio(go, gv),
            "pool_orig": ao, "pool_verified": av, "repro_pool": ratio(ao, av)}


# ---- pools + pick rules ---------------------------------------------------------------------------------------------

def load_pool(d: Path, name: str, grammar_only: bool) -> dict:
    r1 = {r["id"]: r for r in jl(d / "scored" / f"{name}_by_r1.jsonl")}
    r2 = {r["id"]: r for r in jl(d / "scored" / f"{name}_by_r2.jsonl")}
    out = {}
    for i, r in r1.items():
        s2 = {c["sql"]: c.get("rr") for c in r2.get(i, {}).get("cands", []) if c.get("sql")}
        cl, seen = [], set()
        for c in r["cands"]:
            if not c.get("sql") or c["sql"] in seen or "rr" not in c or s2.get(c["sql"]) is None:
                continue
            if grammar_only and c.get("src") in ADDED:
                continue
            seen.add(c["sql"])
            cl.append({"ok": bool(c.get("ok")), "live": live(c), "src": c.get("src"), "r1": c["rr"], "r2": s2[c["sql"]],
                       "ffs": c.get("ff_sig_n", 0) or 0})
        ff = r.get("ff") or {}
        has_ff = (ff.get("n") or 0) > 0 and not grammar_only
        out[i] = {"db": r["db_id"], "cands": cl, "vote_ok": bool(ff.get("vote_ok")) if has_ff else None,
                  "vote_share": (ff.get("vote_n", 0) / ff["live"]) if has_ff and ff.get("live") else 0.0}
    for rec in out.values():  # per-question z of each score over live candidates (all candidates when none is live)
        base = [c for c in rec["cands"] if c["live"]] or rec["cands"]
        for k in ("r1", "r2"):
            v = [c[k] for c in base]
            m = sum(v) / len(v) if v else 0.0
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v)) if v else 0.0
            for c in rec["cands"]:
                c["z" + k] = (c[k] - m) / sd if sd > 0 else 0.0
        for c in rec["cands"]:
            c["sumz"] = c["zr1"] + c["zr2"]
    return out


def zname(score: str) -> str:
    return {"r1": "zr1", "r2": "zr2", "sumz": "sumz"}[score]


def pick_ok(rec: dict, rule: str, score: str, beta: float = 0.0, tau: float = 0.0) -> bool:
    cl = rec["cands"]
    lv = [c for c in cl if c["live"]] or cl
    if rule == "C":
        return bool(rec["vote_ok"])
    if rule == "D":
        g = [c for c in lv if c["src"] not in ADDED] or [c for c in cl if c["src"] not in ADDED]
        return bool(g) and max(g, key=lambda c: c[score])["ok"]
    if not lv:
        return False
    if rule == "A":
        return max(lv, key=lambda c: c[score])["ok"]
    z = zname(score)
    val = sorted(((c[z] + beta * math.log1p(c["ffs"]), j) for j, c in enumerate(lv)), reverse=True)
    best = lv[val[0][1]]
    if rule == "E" and rec["vote_ok"] is not None and rec["vote_share"] >= 0.5:
        margin = val[0][0] - val[1][0] if len(val) > 1 else float("inf")
        if margin < tau:
            return bool(rec["vote_ok"])
    return best["ok"]


def b_table(recs: list[dict], score: str, beta: float) -> list[tuple]:
    """Per record: (db, ok of the rule-B pick, B's top-2 margin, vote_ok, vote_share)."""
    z, rows = zname(score), []
    for r in recs:
        lv = [c for c in r["cands"] if c["live"]] or r["cands"]
        if not lv:
            rows.append((r["db"], False, float("inf"), r["vote_ok"], r["vote_share"]))
            continue
        val = sorted(((c[z] + beta * math.log1p(c["ffs"]), j) for j, c in enumerate(lv)), reverse=True)
        m = val[0][0] - val[1][0] if len(val) > 1 else float("inf")
        rows.append((r["db"], bool(lv[val[0][1]]["ok"]), m, r["vote_ok"], r["vote_share"]))
    return rows


def e_ok(row: tuple, tau: float) -> bool:
    _, ok, margin, vok, share = row
    if vok is not None and share >= 0.5 and margin < tau:
        return bool(vok)
    return ok


def lodo(pool: dict, ids: list[str], rule: str, score: str) -> tuple[int, list]:
    """Leave-one-DB-out: (beta, tau) fitted on the other DBs, scored on the held-out one. -> (right, fitted params)."""
    recs = [pool[i] for i in ids if i in pool]
    taus = TAUS if rule == "E" else [0.0]
    tab = {b: b_table(recs, score, b) for b in BETAS}
    dbs = sorted({r["db"] for r in recs})
    right, fits = 0, []
    for d in dbs:
        best = max(((b, t) for b in BETAS for t in taus),
                   key=lambda p: (sum(e_ok(x, p[1]) for x in tab[p[0]] if x[0] != d), -p[0], -p[1]))
        fits.append(best)
        right += sum(e_ok(x, best[1]) for x in tab[best[0]] if x[0] == d)
    return right, fits


def pool_report(pool: dict, ids: list[str], has_ff: bool) -> dict:
    n = len(ids)
    recs = [pool[i] for i in ids if i in pool]
    out = {"n": n, "records": len(recs), "oracle": pct(sum(any(c["ok"] for c in r["cands"]) for r in recs), n)}
    for score in ("r1", "r2", "sumz"):
        s = {"A": pct(sum(pick_ok(r, "A", score) for r in recs), n),
             "D": pct(sum(pick_ok(r, "D", score) for r in recs), n)}
        if has_ff:
            s["C"] = pct(sum(pick_ok(r, "C", score) for r in recs), n)
            for rule in ("B", "E"):
                right, fits = lodo(pool, ids, rule, score)
                s[rule] = pct(right, n)
                s[rule + "_fits"] = fits
        out[score] = s
    return out


# ---- main -----------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "runs/phase1/ff1/ff1")
    ap.add_argument("--out", type=Path, default=None, help="default: <dir>/../ff1_eval.json")
    ap.add_argument("--bug-seconds", type=float, default=5.0)
    a = ap.parse_args()
    out_p = a.out or a.dir.parent / "ff1_eval.json"
    ids = list(json.loads(DEC.read_text())["ids"])
    tr = json.loads(TRAIN.read_text())
    db_of = {f"bird_train_train_{i}": r["db_id"] for i, r in enumerate(tr)}
    dbs = sorted({db_of[i] for i in ids})
    res: dict = {"dir": str(a.dir), "n_decision": len(ids), "dbs": dbs}

    lab = {k: {r["id"]: r for r in jl(a.dir / "lab" / f"{k}.jsonl")}
           for k in ("omni_dec", "qwen_dec", "omni_noev", "qwen_noev", "omni_bug", "qwen_bug")}
    res["files"] = {k: len(v) for k, v in lab.items()}
    print("files", res["files"])

    # ff alone + P3 / P4
    res["ff_alone"] = {k: gen_stats(lab[k], ids, db_of) for k in ("omni_dec", "qwen_dec", "omni_noev", "qwen_noev")
                       if lab[k]}
    for k, v in res["ff_alone"].items():
        print(f"ff {k:10s} {v}")
    p3 = {}
    for m in ("omni", "qwen"):
        w, wo = res["ff_alone"].get(f"{m}_dec"), res["ff_alone"].get(f"{m}_noev")
        if w and wo:
            p3[m] = {k: round(w[k] - wo[k], 2) for k in ("greedy", "vote@8") if w[k] is not None and wo[k] is not None}
    res["P3_evidence_drop"] = p3
    p4 = None
    if lab["omni_dec"] and lab["qwen_dec"]:
        def dd(skip=None):
            o, q = gen_stats(lab["omni_dec"], ids, db_of, skip), gen_stats(lab["qwen_dec"], ids, db_of, skip)
            return round(o["greedy"] - q["greedy"] - DEV_GAP_GREEDY, 2), round(o["vote@8"] - q["vote@8"] - DEV_GAP_VOTE, 2)
        g, v = dd()
        lo = [dd(d) for d in dbs]
        p4 = {"greedy": g, "vote@8": v, "lodo_greedy": [min(x[0] for x in lo), max(x[0] for x in lo)],
              "lodo_vote@8": [min(x[1] for x in lo), max(x[1] for x in lo)]}
    res["P4_inflation"] = p4

    # P1
    res["P1_text"] = {k: text_probe(lab[k]) for k in ("omni_dec", "qwen_dec") if lab[k]}
    p1_ratio = None
    if len(res["P1_text"]) == 2:
        o, q = (res["P1_text"][k]["gold_text_em_among_right"] for k in ("omni_dec", "qwen_dec"))
        p1_ratio = round(o / q, 2) if o is not None and q else (float("inf") if o else None)
    res["P1_ratio_among_right"] = p1_ratio

    # P2: DECISION bug questions + the training-DB bug set, pooled
    cache = a.dir / "bug_sigs.json"
    bugs = json.loads(cache.read_text()) if cache.exists() else {}
    need = set(ids) | set(json.loads(BUG.read_text())["ids"]) if BUG.exists() else set(ids)
    if not need <= set(bugs.get("_checked", [])):
        found = probe.bug_questions(need - set(bugs.get("_checked", [])), seconds=a.bug_seconds)
        bugs.update(found)
        bugs["_checked"] = sorted(set(bugs.get("_checked", [])) | need)
        cache.write_text(json.dumps(bugs))
    bq = {k: v for k, v in bugs.items() if not k.startswith("_")}
    p2 = {}
    for m in ("omni", "qwen"):
        recs = {**lab[f"{m}_dec"], **lab[f"{m}_bug"]}
        if recs:
            p2[m] = {"pooled": bug_probe(recs, bq), "decision": bug_probe(lab[f"{m}_dec"], bq),
                     "train_dbs": bug_probe(lab[f"{m}_bug"], bq)}
    res["P2_bug"] = p2

    # material? (plan §1.5)
    why = []
    if p4 and (p4["greedy"] >= 5 or p4["vote@8"] >= 5):
        why.append(f"P4 inflation greedy {p4['greedy']} / vote {p4['vote@8']} >= 5")
    if p1_ratio is not None and p1_ratio >= 3:
        why.append(f"P1 gold-text EM among right: OmniSQL {p1_ratio}x Qwen >= 3x")
    if "omni" in p2 and "qwen" in p2:
        ro, rq = p2["omni"]["pooled"]["repro_vote"], p2["qwen"]["pooled"]["repro_vote"]
        if ro is not None and rq is not None and ro - rq >= 0.2:
            why.append(f"P2 repro ratio (vote) OmniSQL {ro} vs Qwen {rq}: +{round(ro - rq, 3)} >= 0.2")
    complete = bool(p4) and p1_ratio is not None and "omni" in p2 and "qwen" in p2
    res["material"] = {"material": bool(why), "reasons": why, "all_probes_present": complete,
                       "proxy_source": "+qwen" if why else "+omni",
                       "note": "material -> DECISION numbers with OmniSQL ff are not a gain estimate; fit rule shape on "
                               "+qwen (proxy) and size the gain on the one owner-approved BIRD-dev look"}

    # pools
    res["pools"] = {}
    for label, name in POOLS:
        if not (a.dir / "scored" / f"{name}_by_r1.jsonl").exists():
            continue
        pool = load_pool(a.dir, name, grammar_only=(label == "grammar"))
        res["pools"][label] = pool_report(pool, ids, has_ff=label not in ("grammar", "+mut"))
        r = res["pools"][label]
        line = " ".join(f"{s}:" + ",".join(f"{k}={v}" for k, v in r[s].items() if not k.endswith("_fits"))
                        for s in ("r1", "r2", "sumz"))
        print(f"pool {label:10s} oracle {r['oracle']:6.2f} | {line}")

    print("P1", json.dumps(res["P1_text"]), "ratio", p1_ratio)
    print("P2", json.dumps({m: v["pooled"] for m, v in p2.items()}))
    print("P3 evidence drop (with - without)", json.dumps(p3))
    print("P4 inflation", json.dumps(p4))
    print(f"MATERIAL {json.dumps(res['material'])}")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=1))
    print(f"-> {out_p}")


if __name__ == "__main__":
    main()
