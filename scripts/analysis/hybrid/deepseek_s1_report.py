"""DeepSeek V4.1 Flash alone vs + System 1's candidates (think off / on) on the 796 DECISION questions (eval only).

Arms (sample 0 of each `teacher.py label` run, gold match from `teacher.py verify`):
  alone_t0     data/teacher/bakeoff/deepseek_t0 (+ deepseek_t0_uncovered)            think off
  alone_think  data/teacher/bakeoff/deepseek_t0_think (+ deepseek_t0_think_uncovered) think on
  s1_t0        data/teacher/bakeoff/deepseek_s1_t0                                    think off, --s1-cands
  s1_think     data/teacher/bakeoff/deepseek_s1_think                                 think on,  --s1-cands
Per arm: EX, tokens, call seconds, wall, $ at DeepSeek list price, $ per correct; +S1 arms: copy share and McNemar
vs the alone arm with the same think setting. Plus "System 1 skips the LLM when confident": the k most confident
questions (VALHARD top-2 class margin) take System 1's own pick, the rest the +S1 answer.

  .venv/bin/python scripts/analysis/hybrid/deepseek_s1_report.py [--out runs/analysis/hybrid/deepseek_s1.json]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BAKE = ROOT / "data/teacher/bakeoff"
DECISION = ROOT / "data/sys1/heldout_v2/decision.json"
CANDS = ROOT / "runs/analysis/hybrid/s1_cands_decision.jsonl"
VALHARD = ROOT / "runs/phase1/valhard/valhard/heldout_by_new.jsonl"
ARMS = {"alone_t0": ["deepseek_t0", "deepseek_t0_uncovered"],
        "alone_think": ["deepseek_t0_think", "deepseek_t0_think_uncovered"],
        "s1_t0": ["deepseek_s1_t0"],
        "s1_think": ["deepseek_s1_think"]}
PAIRS = {"s1_t0": "alone_t0", "s1_think": "alone_think"}
PRICE_IN, PRICE_OUT = 0.15, 0.60  # $ per million tokens, DeepSeek list price off-peak; peak = x2
S1_COST_PER_Q = 0.637 / 1000  # System 1 measured cost per question
SKIP_SHARES = (0.0, 0.25, 0.50, 0.75)


def read_jsonl(p) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def norm_sql(s: str) -> str:
    s = " ".join((s or "").split()).strip()
    return s[:-1].rstrip() if s.endswith(";") else s


def dollars(n_in: int, n_out: int, peak: bool = False) -> float:
    return (n_in * PRICE_IN + n_out * PRICE_OUT) / 1e6 * (2 if peak else 1)


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Two-sided exact McNemar (binomial on discordant pairs). b_only = b right & a wrong."""
    a_only = sum(x and not y for x, y in zip(a, b))
    b_only = sum(y and not x for x, y in zip(a, b))
    n = a_only + b_only
    k = min(a_only, b_only)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": a_only, "b_only": b_only, "p": p}


def s1_groups(cands: list[dict]) -> list[list[dict]]:
    """Same grouping as s1_candidates.py: R1 live candidates by result signature, best group first by VALHARD rr."""
    c = [x for x in cands if x.get("sql") and x.get("src") in ("r1", "both", None)]
    lv = [x for x in c if x.get("runs") and x.get("nonnull")] or c
    groups: dict = {}
    for x in sorted(lv, key=lambda x: -x.get("rr", -1e9)):
        groups.setdefault(x.get("sig") or x["sql"], []).append(x)
    return list(groups.values())


def s1_pick_and_margin(cands: list[dict]):
    """(pick ok, top-2 class margin) or None when there is no candidate. Single class -> margin inf."""
    g = s1_groups(cands)
    if not g:
        return None
    margin = g[0][0].get("rr", 0.0) - g[1][0].get("rr", 0.0) if len(g) > 1 else math.inf
    return bool(g[0][0].get("ok")), margin


def skip_curve(ids, llm_ok: dict, llm_in: dict, llm_out: dict, s1: dict, shares=SKIP_SHARES) -> list[dict]:
    """s1: id -> (pick_ok, margin) for questions with candidates. Skip the k = round(share * len(ids)) most confident
    (largest margin; ties by id) questions: they take System 1's pick and make no LLM call."""
    order = sorted((i for i in ids if i in s1), key=lambda i: (-s1[i][1], i))
    out = []
    for sh in shares:
        k = min(len(order), round(sh * len(ids)))
        skip = set(order[:k])
        ok = sum(s1[i][0] if i in skip else llm_ok[i] for i in ids)
        n_in = sum(llm_in[i] for i in ids if i not in skip)
        n_out = sum(llm_out[i] for i in ids if i not in skip)
        llm = dollars(n_in, n_out)
        tot = llm + S1_COST_PER_Q * len(ids)
        out.append({"skip_share": sh, "skipped": k, "llm_calls": len(ids) - k,
                    "skipped_s1_ex": round(sum(s1[i][0] for i in skip) / k, 4) if k else None,
                    "ex": round(ok / len(ids), 4), "correct": ok, "tokens_in": n_in, "tokens_out": n_out,
                    "llm_usd": round(llm, 4), "llm_usd_peak": round(2 * llm, 4),
                    "usd_with_s1": round(tot, 4), "usd_per_correct": round(tot / max(1, ok), 6)})
    return out


def load_arm(dirs: list[str], ids: set) -> tuple[dict, dict]:
    rows, wall = {}, {}
    for d in dirs:
        rd = BAKE / d
        if not (rd / "verified.jsonl").exists():
            return {}, {}
        meta = json.loads((rd / "run.json").read_text())
        n_rows = 0
        for r in read_jsonl(rd / "verified.jsonl"):
            if r["id"] in ids and r["id"] not in rows:
                rows[r["id"]] = r
                n_rows += 1
        wall[d] = {"rows_in_run": sum(s.get("rows", 0) for s in meta["sessions"][:1]), "decision_rows": n_rows,
                   "wall_seconds": round(sum(s.get("seconds", 0) for s in meta["sessions"]), 1),
                   "think": meta.get("think"), "max_tokens": meta.get("max_tokens"),
                   "prompt_version": meta.get("prompt_version"), "eval_only": meta.get("eval_only")}
    return rows, wall


def sample0(r: dict) -> dict:
    return next((s for s in r["samples"] if s["sample_idx"] == 0), {})


def arm_stats(ids: list[str], rows: dict, cands: dict | None) -> dict:
    s0 = {i: sample0(rows[i]) if i in rows else {} for i in ids}
    ok = {i: bool(s0[i].get("match")) for i in ids}
    n_in = {i: s0[i].get("n_in", 0) or 0 for i in ids}
    n_out = {i: s0[i].get("n_out", 0) or 0 for i in ids}
    secs = [s0[i]["seconds"] for i in ids if s0[i].get("seconds")]
    correct = sum(ok.values())
    usd = dollars(sum(n_in.values()), sum(n_out.values()))
    st = {"n": len(ids), "answered": sum(1 for i in ids if s0[i].get("gen_status") == "ok"),
          "missing_or_failed": sum(1 for i in ids if s0[i].get("gen_status") != "ok"),
          "exec_error": sum(1 for i in ids if str(s0[i].get("exec", "")).startswith("error")),
          "ex": round(correct / len(ids), 4), "correct": correct,
          "tokens_in": sum(n_in.values()), "tokens_out": sum(n_out.values()),
          "n_in_mean": round(statistics.mean(n_in.values()), 1), "n_out_mean": round(statistics.mean(n_out.values()), 1),
          "call_seconds_median": round(statistics.median(secs), 2) if secs else None,
          "call_seconds_mean": round(statistics.mean(secs), 2) if secs else None,
          "call_seconds_sum": round(sum(secs), 1), "wall_at_3_workers_est_seconds": round(sum(secs) / 3, 1),
          "usd_offpeak": round(usd, 4), "usd_peak": round(2 * usd, 4),
          "usd_per_correct_offpeak": round(usd / max(1, correct), 6)}
    if cands is not None:
        st["usd_with_s1"] = round(usd + S1_COST_PER_Q * len(ids), 4)
        st["usd_per_correct_with_s1"] = round((usd + S1_COST_PER_Q * len(ids)) / max(1, correct), 6)
        copy = {i: next((c["label"] for c in cands.get(i, []) if norm_sql(c["sql"]) == norm_sql(s0[i].get("sql", ""))),
                        None) for i in ids}
        has = [i for i in ids if cands.get(i)]
        cp = [i for i in has if copy[i]]
        own = [i for i in has if not copy[i]]
        st["copy"] = {"questions_with_cands": len(has), "copied": len(cp), "copy_share": round(len(cp) / max(1, len(has)), 4),
                      "copied_A": sum(copy[i] == "A" for i in cp),
                      "ex_when_copied": round(sum(ok[i] for i in cp) / max(1, len(cp)), 4),
                      "ex_when_own": round(sum(ok[i] for i in own) / max(1, len(own)), 4),
                      "no_cands_ex": round(sum(ok[i] for i in ids if not cands.get(i)) /
                                           max(1, sum(1 for i in ids if not cands.get(i))), 4)}
    return {"stats": st, "ok": ok, "n_in": n_in, "n_out": n_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs/analysis/hybrid/deepseek_s1.json")
    a = ap.parse_args()
    ids = json.load(open(DECISION))["ids"]
    idset = set(ids)
    cands = {r["question_id"]: r["candidates"] for r in read_jsonl(CANDS)}
    s1 = {}
    for r in read_jsonl(VALHARD):
        if r["id"] in idset and cands.get(r["id"]):
            pm = s1_pick_and_margin(r["cands"])
            if pm:
                s1[r["id"]] = pm
    rep = {"questions": len(ids), "split": "DECISION (data/sys1/heldout_v2/decision.json)", "model": "deepseek-v4.1-flash:cloud",
           "price": {"in_per_M": PRICE_IN, "out_per_M": PRICE_OUT, "peak_multiplier": 2, "s1_per_1000_q": 0.637},
           "system1_pick": {"n_with_cands": len(s1), "ex_all_796": round(sum(v[0] for v in s1.values()) / len(ids), 4),
                            "ex_on_cands": round(sum(v[0] for v in s1.values()) / max(1, len(s1)), 4),
                            "usd": round(S1_COST_PER_Q * len(ids), 4)},
           "arms": {}, "runs": {}, "mcnemar": {}, "skip": {}}
    res = {}
    for arm, dirs in ARMS.items():
        rows, wall = load_arm(dirs, idset)
        if len(rows) < len(ids):
            rep["arms"][arm] = {"missing": f"{len(rows)}/{len(ids)} questions verified"}
            if not rows:
                continue
        res[arm] = arm_stats(ids, rows, cands if arm.startswith("s1") else None)
        rep["arms"][arm] = res[arm]["stats"]
        rep["runs"][arm] = wall
    for arm, base in PAIRS.items():
        if arm in res and base in res:
            m = mcnemar_exact([res[base]["ok"][i] for i in ids], [res[arm]["ok"][i] for i in ids])
            rep["mcnemar"][f"{arm}_vs_{base}"] = {"alone_only": m["a_only"], "s1_only": m["b_only"], "p": round(m["p"], 6),
                                                 "delta_ex": round((res[arm]["stats"]["correct"]
                                                                    - res[base]["stats"]["correct"]) / len(ids), 4)}
            rep["skip"][arm] = skip_curve(ids, res[arm]["ok"], res[arm]["n_in"], res[arm]["n_out"], s1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    print(table(rep))
    print("wrote", a.out)


def table(rep: dict) -> str:
    L = ["arm          EX      n_in   n_out  sec_med sec_mean  $offpk  $peak  $/correct  copy",
         "-" * 86]
    for arm, s in rep["arms"].items():
        if "ex" not in s:
            L.append(f"{arm:12} {s.get('missing')}")
            continue
        cp = f"{s['copy']['copy_share']:.1%}" if "copy" in s else ""
        L.append(f"{arm:12} {s['ex']:.4f} {s['n_in_mean']:7.0f} {s['n_out_mean']:7.0f} {s['call_seconds_median'] or 0:7.2f} "
                 f"{s['call_seconds_mean'] or 0:8.2f} {s['usd_offpeak']:7.4f} {s['usd_peak']:6.4f} {s['usd_per_correct_offpeak']:.6f}  {cp}")
    for k, m in rep["mcnemar"].items():
        L.append(f"McNemar {k}: alone_only={m['alone_only']} s1_only={m['s1_only']} dEX={m['delta_ex']:+.4f} p={m['p']:.4g}")
    for arm, curve in rep["skip"].items():
        for c in curve:
            L.append(f"skip {arm} {c['skip_share']:.0%}: EX={c['ex']:.4f} calls={c['llm_calls']} in={c['tokens_in']:,} "
                     f"out={c['tokens_out']:,} llm$={c['llm_usd']:.4f} total$={c['usd_with_s1']:.4f}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
