"""Pre-registered secondary endpoints for the RESERVED cost confirmation (PREREG.md), from the runs score.py scored.

- Routing baselines at the cascade's agent share (177/348), using the B1 answers on routed questions and System 1's
  pick elsewhere: random routing (expected value), System 1 vote agreement (share of the 32 plans in the top result
  group; lowest agreement goes to the agent) and question length (longest goes to the agent).
- Price sensitivity of the cost saving: caching off (cache writes and reads billed as plain input) and Sonnet 5.5 prices.

  .venv/bin/python experiments/cost_confirmation/secondary.py --box <sandbox> --tasks <dir> \
      --s1 runs/phase2/cost_confirmation/s1_reserved.jsonl --s1-cost-per-q 0.000849 --out experiments/cost_confirmation/secondary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402

PRICES = {  # $/M: input, cache write, cache read, output
    "opus_list": (4.0, 5.0, 0.20, 20.0),
    "opus_no_cache": (4.0, 4.0, 4.0, 20.0),
    "sonnet_list": (2.0, 2.50, 0.10, 10.0),
    "sonnet_no_cache": (2.0, 2.0, 2.0, 10.0),
}


def cost(t, p):
    i, w, r, o = p
    return (t.get("input_tokens", 0) * i + t.get("cache_creation_input_tokens", 0) * w
            + t.get("cache_read_input_tokens", 0) * r + max(t.get("output_tokens", 0), t.get("output_est", 0)) * o) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--s1", type=Path, required=True)
    ap.add_argument("--s1-cost-per-q", type=float, required=True)
    ap.add_argument("--t", type=float, default=3.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tr = json.load(open(score.TRAIN))
    ids = json.load(open(score.ROOT / "data/sys1/heldout_v2/reserved.json"))["ids"]
    gold = {i: tr[int(i.rsplit("_", 1)[1])]["SQL"] for i in ids}
    dbid = {i: tr[int(i.rsplit("_", 1)[1])]["db_id"] for i in ids}
    s1 = {json.loads(l)["id"]: json.loads(l) for l in open(a.s1)}
    routed = {i for i in ids if s1[i]["margin"] is None or s1[i]["margin"] < a.t}
    cands = {i: {c["label"]: c["sql"] for c in s1[i].get("candidates", [])} for i in ids}
    b1 = {}
    for f in sorted(glob.glob(str(a.box / "answers/B1/*.jsonl"))):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            q = r.get("question_id")
            if q in b1 or q not in dbid:
                continue
            p = str(r.get("pick", "NONE")).strip()
            b1[q] = cands[q].get(p) if p.upper() != "NONE" else (r.get("sql") or None)
    tasks = {(dbid[i], s) for i in ids for s in (gold[i], b1.get(i), s1[i]["sql"]) if s}
    with ProcessPoolExecutor(8) as ex:
        res = dict(ex.map(score.run, sorted(tasks), chunksize=16))
    ok = lambda i, s: bool(s) and res.get((dbid[i], gold[i])) is not None and res.get((dbid[i], s)) == res.get((dbid[i], gold[i]))
    agent_ok = {i: ok(i, b1.get(i)) for i in ids}
    s1_ok = {i: ok(i, s1[i]["sql"]) for i in ids}
    k, n = len(routed), len(ids)
    ex_of = lambda sel: round(100 * sum(agent_ok[i] if i in sel else s1_ok[i] for i in ids) / n, 2)
    gain = {i: agent_ok[i] - s1_ok[i] for i in ids}
    agree = {i: (s1[i]["candidates"][0]["n_of_32"] / max(1, sum(c["n_of_32"] for c in s1[i]["candidates"]))) if s1[i].get("candidates") else 0.0 for i in ids}
    qlen = {i: len(s1[i].get("question", "")) for i in ids}
    by_agree = set(sorted(ids, key=lambda i: (agree[i], i))[:k])
    by_len = set(sorted(ids, key=lambda i: (-qlen[i], i))[:k])
    out = {"agent_share": round(k / n, 4), "routing_ex_with_B1_answers": {
        "margin_t3": ex_of(routed),
        "random_expected": round(100 * (sum(s1_ok.values()) + k / n * sum(gain.values())) / n, 2),
        "vote_agreement": ex_of(by_agree),
        "question_length": ex_of(by_len),
        "all_to_agent_B1": ex_of(set(ids)),
        "system1_only": ex_of(set())},
        "routing_note": "vote agreement = top result group's plans / plans in the shown top-8 groups"}
    U = score.usage(a.tasks, a.box)
    out["price_sensitivity"] = {}
    for name, p in PRICES.items():
        tot = {arm: sum(cost(t, p) for t in U.get(arm, {}).values()) for arm in ("B0", "B1", "C")}
        s1c = a.s1_cost_per_q * n
        c_b1, c_c = tot["B1"] + s1c, tot["C"] + s1c
        out["price_sensitivity"][name] = {"per_q": {"B0": round(tot["B0"] / n, 5), "B1": round(c_b1 / n, 5), "C": round(c_c / n, 5)},
                                          "saving_vs_B1": round(1 - c_c / c_b1, 4), "saving_vs_B0": round(1 - c_c / tot["B0"], 4)}
    out["agent_cost_only"] = {arm: round(sum(cost(t, PRICES["opus_list"]) for t in U.get(arm, {}).values()) / n, 5) for arm in ("B0", "B1", "B2", "C")}
    out["s1_share_of_C_cost"] = round(a.s1_cost_per_q * n / (sum(cost(t, PRICES["opus_list"]) for t in U["C"].values()) + a.s1_cost_per_q * n), 4)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
