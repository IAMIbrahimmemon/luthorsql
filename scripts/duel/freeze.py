"""Freeze the gated-duel set on held-out v2 DECISION (memo 2026-10-10 §3 Step 2a, §4 item 4).

  uv run python scripts/duel/freeze.py

Per DECISION question in the wide (k=32) R1 ∪ R2 union pool: top-2 distinct live results by sum_z (best-scored
candidate per result signature `sig`, as llm_judge.shortlist). Gate: sum_z margin (top1 - top2) < 0.5.
Writes data/duel/decision_pairs.jsonl (all gated questions, with labels) + .sha256. No gold SQL is written.
Never reads RESERVED, sealed sets or BIRD dev.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from phase1_eval import IDS, SPLIT, load, rr_arr, z  # noqa: E402

assert SPLIT == "decision", "duel set is DECISION only"
W = ROOT / "runs/phase1/p1w/p1w"
OUT = ROOT / "data/duel/decision_pairs.jsonl"
GATE = 0.5


def top2(r1, r2):
    c = r2["cands"]
    assert [x["sql"] for x in r1["cands"]] == [x["sql"] for x in c]
    s = z(rr_arr(r1["cands"])) + z(rr_arr(c))
    best = {}
    for j, x in enumerate(c):
        if not (x.get("runs") and x.get("nonnull")):
            continue
        k = x.get("sig") or x["sql"]
        if k not in best or s[j] > s[best[k]]:
            best[k] = j
    top = sorted(best.values(), key=lambda j: -s[j])[:2]
    return top, s


def main():
    b1, b2 = load(W / "heldout_union_by_r1.jsonl"), load(W / "heldout_union_by_r2.jsonl")
    rows, n_two = [], 0
    for i in IDS:
        if i not in b2:
            continue
        top, s = top2(b1[i], b2[i])
        if len(top) < 2:
            continue
        n_two += 1
        m = float(s[top[0]] - s[top[1]])
        if m >= GATE:
            continue
        c = b2[i]["cands"]
        a, b = c[top[0]], c[top[1]]
        oka, okb = bool(a.get("ok")), bool(b.get("ok"))
        rows.append({"id": i, "db_id": b2[i]["db_id"], "pick_idx": top[0], "runner_idx": top[1],
                     "sql_pick": a["sql"], "sql_runner": b["sql"], "sig_pick": a.get("sig"), "sig_runner": b.get("sig"),
                     "margin": round(m, 6), "ok_pick": oka, "ok_runner": okb,
                     "label": ("pick" if oka and not okb else "runner" if okb and not oka
                               else "both_right" if oka else "both_wrong")})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    h = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (OUT.parent / (OUT.name + ".sha256")).write_text(f"{h}  {OUT.name}\n")
    from collections import Counter
    cnt = Counter(r["label"] for r in rows)
    print(f"DECISION n={len(IDS)}; with >=2 distinct live results: {n_two}; gated (margin < {GATE}): {len(rows)}")
    print(f"labels: {dict(cnt)}; exactly-one-right: {cnt['pick'] + cnt['runner']}")
    print(f"sha256 {h}")


if __name__ == "__main__":
    main()
