"""Turn the package's gold-free held-out run (predict_heldout.json.cands.jsonl) into System 1 rows and agent chunks.

Per question: System 1's pick (the package's answer), its margin (scripts/analysis/cascade_threshold.py:margin, top-2
distinct-result VALHARD gap over live candidates; None when it has no margin) and the top-8 result groups shown to the
agent (same format as scripts/analysis/hybrid/s1_candidates.py). No gold is read.

  python3 experiments/cost_confirmation/s1_from_package.py --cands <predict_heldout.json.cands.jsonl> \
      --index runs/phase2/cost_confirmation/index_to_id.json --out runs/phase2/cost_confirmation --box <sandbox>
"""
from __future__ import annotations

import argparse
import json
import math
import string
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from cascade_threshold import margin  # noqa: E402

T = 3.0


def groups(cands, top=8):
    lv = [x for x in cands if x.get("sql") and x.get("runs") and x.get("nonnull")] or [x for x in cands if x.get("sql")]
    g = {}
    for x in sorted(lv, key=lambda x: -x.get("rr", -1e9)):
        g.setdefault(x.get("sig") or x["sql"], []).append(x)
    return [{"label": string.ascii_uppercase[k], "sql": v[0]["sql"], "result": v[0].get("res", ""), "n_of_32": len(v)}
            for k, v in enumerate(list(g.values())[:top])]


def chunks(rows, size=25):
    by = defaultdict(list)
    for r in rows:
        by[r["db_id"]].append(r)
    for db, rs in sorted(by.items()):
        k = -(-len(rs) // size)
        n = -(-len(rs) // k)
        for j in range(k):
            yield f"{db}_{j + 1}", rs[j * n:(j + 1) * n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--box", type=Path, required=True)
    a = ap.parse_args()
    idx = json.load(open(a.index))
    split = {i: "reserved" for i in json.load(open(ROOT / "data/sys1/heldout_v2/reserved.json"))["ids"]}
    split.update({i: "decision" for i in json.load(open(ROOT / "data/sys1/heldout_v2/decision.json"))["ids"]})
    rows = {"reserved": [], "decision": []}
    for ln in open(a.cands):
        r = json.loads(ln)
        qid = idx[int(r["index"])]
        m, _ = margin(r["cands"])
        rows[split[qid]].append({"id": qid, "db_id": r["db_id"], "question": r["question"], "evidence": r.get("context") or "",
                                 "sql": r.get("pick"), "margin": None if m == -math.inf else round(m, 4),
                                 "candidates": groups(r["cands"])})
    for s, rs in rows.items():
        (a.out / f"s1_{s}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rs))
    res = rows["reserved"]
    routed = [r for r in res if r["margin"] is None or r["margin"] < T]
    print(f"reserved {len(res)}, decision {len(rows['decision'])}; reserved routed at t={T}: {len(routed)} "
          f"({len(routed) / len(res):.4f}); no margin {sum(r['margin'] is None for r in res)}")
    for arm, rs in (("B1", res), ("C", routed)):
        d = a.box / "chunks" / arm
        d.mkdir(parents=True, exist_ok=True)
        (a.box / "answers" / arm).mkdir(parents=True, exist_ok=True)
        n = 0
        for name, ch in chunks(rs):
            d.joinpath(f"{name}.jsonl").write_text("".join(json.dumps(
                {"question_id": r["id"], "db_id": r["db_id"], "question": r["question"], "evidence": r["evidence"],
                 "candidates": r["candidates"]}) + "\n" for r in ch))
            n += 1
        print(arm, n, "chunks")


if __name__ == "__main__":
    main()
