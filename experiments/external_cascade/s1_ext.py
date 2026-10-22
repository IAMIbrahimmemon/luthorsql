"""System 1 rows and agent chunks for the external cascade test (PREREG.md), from the gold-free package run.

Same margin and top-8 result groups as the RESERVED test (experiments/cost_confirmation/s1_from_package.py). No gold
is read. Writes runs/phase2/external_s1/s1_external.jsonl and the sandbox (chunks B0/B1/C, empty answers, databases
with their description CSVs, the SQL tool).

  python3 experiments/external_cascade/s1_ext.py --cands <predict_external.json.cands.jsonl> --box <sandbox>
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cost_confirmation"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from cascade_threshold import margin  # noqa: E402
from s1_from_package import chunks, groups  # noqa: E402

T = 3.0
BLIND = ROOT / "data/sys1/external_blind"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/phase2/external_s1/s1_external.jsonl")
    ap.add_argument("--tools", type=Path, required=True, help="the SQL tool folder used in the RESERVED sandbox")
    a = ap.parse_args()
    idx = json.load(open(BLIND / "index_to_id.json"))
    rows = []
    for ln in open(a.cands):
        r = json.loads(ln)
        qid = idx[int(r["index"])]
        m, _ = margin(r["cands"])
        rows.append({"id": qid, "set": qid.rsplit("_", 1)[0], "db_id": r["db_id"], "question": r["question"], "evidence": "",
                     "sql": r.get("pick"), "margin": None if m == -math.inf else round(m, 4), "candidates": groups(r["cands"])})
    assert len(rows) == len(idx) == len({r["id"] for r in rows}), "every question exactly once"
    a.out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    routed = [r for r in rows if r["margin"] is None or r["margin"] < T]
    for s in sorted({r["set"] for r in rows}):
        n = sum(r["set"] == s for r in rows); k = sum(r["set"] == s for r in routed)
        print(f"{s}: {n} questions, routed at t={T}: {k} ({k / n:.4f}); no margin {sum(r['set'] == s and r['margin'] is None for r in rows)}")
    print(f"pooled: routed {len(routed)}/{len(rows)} ({len(routed) / len(rows):.4f})")
    shutil.copytree(BLIND / "databases", a.box / "dbs", dirs_exist_ok=True)
    shutil.copytree(a.tools, a.box / "TOOLS", dirs_exist_ok=True)
    (a.box / "work").mkdir(parents=True, exist_ok=True)
    for arm, rs, cand in (("B0", rows, False), ("B1", rows, True), ("C", routed, True)):
        d = a.box / "chunks" / arm
        d.mkdir(parents=True, exist_ok=True)
        (a.box / "answers" / arm).mkdir(parents=True, exist_ok=True)
        n = 0
        for name, ch in chunks(rs):
            d.joinpath(f"{name}.jsonl").write_text("".join(json.dumps(
                {"question_id": r["id"], "db_id": r["db_id"], "question": r["question"], "evidence": ""}
                | ({"candidates": r["candidates"]} if cand else {})) + "\n" for r in ch))
            n += 1
        print(arm, n, "chunks")


if __name__ == "__main__":
    main()
