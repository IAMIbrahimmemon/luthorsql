"""Secondary (descriptive, PREREG amendment 1): agent effort per question by set in the external test's B0 and C arms.

Tool calls, output tokens (max of logged and chars/4) and wall seconds per question, Spider-Realistic vs KaggleDBQA.
Confounded with difficulty; read only alongside P1/P2.

  .venv/bin/python experiments/memorisation_probe/effort_ext.py --box <s2ext sandbox> --tasks <transcripts> --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/external_cascade"))
import score_ext  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dbset = {}
    for s in score_ext.SETS:
        for r in map(json.loads, open(ROOT / f"data/eval/sealed_{s}.jsonl")):
            dbset[r["db_id"]] = s
    pat = re.compile(re.escape(str(a.box)) + r"/chunks/(B0|B1|C)/([A-Za-z0-9_]+_\d+)\.jsonl")
    agg = defaultdict(lambda: defaultdict(float))
    for f in glob.glob(str(a.tasks / "*.output")):
        m = pat.search(open(f).read(20000))
        if not m:
            continue
        arm, chunk = m.groups()
        nq = sum(1 for _ in open(a.box / "chunks" / arm / f"{chunk}.jsonl"))
        t = score_ext._one(f)
        calls = sum(1 for ln in open(f) for _ in re.finditer(r'"type": ?"tool_use"', ln))
        g = agg[(arm, dbset[chunk.rsplit("_", 1)[0]])]
        g["questions"] += nq
        g["chunks"] += 1
        g["tool_calls"] += calls
        g["output_tokens"] += max(t.get("output_tokens", 0), t.get("output_est", 0))
        g["seconds"] += t.get("seconds", 0.0)
    out = {f"{arm}/{s}": {"questions": int(g["questions"]), "chunks": int(g["chunks"]),
                          **{k + "_per_q": round(g[k] / g["questions"], 2) for k in ("tool_calls", "output_tokens", "seconds")}}
           for (arm, s), g in sorted(agg.items())}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
