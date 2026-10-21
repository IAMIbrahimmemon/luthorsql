"""Pool candidate files from several slot-model checkpoints into one list per question (round 7).

Different checkpoints get different questions right (BIRD dev: the full run's first pick 39.7%, either the
column-mention run's or the full run's 47.3%), so the validator gets more right answers to choose from.
Each checkpoint is a LoRA adapter on the same backbone: at serve time this is one more slot pass per adapter.

  python scripts/merge_cands.py --inputs full.jsonl col.jsonl --k 32 --out pooled.jsonl

The first input is the primary model: its greedy plan stays first (so "greedy" and the fallback mean the same
thing). Then candidates are taken round-robin by rank across inputs, repeats (same SQL) dropped, up to k.
Each kept candidate is tagged with src = index of the input it came from.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    files = [{(r := json.loads(l))["id"]: r for l in open(f)} for f in a.inputs]
    out, n_ok, n_ok0 = [], 0, 0
    for qid, base in files[0].items():
        lists = [f[qid]["cands"] if qid in f else [] for f in files]
        seen, merged = set(), []
        for rank in range(max(map(len, lists))):
            for src, cl in enumerate(lists):
                if rank < len(cl) and len(merged) < a.k:
                    c = cl[rank]
                    key = c.get("sql") or f"none:{src}:{rank}"
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append({**c, "rank": len(merged), "src_model": src})
        out.append({**base, "cands": merged})
        n_ok += any(c.get("ok") for c in merged)
        n_ok0 += any(c.get("ok") for c in lists[0])
    a.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"MERGED {len(out)} questions: oracle primary {n_ok0}, pooled {n_ok}")


if __name__ == "__main__":
    main()
