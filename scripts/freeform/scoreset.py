"""Score several merged pools of the same questions with ONE validator pass (CPU helpers; the scoring itself is
`scripts/rerank_sq2.py score`).

  python scripts/freeform/scoreset.py build --pools A.jsonl B.jsonl ... --out superset.jsonl
  python scripts/rerank_sq2.py score --rr V --cards C --dev-cands superset.jsonl --out superset_by_r1.jsonl ...
  python scripts/freeform/scoreset.py attach --scored superset_by_r1.jsonl --pools A.jsonl B.jsonl ... \
      --suffix _by_r1 --out-dir scored/

build: per question id, the first pool's record (pass the largest pool first) plus every candidate of the other pools
whose SQL text is not already there; any "rr" is dropped. The merged pools differ only in which ff / mut
representatives they add, so the superset holds each distinct SQL once.
attach: writes <out-dir>/<pool stem><suffix>.jsonl = each pool with rr = the superset's score of the same (id, SQL).
The validator scores each (question, SQL) on its own; the schema view (rerank_sq2.tables_for) depends on the
candidates scored in the same chunk, so a pool's scores can differ slightly from a pass over that pool alone (only for
schemas over the budget). Candidates with no SQL get no rr (rerank_sq2.unique skips them too).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(pools: list[Path], out: Path) -> dict:
    sup: dict[str, dict] = {}
    order: list[str] = []
    added = 0
    for p in pools:
        for r in map(json.loads, open(p)):
            if r["id"] not in sup:
                r = dict(r)
                r["cands"] = [{k: v for k, v in c.items() if k != "rr"} for c in r["cands"]]
                sup[r["id"]] = r
                order.append(r["id"])
                continue
            have = {c["sql"] for c in sup[r["id"]]["cands"] if c.get("sql")}
            for c in r["cands"]:
                if c.get("sql") and c["sql"] not in have:
                    sup[r["id"]]["cands"].append({k: v for k, v in c.items() if k != "rr"})
                    have.add(c["sql"])
                    added += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(sup[i]) + "\n" for i in order))
    n_c = sum(len(sup[i]["cands"]) for i in order)
    return {"records": len(order), "cands": n_c, "added_from_other_pools": added}


def attach(scored: Path, pools: list[Path], suffix: str, out_dir: Path) -> dict:
    rr = {}
    for r in map(json.loads, open(scored)):
        for c in r["cands"]:
            if c.get("sql") and "rr" in c:
                rr[(r["id"], c["sql"])] = c["rr"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for p in pools:
        n = miss = 0
        lines = []
        for r in map(json.loads, open(p)):
            for c in r["cands"]:
                c.pop("rr", None)
                if c.get("sql"):
                    v = rr.get((r["id"], c["sql"]))
                    if v is None:
                        miss += 1
                    else:
                        c["rr"] = v
                        n += 1
            lines.append(json.dumps(r) + "\n")
        dst = out_dir / f"{p.stem}{suffix}.jsonl"
        dst.write_text("".join(lines))
        stats[dst.name] = {"scored": n, "missing": miss}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "attach"])
    ap.add_argument("--pools", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--scored", type=Path)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args()
    if a.cmd == "build":
        print("SUPERSET " + json.dumps(build(a.pools, a.out)), flush=True)
    else:
        print("ATTACH " + json.dumps(attach(a.scored, a.pools, a.suffix, a.out_dir)), flush=True)


if __name__ == "__main__":
    main()
