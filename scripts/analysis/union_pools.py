"""Union of two candidate files (rerank_sq2 gen format) into one pool per question, as data/ens1/*_union.jsonl.

  python scripts/analysis/union_pools.py A.jsonl B.jsonl OUT.jsonl [--names r1 r2]

Candidates are deduplicated by SQL text. Each kept candidate gets src = name of its first file, or "both"; rank/cost
are from the first file that has it (they are only used for display; scoring re-reads the SQL). Questions must match.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--names", nargs=2, default=["r1", "r2"])
    x = ap.parse_args()
    A = {r["id"]: r for r in map(json.loads, open(x.a))}
    B = {r["id"]: r for r in map(json.loads, open(x.b))}
    assert A.keys() == B.keys(), f"question sets differ: {len(A.keys() ^ B.keys())} ids"
    n = 0
    with open(x.out, "w") as f:
        for i, ra in A.items():
            seen = {}
            for name, r in ((x.names[0], ra), (x.names[1], B[i])):
                for c in r["cands"]:
                    if c["sql"] in seen:
                        if seen[c["sql"]]["src"] != name:
                            seen[c["sql"]]["src"] = "both"
                    else:
                        seen[c["sql"]] = dict(c, src=name)
            f.write(json.dumps(dict(ra, cands=list(seen.values()))) + "\n")
            n += len(seen)
    print(f"{len(A)} questions, {n / len(A):.2f} candidates per question -> {x.out}")


if __name__ == "__main__":
    main()
