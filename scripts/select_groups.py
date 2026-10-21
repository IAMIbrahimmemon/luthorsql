"""Execution-grouped selection over scored candidate files (round 7; Mac, no GPU).

Candidates that return the same rows (same `sig`, written by rerank_sq2.py gen) form one group. Each group is
scored  R(g) = max validator score in g + beta * log|g| + gamma * max slot log-score in g  (R3-SQL / CHASE-style
grouping), and errors / empty results only win when nothing returns rows. beta and gamma are fitted on the
held-out TRAINING databases' file (--fit), never on BIRD dev, then applied unchanged to every --eval file.

  python scripts/select_groups.py --fit runs/.../rr32w/dev_scored.jsonl \
      --eval bird_dev=runs/.../rr32w/bird_dev_scored.jsonl spider_dk=... --totals bird_dev=1534 spider_dk=535
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
from pathlib import Path

BETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
GAMMAS = [0.0, 0.1, 0.25, 0.5, 1.0]


def pick(rec: dict, beta: float, gamma: float, grouped: bool = True) -> dict | None:
    cs = [c for c in rec["cands"] if c.get("sql") and "rr" in c]
    if not cs:
        return rec["cands"][0] if rec["cands"] else None
    good = [c for c in cs if c.get("runs") and c.get("nonnull")] or cs
    groups = collections.defaultdict(list)
    for i, c in enumerate(good):
        groups[(c.get("sig") or f"solo{i}") if grouped else f"solo{i}"].append(c)

    def score(g):
        return (max(c["rr"] for c in g) + beta * math.log(len(g)) + gamma * max(-c.get("cost", 0.0) for c in g))
    best = max(groups.values(), key=score)
    return max(best, key=lambda c: c["rr"])


def accuracy(recs: list[dict], beta: float, gamma: float, grouped: bool = True) -> int:
    return sum(bool((p := pick(r, beta, gamma, grouped)) and p.get("ok")) for r in recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=Path, required=True, help="scored candidates on held-out training databases")
    ap.add_argument("--eval", nargs="+", required=True, metavar="NAME=FILE")
    ap.add_argument("--totals", nargs="*", default=[], metavar="NAME=N", help="all questions in the set (all-question EX)")
    a = ap.parse_args()
    fit = [json.loads(l) for l in open(a.fit)]
    grid = {(b, g): accuracy(fit, b, g) for b, g in itertools.product(BETAS, GAMMAS)}
    (beta, gamma), best = max(grid.items(), key=lambda kv: (kv[1], -kv[0][0] - kv[0][1]))
    base = accuracy(fit, 0.0, 0.0, grouped=False)
    print(f"FIT on {a.fit.name} (n={len(fit)}): validator alone {base}, grouped best {best} at beta={beta} gamma={gamma}")
    totals = dict(t.split("=") for t in a.totals)
    for spec in a.eval:
        name, f = spec.split("=", 1)
        recs = [json.loads(l) for l in open(f)]
        n = int(totals.get(name, len(recs)))
        ora = sum(any(c.get("ok") for c in r["cands"]) for r in recs)
        v = accuracy(recs, 0.0, 0.0, grouped=False)
        g = accuracy(recs, beta, gamma)
        print(f"{name}: n={len(recs)} of {n}  oracle {ora / n:.1%}  validator {v / n:.1%}  grouped {g / n:.1%} "
              f"({g - v:+d} questions)")


if __name__ == "__main__":
    main()
