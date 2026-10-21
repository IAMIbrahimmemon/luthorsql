"""Phase 3 (v4desc): an upweighting plan set for arithmetic with numeric constants (the division-by-constant blind spot,
docs/paper_notes/research/generator_misses_2026-10-10.md: only 37 of ~4,500 covered BIRD-train golds contain `/ number`).

Reads grammar-v4 plan sets (train data only) and copies every verified plan row whose compiled SQL does arithmetic with a
numeric constant into one plan dir (plans.jsonl + the cards it needs), for train_sq2.py --extra-train DIR:WEIGHT.

  uv run python scripts/build_div_extra.py                     # -> data/plans_v4/div_extra/{plans.jsonl,cards/,stats.json}

Classes (string literals are blanked first, so '2011/01/01' or LIKE '%/%' never count):
  div_const   `/ <number>` anywhere (÷100, ÷N, incl. `* 100 / 3`)            -> kept
  mul_const   `* <number>` with no division in the same SELECT (bare ×100, ×N)  -> kept
  pct_ratio   `* 100 / <expr>` / `100.0 * x / y` (percentage over a non-constant: the generator already gets these
              right, generator_misses §1) -> counted only, NOT kept
Never kept: rows on --exclude-dbs (default: the 9 held-out BIRD-train DBs data/sys1/dev_dbs_bird.json, which hold
DECISION and RESERVED) and --drop-dbs (data/sys1/contaminated_dbs.json); rows that are not verified plans
(outcome != plan) or have an empty result (train_sq2 strict loading skips those anyway). BIRD dev is never read: the plan
sets here are BIRD-train / SynSQL train only, and train_sq2 --extra-train also drops rows on the gate's databases.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = r"-?\d+(?:\.\d+)?"
STR = re.compile(r"'(?:[^']|'')*'")
DIV_CONST = re.compile(rf"/\s*(?:{NUM}\b(?!\s*[-+*/(.])|\(\s*{NUM}\s*\)|CAST\s*\(\s*{NUM}\s+AS\s+\w+\s*\))", re.I)
MUL_CONST = re.compile(rf"\*\s*\(?\s*{NUM}\b|\b{NUM}\s*\*", re.I)
PCT = re.compile(rf"(\*\s*{NUM}\s*/)|(\b100(?:\.0+)?\s*\*.*?/)", re.I | re.S)


def classify(sql: str) -> str | None:
    s = STR.sub("''", sql or "")
    s = re.sub(r'"[^"]*"', "c", s)          # quoted identifiers (column names with digits)
    if DIV_CONST.search(s):
        return "div_const"
    if MUL_CONST.search(s):
        if "/" in s:
            return "pct_ratio" if PCT.search(s) else "mul_const"
        return "mul_const"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", type=Path, nargs="+",
                    default=[ROOT / "data/plans_v4/train/bird_train", ROOT / "data/plans_v4/train_big/synsql_train"])
    ap.add_argument("--exclude-dbs", type=Path, default=ROOT / "data/sys1/dev_dbs_bird.json")
    ap.add_argument("--drop-dbs", type=Path, default=ROOT / "data/sys1/contaminated_dbs.json")
    ap.add_argument("--keep", nargs="+", default=["div_const", "mul_const"])
    ap.add_argument("--out", type=Path, default=ROOT / "data/plans_v4/div_extra")
    a = ap.parse_args()
    for p in a.plans:
        if "bird_dev" in str(p) or "sealed" in str(p):
            raise SystemExit(f"refusing evaluation plan set {p}")
    excl = set(json.loads(a.exclude_dbs.read_text()))
    d = json.loads(a.drop_dbs.read_text()) if a.drop_dbs and a.drop_dbs.exists() else []
    excl |= set(d.get("dbs", []) if isinstance(d, dict) else d)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "cards").mkdir(exist_ok=True)
    counts: dict[str, Counter] = {}
    kept, need = [], {}
    for pdir in a.plans:
        c = counts.setdefault(pdir.name, Counter())
        with open(pdir / "plans.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r.get("outcome") != "plan" or r.get("empty_result"):
                    continue
                k = classify(r.get("compiled_sql") or "")
                if k is None:
                    continue
                twin = "twin" if r.get("twin_of") else "base"
                c[f"{k}:{twin}"] += 1
                if r["db_id"] in excl:
                    c[f"{k}:excluded_db"] += 1
                    continue
                if k in a.keep:
                    r = dict(r, source=f"{r.get('source', '?')}_divx")  # its own provenance line in train_mix.json
                    kept.append(r)
                    need[r["db_id"]] = pdir / "cards" / f"{r['db_id']}.json"
                    c[f"{k}:kept"] += 1
    with open(a.out / "plans.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    for db, src in need.items():
        shutil.copyfile(src, a.out / "cards" / f"{db}.json")
    stats = {"plans": [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in a.plans],
             "excluded_dbs": sorted(excl), "keep": a.keep, "rows": len(kept), "dbs": len(need),
             "counts": {k: dict(sorted(v.items())) for k, v in counts.items()}}
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
