"""Grammar v4 training plans: scripts/prepare_plans.py run with LUTHOR_GRAMMAR=v4 (CPU only, no model).

Same arguments as prepare_plans.py; the output folder layout is the same (<out>/<format>_<split>/{plans.jsonl,
stats.json, cards/}), so point --out at a NEW folder (data/plans_v4/...), never at a grammar v3 one. A manifest
(code revision, grammar, command) is written next to stats.json.

  python scripts/convert_v4.py --format bird   --root data/raw/bird_train  --split train --out data/plans_v4/train
  python scripts/convert_v4.py --format spider --root data/raw/spider_data --split train --out data/plans_v4/train
  python scripts/convert_v4.py --format synsql --root data/raw/synsql      --split train --out data/plans_v4/train_big \
      --limit 600000 --unsup-sample 0
  python scripts/convert_v4.py rejoin data/rare1 data/plans_v4/rare1   # plans not converted from gold SQL

Grammar v4.1 adds a second condition to a conditional aggregate (Proj.cond2). prepare_plans._ground_plan grounds
it natively (since 2026-10-08), like Proj.cond: a candidate the model can point at, rewritten to the candidate's
value, else the plan is dropped as ungrounded_value. The wrapper below is only a fallback for a prepare_plans.py
without that support (detected from its source; the manifest's cond2_grounding says which one ran).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if os.environ.get("LUTHOR_GRAMMAR", "v4") != "v4":
    sys.exit("convert_v4.py writes grammar v4 plans: unset LUTHOR_GRAMMAR or set it to v4")
os.environ["LUTHOR_GRAMMAR"] = "v4"  # before any luthor import; inherited by the worker processes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import prepare_plans as pp  # noqa: E402
from luthor.sq import plan as _plan  # noqa: E402
from luthor.sq.candidates import ground  # noqa: E402

assert _plan.GRAMMAR_V4

_orig_ground_plan = pp._ground_plan


def _ground_plan_v4(plan, cands, value_labels, top):
    if not _orig_ground_plan(plan, cands, value_labels, top):  # sub-plans / set-op blocks recurse through here
        return False
    for pr in plan.projections:
        cf = getattr(pr, "cond2", None)
        if cf is None:
            continue
        idx = [ground(v, cf.col.id, cands) for v in cf.values]
        if any(i is None for i in idx):
            return False
        cf.values = [cands[i].value for i in idx]
    return True


NATIVE_COND2 = "cond2" in inspect.getsource(_orig_ground_plan)
if not NATIVE_COND2:  # module level: spawned workers re-run it
    pp._ground_plan = _ground_plan_v4


def _manifest(argv: list[str], t0: float) -> None:
    ap = argparse.ArgumentParser(add_help=False)
    for k in ("--out", "--format", "--split"):
        ap.add_argument(k, required=True)
    a = ap.parse_known_args(argv)[0]
    out = Path(a.out) / f"{a.format}_{a.split}"
    if not out.exists():
        return
    rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src/luthor/sq", "src/luthor/sq2",
                            "scripts/prepare_plans.py"], capture_output=True, text=True).stdout.split("\n")
    (out / "manifest.json").write_text(json.dumps({
        "grammar": "v4", "converter_features": sorted(__import__("luthor.sq.convert", fromlist=["V4"]).V4),
        "join_hint_mode": _plan.JOIN_HINT_MODE, "cond2_grounding": "convert_v4 wrapper"
        if pp._ground_plan is _ground_plan_v4 else "prepare_plans", "git_head": rev,
        "uncommitted_grammar_files": [d for d in dirty if d], "command": ["scripts/convert_v4.py"] + argv,
        "seconds": round(time.time() - t0)}, indent=1))


def rejoin(src: Path, out: Path, db_root: Path) -> None:
    """Plans that were not converted from gold SQL (data/rare1: mutants of grammar v3 plans with written
    questions) keep their v3 join_cols, an unordered column set, which the v4 join slots would read as ordered
    (left, right) pairs. Re-derive the pairs from each row's compiled_sql with the v4 converter, recompile, and
    keep the row only if the v4 SQL returns the same rows; everything else in the plan (grounded values, labels)
    is unchanged."""
    import shutil
    import sqlite3
    from luthor.sq.convert import convert
    from luthor.sq.plan import CompileError, Plan, compile_plan
    from luthor.sq.schema_card import SchemaCard
    t0, stats, rows, cards, cons = time.time(), __import__("collections").Counter(), [], {}, {}
    (out / "cards").mkdir(parents=True, exist_ok=True)
    for line in open(src / "plans.jsonl"):
        r = json.loads(line)
        if r.get("outcome") != "plan":
            rows.append(r)
            stats["kept_not_plan"] += 1
            continue
        db = r["db_id"]
        if db not in cards:
            cards[db] = SchemaCard.load(src / "cards" / f"{db}.json")
            shutil.copy(src / "cards" / f"{db}.json", out / "cards" / f"{db}.json")
            cons[db] = sqlite3.connect(f"file:{db_root / db / (db + '.sqlite')}?mode=ro", uri=True)
            cons[db].text_factory = lambda b: b.decode("utf-8", "replace")
        g, plan = cards[db].graph(), Plan.from_dict(r["plan"])
        if plan.join_cols:
            c = convert(r["compiled_sql"], g)
            if c.plan is None:
                stats["drop: compiled_sql does not convert under v4"] += 1
                continue
            plan.join_cols = c.plan.join_cols
        try:
            sql = compile_plan(plan, g)
            same = pp.norm_rows(pp.run(cons[db], sql)) == pp.norm_rows(pp.run(cons[db], r["compiled_sql"]))
        except (CompileError, sqlite3.Error) as e:
            stats[f"drop: {type(e).__name__}"] += 1
            continue
        if not same:
            stats["drop: v4 join changes the rows"] += 1
            continue
        stats["plan_ok" if sql == r["compiled_sql"] else "plan_ok_sql_differs_rows_same"] += 1
        rows.append({**r, "plan": plan.to_dict(), "compiled_sql": sql})
    with open(out / "plans.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    summary = {"rows_in": sum(stats.values()), "rows_out": len(rows), "stats": dict(stats.most_common()),
               "source": str(src), "seconds": round(time.time() - t0)}
    (out / "stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    t0 = time.time()
    if sys.argv[1:2] == ["rejoin"]:  # convert_v4.py rejoin <src plans dir> <out dir> [<db root>]
        a = sys.argv[2:]
        rejoin(Path(a[0]), Path(a[1]), Path(a[2]) if len(a) > 2 else ROOT / "data/raw/bird_train/train_databases")
    else:
        pp.main()
        _manifest(sys.argv[1:], t0)
