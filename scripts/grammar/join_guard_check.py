"""Join-route guard (LUTHOR_JOIN_GUARD=name, src/luthor/sq/plan.py): does it cost any gold plan? Mac CPU, no model.

For every verified gold plan (outcome == "plan") of a plan set, compile it in one process with the guard off (must give
the stored compiled_sql byte for byte, i.e. the current behaviour) and with each guard (same, name, direct). When a guard's SQL differs from the guard-off SQL, run
both against the SQLite database and compare row sets (prepare_plans.norm_rows, the converter's own check) with the
stored compiled SQL's rows (= the gold rows: the converter keeps a plan only when they match).

  uv run --offline python scripts/grammar/join_guard_check.py --grammar v4                 # bird, spider, rare1, div_extra
  uv run --offline python scripts/grammar/join_guard_check.py --grammar v4 --sets synsql   # 210k SynSQL plans (~10 min)
  uv run --offline python scripts/grammar/join_guard_check.py --grammar v3                 # v3 plans: byte identity

Writes runs/analysis/grammar_v4_fix/join_guard_check_<grammar>_<set>.json (counts + every changed plan's id / SQL).
The DECISION ids (data/sys1/heldout_v2/decision.json) are counted separately inside bird; no RESERVED split is read.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SETS = {  # name -> (plans dir, sqlite path template)
    "v4": {
        "bird": ("data/plans_v4/train/bird_train", "data/raw/bird_train/train_databases/{db}/{db}.sqlite"),
        "spider": ("data/plans_v4/train/spider_train", "data/raw/spider_data/database/{db}/{db}.sqlite"),
        "rare1": ("data/plans_v4/rare1", "data/raw/bird_train/train_databases/{db}/{db}.sqlite"),
        "div_extra": ("data/plans_v4/div_extra", None),
        "synsql": ("data/plans_v4/train_big/synsql_train", "data/raw/synsql/databases/{db}/{db}.sqlite"),
    },
    "v3": {
        "bird": ("data/sys1/bird_train", "data/raw/bird_train/train_databases/{db}/{db}.sqlite"),
        "spider": ("data/sys1/spider_train", "data/raw/spider_data/database/{db}/{db}.sqlite"),
        "synsql": ("data/sys1/synsql_train", "data/raw/synsql/databases/{db}/{db}.sqlite"),
    },
}
DEFAULT_SETS = {"v4": ["bird", "spider", "rare1", "div_extra"], "v3": ["bird", "spider"]}


def db_path(tmpl, row):
    db = row["db_id"]
    if tmpl:
        return ROOT / tmpl.format(db=db)
    for t in ("data/raw/bird_train/train_databases/{db}/{db}.sqlite", "data/raw/synsql/databases/{db}/{db}.sqlite"):
        if (ROOT / t.format(db=db)).exists():
            return ROOT / t.format(db=db)
    return None


def run(con, sql, timeout=20.0):
    deadline = time.perf_counter() + timeout
    con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 20_000)
    try:
        return con.execute(sql).fetchmany(200_001)
    finally:
        con.set_progress_handler(None, 0)


def norm_rows(rows):
    return {tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in rows}


GUARDS = ("same", "name", "direct")


def work(args):
    """One database: compile each plan with the guard off and with each guard; execute only when the SQL changed."""
    grammar, cards_dir, tmpl, db, rows, decision = args
    os.environ["LUTHOR_GRAMMAR"] = grammar
    sys.path.insert(0, str(ROOT / "src"))
    import luthor.sq.plan as pl
    from luthor.sq.schema_card import SchemaCard
    c = collections.Counter()
    changed = []
    try:
        graph = SchemaCard.load(Path(cards_dir) / f"{db}.json").graph()
    except Exception as e:  # noqa: BLE001
        c["card_error"] += len(rows)
        return c, [{"db": db, "error": repr(e)}]
    con = None
    ref_cache = {}

    def comp(plan, guard):
        pl.JOIN_GUARD = guard
        try:
            return pl.compile_plan(plan, graph)
        except pl.CompileError:
            return None
        finally:
            pl.JOIN_GUARD = "off"

    for r in rows:
        dec = r.get("id") in decision
        plan = pl.Plan.from_dict(r["plan"])
        c["plans"] += 1
        c["decision_plans"] += dec
        off = comp(plan, "off")
        c["off_equals_stored"] += off == r["compiled_sql"]
        rec = None
        for g in GUARDS:
            on = comp(plan, g)
            if on == off:
                continue
            c[f"{g}:sql_changed"] += 1
            c[f"{g}:decision_sql_changed"] += dec
            if rec is None:
                rec = {"id": r.get("id"), "db": db, "decision": dec, "off": off,
                       "join_cols": [f'{j["table"]}.{j["column"]}' for j in r["plan"].get("join_cols") or []]}
            if on is None:
                v = "compile_error"
            else:
                if con is None:
                    p = db_path(tmpl, r)
                    if p is None or not p.exists():
                        v = "no_db"
                        rec[g] = {"sql": on, "verdict": v}
                        c[f"{g}:v:{v}"] += 1
                        continue
                    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
                    con.text_factory = lambda b: b.decode("utf-8", "replace")
                try:
                    if r["compiled_sql"] not in ref_cache:
                        ref_cache.clear()
                        ref_cache[r["compiled_sql"]] = norm_rows(run(con, r["compiled_sql"]))
                    v = "same_rows" if norm_rows(run(con, on)) == ref_cache[r["compiled_sql"]] else "rows_differ"
                except sqlite3.Error as e:
                    v = f"exec_error:{type(e).__name__}"
            rec[g] = {"sql": on, "verdict": v}
            c[f"{g}:v:{v}"] += 1
            if dec:
                c[f"{g}:decision_v:{v}"] += 1
        if rec is not None:
            changed.append(rec)
    if con is not None:
        con.close()
    return c, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", choices=("v3", "v4"), required=True)
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-dbs", type=int, default=0, help="debug: first N databases per set")
    a = ap.parse_args()
    os.environ["LUTHOR_GRAMMAR"] = a.grammar  # inherited by the workers (read at their first luthor import)
    os.environ.pop("LUTHOR_JOIN_GUARD", None)  # workers switch plan.JOIN_GUARD themselves
    decision = set(json.load(open(ROOT / "data/sys1/heldout_v2/decision.json"))["ids"])
    out = ROOT / "runs/analysis/grammar_v4_fix"
    out.mkdir(parents=True, exist_ok=True)
    for name in a.sets or DEFAULT_SETS[a.grammar]:
        pdir, tmpl = SETS[a.grammar][name]
        by_db = collections.defaultdict(list)
        n_rows = 0
        with open(ROOT / pdir / "plans.jsonl") as f:
            for line in f:
                r = json.loads(line)
                n_rows += 1
                if r.get("outcome") == "plan" and r.get("plan") and r.get("compiled_sql"):
                    by_db[r["db_id"]].append({k: r.get(k) for k in ("id", "db_id", "plan", "compiled_sql")})
        dbs = sorted(by_db)[: a.limit_dbs or None]
        t0 = time.time()
        tot = collections.Counter()
        changed = []
        jobs = [(a.grammar, str(ROOT / pdir / "cards"), tmpl, d, by_db[d], decision) for d in dbs]
        with ProcessPoolExecutor(a.workers) as ex:
            for c, ch in ex.map(work, jobs, chunksize=8):
                tot.update(c)
                changed += ch
        del by_db
        res = {"grammar": a.grammar, "set": name, "plans_dir": pdir, "rows_in_file": n_rows, "dbs": len(dbs),
               "seconds": round(time.time() - t0, 1), "counts": dict(sorted(tot.items())), "changed": changed}
        (out / f"join_guard_check_{a.grammar}_{name}.json").write_text(json.dumps(res, indent=1))
        print(f"[{a.grammar} {name}] {json.dumps(res['counts'])}  ({res['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
