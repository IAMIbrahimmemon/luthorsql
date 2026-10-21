"""Rare-skeleton training data, step 1 (Mac, CPU): pick BIRD-train plan mutants whose skeleton is rare in
BIRD train, for scripts/qwrite.py to write questions for (docs/SYSTEM1_FANOUT_R1.md section 2.4).

  python scripts/build_rare_mix.py select [--max-support 20] [--cap 10000] [--out data/rare1]

Inputs (all public BIRD-train material):
  - mutants: data/rare1/mutants.jsonl, the round-1 LLM-free plan mutants of BIRD-train seed plans
    (reports/fanout_r1/data/r1/mutate.py). Regenerated with PYTHONHASHSEED=0 if missing (~1 h on the M3).
  - support: data/sys1/bird_train/plans.jsonl, the BIRD-train rows sys1_full trained on (non-empty plans,
    twins included, held-out training dbs excluded): exactly the "bird" support source of
    reports/fanout_r1/data/r2/support.py, at its level L1 (the level of the 28.3% / 79% finding).
Kept: BIRD-train dbs only, never a BIRD dev db, a sealed-set db or one of the 9 held-out training dbs
(data/sys1/dev_dbs_bird.json, sys1's checkpoint-selection dbs); L1 support < --max-support; result
non-empty and non-trivial, re-executed under 5 s; not implausible-by-key (SUM/AVG/MIN/MAX or a range filter
on an id / primary-key column, or an added id column); then balanced across skeletons and databases.
Output: <out>/selected.jsonl (compiled SQL, result signature, sample rows, touched-table schema snippet)
and <out>/select_stats.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sqlw.execute import connect, signature  # noqa: E402
from luthor.sqlw.prompt import table_ddl, used_tables  # noqa: E402

MUTATE = REPO / "reports/fanout_r1/data/r1/mutate.py"
SUPPORT_PLANS = REPO / "data/sys1/bird_train/plans.jsonl"
CARDS = REPO / "data/sys1/bird_train/cards"
HELD_DBS = REPO / "data/sys1/dev_dbs_bird.json"        # sys1's held-out BIRD-train dbs (checkpoint selection)
BIRD_DEV_DBS = REPO / "data/sys1/bird_dev_dbs.json"    # the 11 BIRD dev databases (names only)
SEALED_DBS = REPO / "data/sys1/sealed/dbs"             # sealed-set databases: directory NAMES only, nothing read
BIRD_DEV_11 = {"california_schools", "card_games", "codebase_community", "debit_card_specializing", "european_football_2",
               "financial", "formula_1", "student_club", "superhero", "thrombosis_prediction", "toxicology"}
SAMPLE_ROWS = 5


# ---- skeletons: copied from reports/fanout_r1/data/r2/support.py (level L1 there) -----------------------------
def plan_tables(p: dict) -> set:
    t = {x["col"]["table"] for x in p["projections"]} | {f["col"]["table"] for f in p["filters"]}
    t |= {g["table"] for g in p.get("group_by") or []}
    if p.get("order"):
        t.add(p["order"]["col"]["table"])
    t |= {c["table"] for c in p.get("join_cols") or []}
    t |= set(p.get("require") or [])
    t.discard(None)
    return t


def skeleton(p: dict) -> str:
    """support.py skel(p, 1): counts/aggregates/ops/tables/group/order/limit/distinct/compose/set-op/having."""
    P, F = p["projections"], p["filters"]
    lim = "" if p.get("limit") is None else ("L1" if p["limit"] == 1 else "Ln")
    return str((len(P), tuple(sorted(x["agg"] for x in P)), len(F), tuple(sorted(f["op"] for f in F)),
                min(len(plan_tables(p)), 4), len(p.get("group_by") or []), p.get("order_dir", "none"), lim,
                bool(p.get("distinct")), p.get("compose", "none"), p.get("set_op", "none"), len(p.get("having") or [])))


def bird_support(path: Path = SUPPORT_PLANS, held: set | None = None) -> collections.Counter:
    held = held if held is not None else set(json.loads(HELD_DBS.read_text()))
    c = collections.Counter()
    for line in open(path):
        r = json.loads(line)
        if r.get("outcome", "plan") != "plan" or r.get("empty_result") or r["db_id"] in held:
            continue
        c[skeleton(r["plan"])] += 1
    return c


def support_bucket(n: int) -> str:
    return "0" if n == 0 else "1-4" if n < 5 else "5-19" if n < 20 else "20-99" if n < 100 else "100+"


# ---- the naturalness proxy of reports/fanout_r1/data/r1/mut_summary.py --------------------------------------
_KEYS: dict = {}


def _iskey(db: str, t: str, c: str) -> bool:
    if c == "*":
        return False
    if db not in _KEYS:
        d = json.loads((CARDS / f"{db}.json").read_text())
        _KEYS[db] = {(x["name"], y["name"]): y.get("pk") for x in d["tables"] for y in x["columns"]}
    return bool(_KEYS[db].get((t, c))) or re.search(r"(^|_)id$|id$", c.lower()) is not None


def implausible(m: dict) -> bool:
    p, db = m["plan"], m["db_id"]
    return any(x["agg"] in ("sum", "avg", "min", "max") and _iskey(db, x["col"]["table"], x["col"]["column"])
               for x in p["projections"]) \
        or any(f["op"] in (">", "<", ">=", "<=") and _iskey(db, f["col"]["table"], f["col"]["column"]) for f in p["filters"]) \
        or (m.get("op") == "proj_add" and any(x["agg"] == "none" and _iskey(db, x["col"]["table"], x["col"]["column"])
                                             for x in p["projections"][-1:]))


# ---- databases ---------------------------------------------------------------------------------------------
def forbidden_dbs() -> set:
    bad = set(BIRD_DEV_11)
    if BIRD_DEV_DBS.exists():
        bad |= set(json.loads(BIRD_DEV_DBS.read_text()))
    if SEALED_DBS.is_dir():  # names of the sealed-set database folders; no file inside is opened
        bad |= {p.name for p in SEALED_DBS.iterdir() if p.is_dir()}
    return bad


def db_paths(roots) -> dict[str, str]:
    out = {}
    for root in roots:
        for p in Path(root).rglob("*.sqlite"):
            out.setdefault(p.stem, str(p))
    return out


def schema_snippet(card: SchemaCard, sql: str) -> tuple[list[str], str]:
    tabs = used_tables(card, sql)
    by = {t.name: t for t in card.tables}
    return tabs, "\n\n".join(table_ddl(card, by[t]) for t in tabs)


def _cell(v):
    if isinstance(v, float):
        v = round(v, 4)
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "..."


def exec_task(task):
    """(db path, [(id, sql)], seconds) -> [(id, status, n_rows, signature, sample, seconds)]."""
    path, items, seconds = task
    run = connect(path, seconds)
    out = []
    for mid, sql in items:
        t0 = time.time()
        try:
            rows = run(sql)
        except Exception as e:
            out.append((mid, "error:" + str(e)[:80], 0, "", [], time.time() - t0))
            continue
        dt = time.time() - t0
        trivial = len(rows) == 0 or all(all(v in (None, 0) for v in r) for r in rows)
        out.append((mid, "trivial" if trivial else "ok", len(rows), signature(rows),
                    [[_cell(v) for v in r] for r in rows[:SAMPLE_ROWS]], dt))
    return out


def balance(pool: list[dict], cap: int, per_skel: int, per_db: int, seed: int = 0) -> list[dict]:
    """Round-robin over skeletons (rarest first), each skeleton taking from its least-used database next,
    under per-skeleton and per-database caps, until `cap` rows."""
    rnd = random.Random(seed)
    by_skel = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in pool:
        by_skel[m["skel"]][m["db_id"]].append(m)
    for dbs in by_skel.values():
        for lst in dbs.values():
            rnd.shuffle(lst)
    order = sorted(by_skel, key=lambda s: (pool_support(by_skel[s]), s))
    n_db, n_sk, out = collections.Counter(), collections.Counter(), []
    progress = True
    while len(out) < cap and progress:
        progress = False
        for s in order:
            if len(out) >= cap:
                break
            if n_sk[s] >= per_skel:
                continue
            dbs = [d for d, lst in by_skel[s].items() if lst and n_db[d] < per_db]
            if not dbs:
                continue
            d = min(dbs, key=lambda d: (n_db[d], d))
            out.append(by_skel[s][d].pop())
            n_db[d] += 1
            n_sk[s] += 1
            progress = True
    return out


def pool_support(dbs) -> int:
    return next(m["bird_support"] for lst in dbs.values() for m in lst)


def select(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mut_path = Path(a.mutants)
    if not mut_path.exists():
        print(f"{mut_path} missing: regenerating with mutate.py (PYTHONHASHSEED=0)", flush=True)
        env = dict(os.environ, PYTHONHASHSEED="0")
        with open(mut_path.with_suffix(".log"), "w") as log:
            subprocess.run([sys.executable, "-I", str(MUTATE), str(mut_path), "--workers", str(a.workers)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    muts = [json.loads(l) for l in open(mut_path)]
    held = set(json.loads(HELD_DBS.read_text()))
    bad = forbidden_dbs()
    sup = bird_support(held=held)
    train_dbs = {p.stem for p in CARDS.glob("*.json")} - {"train_tables"}
    paths = db_paths(a.db_roots)
    allow = set(paths)
    if a.db_allow:
        allow &= {p.name for p in Path(a.db_allow).iterdir() if p.is_dir()}
    st = collections.Counter(total=len(muts))
    for m in muts:
        m["skel"] = skeleton(m["plan"])
        m["bird_support"] = sup[m["skel"]]
    before = collections.Counter(support_bucket(m["bird_support"]) for m in muts)
    pool = []
    for m in muts:
        db = m["db_id"]
        assert db in train_dbs, f"mutant on a non-BIRD-train db: {db}"
        if db in bad:
            st["drop:forbidden_db"] += 1
        elif db in held:
            st["drop:held_out_db"] += 1
        elif db not in allow:
            st["drop:db_not_available"] += 1
        elif m["bird_support"] >= a.max_support:
            st["drop:common_skeleton"] += 1
        elif implausible(m):
            st["drop:implausible_key"] += 1
        else:
            pool.append(m)
    st["candidates"] = len(pool)
    print(f"{len(muts)} mutants, {len(pool)} rare-skeleton candidates; executing", flush=True)
    by_db = collections.defaultdict(list)
    for m in pool:
        by_db[m["db_id"]].append((m["id"], m["compiled_sql"]))
    tasks = []
    for db, items in by_db.items():
        for i in range(0, len(items), 50):
            tasks.append((paths[db], items[i:i + 50], a.seconds))
    res = {}
    with Pool(a.workers) as p:
        for part in p.imap_unordered(exec_task, tasks):
            for mid, status, n, sig, sample, dt in part:
                res[mid] = (status, n, sig, sample, dt)
    ok = []
    for m in pool:
        status, n, sig, sample, dt = res[m["id"]]
        if status != "ok":
            st["drop:exec_" + status.split(":")[0]] += 1
            continue
        if dt >= a.seconds:
            st["drop:slow"] += 1
            continue
        m.update(n_rows=n, sig=sig, sample_rows=sample, exec_s=round(dt, 3))
        ok.append(m)
    st["executed_ok"] = len(ok)
    chosen = balance(ok, a.cap, a.per_skel, a.per_db)
    cards = {}
    with open(out / "selected.jsonl", "w") as fh:
        for m in sorted(chosen, key=lambda m: m["id"]):
            db = m["db_id"]
            assert db not in bad and db not in BIRD_DEV_11 and db not in held
            card = cards.setdefault(db, SchemaCard.load(CARDS / f"{db}.json"))
            tabs, ddl = schema_snippet(card, m["compiled_sql"])
            fh.write(json.dumps({"id": m["id"], "seed_id": m["seed_id"], "db_id": db, "op": m["op"], "skel": m["skel"],
                                 "bird_support": m["bird_support"], "plan": m["plan"], "compiled_sql": m["compiled_sql"],
                                 "sig": m["sig"], "n_rows": m["n_rows"], "sample_rows": m["sample_rows"],
                                 "exec_s": m["exec_s"], "tables": tabs, "schema": ddl}, default=str) + "\n")
    after = collections.Counter(support_bucket(m["bird_support"]) for m in chosen)
    skel_n = collections.Counter(m["skel"] for m in chosen)
    stats = {"args": {k: str(v) for k, v in vars(a).items()}, "counts": dict(st), "selected": len(chosen),
             "skeletons_selected": len(skel_n), "skeletons_in_pool": len({m["skel"] for m in ok}),
             "dbs_selected": len({m["db_id"] for m in chosen}),
             "per_db": dict(collections.Counter(m["db_id"] for m in chosen).most_common()),
             "rows_per_skeleton": dict(sorted(collections.Counter(skel_n.values()).items())),
             "support_bucket_all_mutants": dict(before), "support_bucket_selected": dict(after),
             "ops_selected": dict(collections.Counter(m["op"] for m in chosen))}
    (out / "select_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "per_db"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select"])
    ap.add_argument("--mutants", default=str(REPO / "data/rare1/mutants.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data/rare1"))
    ap.add_argument("--db-roots", nargs="+", default=[str(REPO / "data/raw/bird_train/train_databases")])
    ap.add_argument("--db-allow", default=str(REPO / "data/sys1/rrdbs"),
                    help="only dbs with a folder here (sys1/rrdbs: the 48 BIRD-train dbs the GPU box already has); '' = all")
    ap.add_argument("--max-support", type=int, default=20)
    ap.add_argument("--cap", type=int, default=10000)
    ap.add_argument("--per-skel", type=int, default=12)
    ap.add_argument("--per-db", type=int, default=400)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    select(a)


if __name__ == "__main__":
    main()
