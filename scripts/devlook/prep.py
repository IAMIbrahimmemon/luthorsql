"""BIRD dev look #1, Mac side (CPU, < 1 GB RAM): build the BLIND box inputs. Prints counts only; never reads a label.

  .venv/bin/python scripts/devlook/prep.py pool-a      # R1's own k=32 BIRD dev list -> data/devlook/r1/poolA.jsonl
  .venv/bin/python scripts/devlook/prep.py questions   # questions + cards, no gold      -> data/devlook/r1/qs/
  .venv/bin/python scripts/devlook/prep.py plan        # repair trigger counts on pool A (label-free; for the estimate)
  .venv/bin/python scripts/devlook/prep.py audit F...  # refuse any label field in the files that will be pushed

pool-a: the R1 share (src in {r1, both}) of the p1w R1 u R2 BIRD dev pool, in R1's order -- exactly how VALHARD's
DECISION pool was built (scripts/valhard/pool.py --keep-src r1 both). It equals R1's own k=32 distinct list
(runs/phase1/p1w/p1w/r1/cands_bird_dev.jsonl) deduplicated by SQL (checked: same SQL in the same order for all 1,534).
Whitelisted fields only: question id / db_id / question / context; candidate rank / cost / sql / runs / nonnull / sig /
res (execution facts, not labels), src set to "k32". ok, gold_sql, gold_error, covered, in_slots and rr never travel and
are never read here (the whitelist copies named keys; nothing else of a row is touched).

questions: the rows rerank_sq2.py gen loads for BIRD dev with --all-questions (data/sys1/bird_dev/plans.jsonl rows that
are not twins, in file order, then dev.json questions with no plan row, question / evidence stripped, as
train_sq2.raw_rows does), reduced to id / db_id / question / context / outcome="missing" -- the decoder then sees
exactly the inputs of the p1w run (same text, same order, so the same chunks of 8), and the box has no plan or SQL.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P1W = ROOT / "runs/phase1/p1w/p1w"
DEV_JSON = ROOT / "data/bench/dev_20240627/dev.json"
DEV_DBS = ROOT / "data/bench/dev_20240627/dev_databases"
PLANS = ROOT / "data/sys1/bird_dev"
OUT = ROOT / "data/devlook/r1"
Q_KEYS = ("id", "db_id", "question", "context")
C_KEYS = ("rank", "cost", "sql", "runs", "nonnull", "sig", "res")
LABEL_KEYS = {"ok", "gold_sql", "gold_error", "covered", "in_slots", "compiled_sql", "plan", "SQL", "gold", "rr",
              "gold_rows", "gold_sig", "value_labels", "limit_label", "empty_result", "twin_of"}


def dev_ids() -> list[str]:
    """Question ids only (the gold column of dev.json is never touched)."""
    return [f"bird_dev_{q['question_id']}" for q in json.loads(DEV_JSON.read_text())]


def pool_a(a):
    ids = set(dev_ids())
    out, nc, seen = [], 0, set()
    for line in open(a.src):
        r = json.loads(line)
        assert r["id"] in ids and r["id"] not in seen, r["id"]
        seen.add(r["id"])
        cands = [{**{k: c[k] for k in C_KEYS if k in c}, "src": "k32"}
                 for c in r["cands"] if c.get("src") in ("r1", "both")]
        nc += len(cands)
        out.append({k: r[k] for k in Q_KEYS} | {"cands": cands})
    assert seen == ids, f"{len(ids - seen)} dev questions missing from {a.src}"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    audit_file(a.out)
    print(f"POOL_A {len(out)} questions, {nc} candidates ({nc / len(out):.2f} per question) -> {a.out}")


def questions(a):
    from collections import OrderedDict
    rows = OrderedDict()
    for line in open(a.plans / "plans.jsonl"):
        r = json.loads(line)
        if r.get("twin_of"):
            continue  # load_sets(strict=False) drops twins
        rows[r["id"]] = {"id": r["id"], "db_id": r["db_id"], "question": r["question"],
                         "context": r.get("context", ""), "outcome": "missing"}
    n_plan = len(rows)
    for i, q in enumerate(json.loads(DEV_JSON.read_text())):  # train_sq2.raw_rows: question / evidence / db_id only
        rid = f"bird_dev_{q.get('question_id', i)}"
        if rid not in rows and (a.plans / "cards" / f"{q['db_id']}.json").exists():
            rows[rid] = {"id": rid, "db_id": q["db_id"], "question": q["question"].strip(),
                         "context": (q.get("evidence") or "").strip(), "outcome": "missing"}
    assert set(rows) == set(dev_ids()), "question set differs from dev.json"
    out = a.out
    if out.exists():
        shutil.rmtree(out)
    (out / "cards").mkdir(parents=True)
    for p in sorted((a.plans / "cards").glob("*.json")):
        shutil.copy2(p, out / "cards" / p.name)
    (out / "plans.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    audit_file(out / "plans.jsonl")
    # order check against pool A (ids only): rerank_sq2 gen groups load order by database (first appearance)
    if a.check_order and a.check_order.exists():
        by_db = OrderedDict()
        for r in rows.values():
            by_db.setdefault(r["db_id"], []).append(r["id"])
        want = [i for v in by_db.values() for i in v]
        got = [json.loads(l)["id"] for l in open(a.check_order)]
        print(f"order vs {a.check_order.name}: {'SAME' if want == got else 'DIFFERENT'} "
              f"({sum(x == y for x, y in zip(want, got))}/{len(got)} positions agree)")
    print(f"QUESTIONS {len(rows)} ({n_plan} plan-set rows, {len(rows) - n_plan} dev.json-only), "
          f"{len(list((out / 'cards').glob('*.json')))} cards -> {out}")


def plan(a):
    """Repair trigger counts on a pool (label-free: execution facts + evidence parse), for the GPU estimate."""
    sys.path.insert(0, str(ROOT / "scripts/repair"))
    sys.path.insert(0, str(ROOT / "src"))
    import collections
    import repair_gen as RG
    c = collections.Counter()
    n = n_var = 0
    for line in open(a.pool):
        r = json.loads(line)
        n += 1
        cols = set(RG.EV.db_columns_path(str(a.dbs / r["db_id"] / f"{r['db_id']}.sqlite")))
        t = RG.triggers(r, cols, 8)
        c.update(t)
        if t:
            c["any"] += 1
            n_var += (4 if {"degenerate", "all_same", "no_live"} & set(t) else 0) + (3 if "evidence" in t else 0)
    print(json.dumps({"questions": n, "triggered": dict(c), "variant_runs_upper_bound": n_var}))


def audit_file(p: Path) -> int:
    bad = 0
    for line in open(p):
        r = json.loads(line)
        keys = set(r) | {k for c in r.get("cands", []) for k in c}
        if keys & LABEL_KEYS:
            print(f"AUDIT_FAIL {p}: label field(s) {sorted(keys & LABEL_KEYS)} (row {r.get('id')})")
            bad += 1
            break
    return bad


def audit(a):
    bad = 0
    for p in a.files:
        files = [p] if p.is_file() else sorted(p.rglob("*.jsonl"))
        for f in files:
            bad += audit_file(f)
    if bad:
        sys.exit(1)
    print(f"AUDIT_OK {len(a.files)} path(s): no label field")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool-a")
    p.add_argument("--src", type=Path, default=P1W / "bird_dev_union.jsonl")
    p.add_argument("--out", type=Path, default=OUT / "poolA.jsonl")
    q = sub.add_parser("questions")
    q.add_argument("--plans", type=Path, default=PLANS)
    q.add_argument("--out", type=Path, default=OUT / "qs")
    q.add_argument("--check-order", type=Path, default=OUT / "poolA.jsonl")
    pl = sub.add_parser("plan")
    pl.add_argument("--pool", type=Path, default=OUT / "poolA.jsonl")
    pl.add_argument("--dbs", type=Path, default=DEV_DBS)
    au = sub.add_parser("audit")
    au.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args()
    {"pool-a": pool_a, "questions": questions, "plan": plan, "audit": audit}[a.cmd](a)


if __name__ == "__main__":
    main()
