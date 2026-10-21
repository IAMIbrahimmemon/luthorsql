"""Build the round-2 training mix (architecture v2, grammar v2).

  python scripts/build_mix2.py --out data/mix2

Sources (all public):
  human   plans_v2/{bird_train, spider_train, spider_syn_train}   repeated --real-repeat times
  wiki    plans_v2/wikisql_train                                   capped at --wiki-cap originals
  gen     data/qgen/{spider,bird}/{ds,claude}: generated questions on real public schemas.
          A question with a checker verdict (checked_NNNN.json) is kept only if the verdict
          matches its plan; one without passes only the wording filter (marked "unchecked").
Every generated plan is recompiled with today's join graph and re-executed; plans that fail,
time out, or return nothing (or only zeros / NULLs) are dropped.
Held out: --dev-dbs (training databases kept for checkpoint selection), and every database of
the sealed sets; training questions with >= 50% word overlap with a sealed question are dropped.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_plans import dedup_against  # noqa: E402
from qgen import grade_one, leaky  # noqa: E402

from luthor.sq.plan import Plan, compile_plan  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402

TR = Path("/Volumes/MacSSD/AI/datasets/text-to-sql/plans_v2")
EVAL = Path("data/eval_plans_v2")
DBS = {"spider": Path("data/raw/spider_data/database"), "bird": Path("data/raw/bird_train/train_databases")}
CARDS = {"spider": TR / "spider_train" / "cards", "bird": TR / "bird_train" / "cards"}


def _exec_check(args):
    """Recompile + execute one generated plan. Returns (key, sql) or (key, None)."""
    key, src, db, plan = args
    card = SchemaCard.load(CARDS[src] / f"{db}.json")
    try:
        sql = compile_plan(Plan.from_dict(plan), card.graph())
    except Exception:
        return key, None
    path = DBS[src] / db / f"{db}.sqlite"
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    deadline = time.time() + 5
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10_000)
    try:
        rows = con.execute(sql).fetchmany(1000)
    except Exception:
        return key, None
    finally:
        con.close()
    if not rows or all(all(v in (None, 0, 0.0, "") for v in r) for r in rows):
        return key, None
    return key, sql


def gen_rows(src: str, sub: str) -> tuple[list[dict], collections.Counter]:
    d = Path("data/qgen") / src / sub
    st = collections.Counter()
    plans = {json.loads(l)["plan_id"]: json.loads(l) for l in open(d / "plans_with_ids.jsonl")}
    out = []
    for f in sorted(d.glob("questions_*.json")):
        try:
            items = json.loads(f.read_text())["items"]
        except Exception:
            continue
        cf = f.with_name(f.name.replace("questions_", "checked_"))
        verdicts = {}
        if cf.exists():  # the checker's blind readings, by qid "<plan_id>_<k>"
            try:
                verdicts = {c["qid"]: c for c in json.loads(cf.read_text())["items"] if c.get("qid")}
            except Exception:
                verdicts = {}
        for it in items:
            if it.get("skip") or it.get("plan_id") not in plans:
                continue
            p = plans[it["plan_id"]]["plan"]
            lits = {str(v) for fl in p.get("filters") or [] for v in fl.get("values") or []}
            for k, q in enumerate(it.get("questions") or []):
                c = verdicts.get(f"{it['plan_id']}_{k}")
                if c is not None:
                    st["checked_total"] += 1
                    ok, why = grade_one(p, q, c)
                    if not ok:
                        st["checked_rejected"] += 1
                        continue
                    out.append({"plan_id": it["plan_id"], "question": q["text"], "style": q.get("style"), "checked": True})
                    continue
                st["unchecked_total"] += 1
                if leaky(q["text"], lits):
                    st["unchecked_leaky"] += 1
                    continue
                out.append({"plan_id": it["plan_id"], "question": q["text"], "style": q.get("style"), "checked": False})
    rows = []
    for k, r in enumerate(out):
        p = plans[r["plan_id"]]
        rows.append({"id": f"gen_{src}_{sub}_{r['plan_id']}_{k}", "source": f"gen_{src}", "db_id": p["db_id"],
                     "question": r["question"], "context": "", "outcome": "plan", "plan": p["plan"],
                     "checked": r["checked"], "style": r["style"], "empty_result": False, "_src": src,
                     "_plan_key": (src, sub, r["plan_id"])})
    return rows, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/mix2"))
    ap.add_argument("--dev-dbs", type=Path, default=Path("data/mix2_dev_dbs.json"))
    ap.add_argument("--real-repeat", type=int, default=2)
    ap.add_argument("--wiki-cap", type=int, default=30000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rnd = random.Random(0)
    dev_dbs = set(json.loads(a.dev_dbs.read_text()))

    eval_q, eval_dbs = [], set()
    for d in EVAL.iterdir():
        for l in open(d / "plans.jsonl"):
            r = json.loads(l)
            eval_q.append(r["question"])
            eval_dbs.add(r["db_id"])
    print(f"sealed: {len(eval_q)} questions, {len(eval_dbs)} databases", flush=True)

    stats = collections.Counter()
    a.out.mkdir(parents=True, exist_ok=True)
    cards_out = a.out / "cards"
    cards_out.mkdir(exist_ok=True)
    rows = []

    def keep_db(db):
        return db not in eval_dbs and db not in dev_dbs

    for name, kind in (("bird_train", "human"), ("spider_train", "human"), ("spider_syn_train", "human"),
                       ("wikisql_train", "wiki")):
        src = [json.loads(l) for l in open(TR / name / "plans.jsonl")]
        src = [r for r in src if r.get("outcome") == "plan" and not r.get("empty_result") and keep_db(r["db_id"])]
        if kind == "wiki":
            orig = [r for r in src if not r.get("twin_of")]
            rnd.shuffle(orig)
            keep = {r["id"] for r in orig[:a.wiki_cap]}
            src = [r for r in src if (r.get("twin_of") or r["id"]) in keep]
        for r in src:
            for c in {r["db_id"]}:
                f = TR / name / "cards" / f"{c}.json"
                if f.exists() and not (cards_out / f.name).exists():
                    shutil.copy(f, cards_out / f.name)
        reps = a.real_repeat if kind == "human" else 1
        stats[f"{name}"] = len(src)
        for k in range(reps):
            rows += [{**r, "id": f"{r['id']}#{k}" if k else r["id"]} for r in src]

    gen = []
    for src in ("spider", "bird"):
        for sub in ("ds", "claude"):
            if not (Path("data/qgen") / src / sub / "plans_with_ids.jsonl").exists():
                continue
            g, st = gen_rows(src, sub)
            for k, v in st.items():
                stats[f"gen_{src}_{sub}_{k}"] = v
            gen += [r for r in g if keep_db(r["db_id"])]
    # recompile + execute each distinct generated plan once
    keys = {}
    for r in gen:
        keys.setdefault(r["_plan_key"], (r["_src"], r["db_id"], r["plan"]))
    print(f"generated: {len(gen)} questions over {len(keys)} plans; executing...", flush=True)
    ok = {}
    with Pool(a.workers) as pool:
        for key, sql in pool.imap_unordered(_exec_check, [(k, *v) for k, v in keys.items()], chunksize=64):
            ok[key] = sql
    stats["gen_plans_ok"] = sum(v is not None for v in ok.values())
    stats["gen_plans_dropped"] = sum(v is None for v in ok.values())
    for r in gen:
        sql = ok.get(r["_plan_key"])
        if sql is None:
            continue
        r["compiled_sql"] = sql
        src = r.pop("_src")
        r.pop("_plan_key")
        f = CARDS[src] / f"{r['db_id']}.json"
        if not (cards_out / f.name).exists():
            shutil.copy(f, cards_out / f.name)
        rows.append(r)
    stats["gen_kept"] = sum(1 for r in rows if r["id"].startswith("gen_"))

    before = len(rows)
    rows = dedup_against(rows, eval_q)
    stats["dropped_eval_overlap"] = before - len(rows)
    rnd.shuffle(rows)
    with open(a.out / "plans.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    by_src = collections.Counter(r.get("source", "?") for r in rows)
    stats.update({f"rows_{k}": v for k, v in by_src.items()})
    stats["rows_total"] = len(rows)
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
