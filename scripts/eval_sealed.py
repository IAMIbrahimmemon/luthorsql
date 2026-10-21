"""Score a trained Luthor checkpoint on the SEALED sets, once, on the Mac (MLX 8-bit).

  python scripts/eval_sealed.py --run runs/sq4b_v1 --sets bird_dev spider_dev spider_test spider_syn_dev wikisql_test

Per set:
  covered EX   execution accuracy on questions the plan grammar covers (answered and right)
  all EX       the same over ALL questions in the set (uncovered ones count as wrong)
  abstain      share of grammar-unsupported questions the model correctly declines
  latency      per question, schema already cached (encode + heads + decode + compile)
Gold = the dataset's own gold SQL, executed on its SQLite database (BIRD's result-set rule).
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luthor.sq.model import LuthorSQ  # noqa: E402
from luthor.sq.runtime import MLXEncoder, Runtime  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402

PLANS = Path("data/eval_plans")   # sealed sets, kept on the Mac (not on the external drive)
RAW = Path("data/eval_raw")
DB_PATH = {
    "bird_dev": lambda db: Path(f"data/bench/dev_20240627/dev_databases/{db}/{db}.sqlite"),
    "spider_dev": lambda db: RAW / f"spider_data/database/{db}/{db}.sqlite",
    "spider_test": lambda db: RAW / f"spider_data/test_database/{db}/{db}.sqlite",
    "spider_syn_dev": lambda db: RAW / f"spider_data/database/{db}/{db}.sqlite",
    "wikisql_dev": lambda db: RAW / "wikisql/dev.db",
    "wikisql_test": lambda db: RAW / "wikisql/test.db",
}


class _HeadsOnly:
    """Stands in for the backbone when only the heads run in torch (the backbone is MLX)."""

    def __init__(self, hidden: int):
        self.hidden = hidden


def load_runtime(run: Path, bits: int = 8) -> Runtime:
    merged = run / "merged"
    enc = MLXEncoder(str(merged), bits=bits)
    cfg = json.loads((merged / "config.json").read_text())
    hidden = cfg.get("text_config", cfg)["hidden_size"]
    state = torch.load(merged / "best_heads.pt", map_location="cpu")
    d_proj = state["o_proj.weight"].shape[0]
    heads = LuthorSQ.__new__(LuthorSQ)
    torch.nn.Module.__init__(heads)
    LuthorSQ.__init__(heads, _HeadsOnly(hidden), d_proj=d_proj)
    heads.load_state_dict(state, strict=False)
    serving = json.loads((merged / "serving.json").read_text()) if (merged / "serving.json").exists() else {}
    return Runtime(heads.eval(), enc, serving=serving)


def run_sql(con, sql, timeout=10.0):
    deadline = time.time() + timeout
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10_000)
    try:
        return set(map(tuple, con.execute(sql).fetchmany(200_000)))
    finally:
        con.set_progress_handler(None, 0)


def evaluate_set(rt: Runtime, name: str, limit: int | None, seed: int = 0, dump=None) -> dict:
    d = PLANS / name
    stats = json.loads((d / "stats.json").read_text())
    rows = [json.loads(l) for l in open(d / "plans.jsonl")]
    rows = [r for r in rows if not r.get("twin_of")]  # originals only
    if limit and len(rows) > limit:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    by_db: dict[str, list] = {}
    for r in rows:
        by_db.setdefault(r["db_id"], []).append(r)
    res = {"plan_right": 0, "plan_n": 0, "unsup_right": 0, "unsup_n": 0, "errors": 0}
    lat = []
    t0 = time.time()
    for i, (db, rs) in enumerate(by_db.items()):
        card = SchemaCard.load(d / "cards" / f"{db}.json")
        rt.index(card)
        con = sqlite3.connect(f"file:{DB_PATH[name](db)}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode("utf-8", "replace")
        for r in rs:
            try:
                plan, sql, t = rt.plan(db, r["question"], r.get("context", ""))
            except Exception:
                res["errors"] += 1
                plan, sql, t = None, None, {"total_ms": 0}
            lat.append(t.get("total_ms", 0))
            rec = {"set": name, "id": r["id"], "db_id": db, "question": r["question"], "outcome": r["outcome"],
                   "gold_plan": r.get("plan"), "pred_plan": plan.to_dict() if plan is not None else None,
                   "pred_sql": sql, "gold_sql": r.get("gold_sql"), "ms": round(t.get("total_ms", 0), 1),
                   "reason": r.get("reason", "")}
            if r["outcome"] == "unsupported":
                # outside the plan grammar: Luthor still answers; score its SQL against gold by execution
                res["unsup_n"] += 1
                ok = False
                if sql is not None:
                    try:
                        ok = run_sql(con, sql) == run_sql(con, r["gold_sql"])
                    except Exception:
                        res["errors"] += 1
                res["unsup_right"] += ok
                rec["correct"] = ok
            else:
                res["plan_n"] += 1
                ok = False
                if sql is not None:
                    try:
                        ok = run_sql(con, sql) == run_sql(con, r["gold_sql"])
                    except Exception:
                        res["errors"] += 1
                res["plan_right"] += ok
                rec["correct"] = ok
            if dump is not None:
                dump.write(json.dumps(rec, default=str) + "\n")
        con.close()
        rt.cache.pop(db, None)
        if (i + 1) % 10 == 0:
            print(f"  {name}: {i + 1}/{len(by_db)} dbs, covered EX so far "
                  f"{res['plan_right'] / max(1, res['plan_n']):.3f}", flush=True)
    covered = res["plan_right"] / max(1, res["plan_n"])
    # all-questions EX: questions that never became a verified plan cannot be answered right, so
    # all_ex = covered_ex * (verified plan questions in the set / all questions in the set)
    full = [json.loads(l) for l in open(d / "plans.jsonl")]
    n_plan_full = sum(1 for r in full if r["outcome"] == "plan" and not r.get("twin_of"))
    n_unsup_full = sum(1 for r in full if r["outcome"] == "unsupported")
    unsup_rate = res["unsup_right"] / max(1, res["unsup_n"])
    # every question in the set counts; the few that could not be converted at all (no verified
    # plan, no grammar reason) are counted as wrong
    all_ex = (covered * n_plan_full + unsup_rate * n_unsup_full) / max(1, stats["questions"])
    return {"set": name, "questions_in_set": stats["questions"], "scored_plan_questions": res["plan_n"],
            "scored_unsupported": res["unsup_n"], "covered_ex": round(covered, 4), "all_ex": round(all_ex, 4),
            "outside_grammar_ex": round(res["unsup_right"] / max(1, res["unsup_n"]), 4), "errors": res["errors"],
            "latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
            "latency_ms_p90": round(sorted(lat)[int(0.9 * len(lat))], 1) if lat else None,
            "minutes": round((time.time() - t0) / 60, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--sets", nargs="+", default=["bird_dev", "spider_dev", "spider_test", "spider_syn_dev", "wikisql_test"])
    ap.add_argument("--limit", type=int, default=0, help="sample N original questions per set (0 = all)")
    ap.add_argument("--bits", type=int, default=8)
    a = ap.parse_args()
    rt = load_runtime(a.run, a.bits)
    out = []
    for s in a.sets:
        with open(a.run / f"predictions_{s}{'_n' + str(a.limit) if a.limit else ''}.jsonl", "w") as dump:
            r = evaluate_set(rt, s, a.limit or None, dump=dump)
        print(json.dumps(r), flush=True)
        out.append(r)
    tag = "_".join(a.sets) + (f"_n{a.limit}" if a.limit else "")
    (a.run / f"sealed_eval_{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
