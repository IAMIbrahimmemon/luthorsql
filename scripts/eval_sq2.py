"""Evaluate a LuthorSQ2 checkpoint on the Mac (torch, MPS): greedy vs execution-guided top-K.

  python scripts/eval_sq2.py --run runs/sq2_r2 --plans PLANS_DIR... --dev-dbs data/mix2_dev_dbs.json \
      --db-roots DIR... --out reports/eval_sq2_dev.jsonl [--limit 600] [--k 8]

Use held-out TRAINING databases (--dev-dbs) for tuning decoding. Sealed sets are scored once, at the end,
with --sealed (which skips the --dev-dbs filter).
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":  # --grammar: LUTHOR_GRAMMAR before the first luthor.sq import (workers inherit it)
    from luthor.grammar_cli import preset
    preset()

import train_sq2  # noqa: E402
from train_sq2 import load_sets  # noqa: E402

from luthor import grammar_cli  # noqa: E402
from luthor.sq import bird_desc  # noqa: E402
from luthor.sq.model import Backbone  # noqa: E402
from luthor.sq.plan import compile_plan  # noqa: E402
from luthor.sq2.model import LuthorSQ2, TorchScores  # noqa: E402
from luthor.sq2.topk import candidates, choose  # noqa: E402
from luthor.sq.candidates import ValueIndex, find_candidates  # noqa: E402
from luthor.sq2.train import encode_db, question_body, select_cands, serve_chunks  # noqa: E402

_LOG = logging.getLogger("luthorsql")


class DB:
    def __init__(self, path: Path, seconds: float = 5.0):
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.con.text_factory = lambda b: b.decode("utf-8", "replace")
        self.deadline, self.seconds = 0.0, seconds
        self.con.set_progress_handler(lambda: 1 if time.time() > self.deadline else 0, 10_000)

    def rows(self, sql: str, n: int = 100_000):
        self.deadline = time.time() + self.seconds
        return self.con.execute(sql).fetchmany(n)


def run_args(run: Path) -> dict:
    """Training arguments saved with the run (args.json); {} for runs that predate it."""
    p = run / "args.json"
    return json.loads(p.read_text()) if p.exists() else {}


def cand_order(ra: dict) -> str:
    """The run's literal-candidate order (--cand-order); v1 for runs that predate the flag."""
    return ra.get("cand_order", "v1")


def use_cand_order(ra: dict) -> str:
    """Build candidates (load_sets workers, select_cands) the way the run was trained."""
    import luthor.sq2.train as _tr
    train_sq2.CAND_ORDER = _tr.CAND_ORDER = cand_order(ra)
    return _tr.CAND_ORDER


def gold_sql(r: dict) -> str:
    """Reference SQL: the verified compiled plan when the question has one, else the original SQL (uncovered)."""
    return r.get("compiled_sql") or r["gold_sql"]


def build_model(bb, ra: dict) -> LuthorSQ2:
    """The slot model with the heads the run's args.json describes ({} = a run before these flags)."""
    return LuthorSQ2(bb, joint_layers=int(ra.get("joint_layers", 0)), col_mentions=ra.get("col_mentions") == "True",
                     mention_weight=os.environ.get("LUTHOR_MENTION_WEIGHT") or ra.get("mention_weight", "flat"),
                     value_side=ra.get("value_side") == "True", value_entries=int(ra.get("value_entries", 24)))


def load_model(run: Path, base: str, sub: str, device: str) -> LuthorSQ2:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    ra = run_args(run)
    bb = Backbone.from_pretrained(base, sub or None, lora_r=int(ra.get("lora_r", 32)), grad_ckpt=False)
    m = build_model(bb, ra)
    set_peft_model_state_dict(m.backbone.model, load_file(str(run / "best_lora" / "adapter_model.safetensors")))
    m.load_state_dict(torch.load(run / "best_heads.pt", map_location="cpu"), strict=False)
    m.backbone.model.to(torch.bfloat16)
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--plans", nargs="+", type=Path, required=True)
    ap.add_argument("--dev-dbs", type=Path)
    ap.add_argument("--sealed", action="store_true")
    ap.add_argument("--db-roots", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--distinct", action="store_true", help="k different plans (round 7)")
    ap.add_argument("--old-cands", action="store_true", help="use the candidates stored at load time (round-2 behaviour)")
    ap.add_argument("--max-cost", type=float, default=1.0, help="fallback only to alternatives this close (cross-validated)")
    ap.add_argument("--base", default="AlexWortega/openjev")
    ap.add_argument("--subfolder", default="qwen3.5-4b-nli-v5")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--all-questions", action="store_true",
                    help="score every question, uncovered ones too (no plan: gold = original SQL; counted in n)")
    ap.add_argument("--questions", type=Path, help="with --all-questions: BIRD-format dev.json, adds questions with no plan row")
    ap.add_argument("--bird-desc", nargs="?", const="1", default="",
                    help="BIRD-train column descriptions in the schema cards (luthor.sq.bird_desc): bare = "
                         "data/raw/bird_train_desc, or a directory. Default: the run's own setting (args.json)")
    grammar_cli.add_argument(ap)
    a = ap.parse_args()
    a.grammar = grammar_cli.check_run(a.run, a.grammar)  # the run's args.json grammar (missing = v3)

    ra = run_args(a.run)
    bird_desc.enable(a.bird_desc or ra.get("bird_desc", ""))  # cards as the run was trained
    if ra.get("base"):  # the run's own backbone wins over the round-2 defaults
        a.base, a.subfolder = ra["base"], ra.get("subfolder", "")
    if (a.run / "vocab.json").exists():  # grammar v3: the template / number vocabulary the heads were built for
        v = json.loads((a.run / "vocab.json").read_text())
        train_sq2.VOCAB_ARGS = (v["templates"], v["nums"])
        train_sq2._init_worker(train_sq2.VOCAB_ARGS)
    order = use_cand_order(ra)
    print(f"candidate order {order}, value-side {ra.get('value_side') == 'True'}", flush=True)
    cards, exs, gold = load_sets(a.plans, workers=6, strict=False, all_rows=a.all_questions, questions=a.questions)
    exs = [e for e in exs if not gold[e.id].get("twin_of")]
    if not a.sealed:
        dev = set(json.loads(a.dev_dbs.read_text()))
        exs = [e for e in exs if e.db_id in dev]
    exs.sort(key=lambda e: (e.db_id, e.id))
    if a.limit:  # spread the limit over databases, not the first few
        step = max(1, len(exs) // a.limit)
        exs = exs[::step][:a.limit]
    paths = {}
    for root in a.db_roots:
        for p in Path(root).rglob("*.sqlite"):
            paths.setdefault(p.stem, p)
    print(f"{len(exs)} questions on {len({e.db_id for e in exs})} databases", flush=True)

    m = load_model(a.run, a.base, a.subfolder, a.device)
    by_db = collections.defaultdict(list)
    for e in exs:
        by_db[e.db_id].append(e)
    out, t0 = [], time.time()
    tot = collections.Counter()
    with torch.no_grad():
        for db, group in by_db.items():
            if db not in paths:
                continue
            con, card = DB(paths[db]), cards[db]
            g = card.graph()
            if not a.old_cands:  # serve-time candidates: card values + live database lookup
                vindex = ValueIndex.build(con.con, card)
                for e in group:
                    body = question_body(gold[e.id])
                    e.cands, _ = select_cands(find_candidates(body, card, extra=vindex.match(body)), order=order, body=body)
            for tabs, chunk in serve_chunks(card, group, 8):
                st, oix, enc = encode_db(m, card, chunk, tabs)
                for e, x in zip(chunk, enc):
                    sc = TorchScores(m, x, st.option_ids, e.cands)
                    try:
                        ref = set(map(tuple, con.rows(gold_sql(gold[e.id]))))
                    except Exception:
                        _LOG.debug("%s: gold query failed to run", e.id, exc_info=True)
                        if a.all_questions:  # every question counts: one whose gold does not run scores 0
                            tot["n"] += 1
                            tot["gold_error"] += 1
                        continue
                    t1 = time.time()
                    try:
                        pc = candidates(sc, k=a.k, with_cost=True, distinct=a.distinct,
                                        compile_fn=lambda p: compile_plan(p, g))
                    except Exception:
                        _LOG.warning("%s: candidate search failed, no candidates", e.id, exc_info=True)
                        pc = []
                    t_dec = time.time() - t1
                    cand_log, seen = [], {}
                    for rank, (c, cost) in enumerate(pc):
                        st_ = {"rank": rank, "cost": round(cost, 4), "runs": False, "rows": 0, "nonnull": False, "ok": False}
                        try:
                            sql = compile_plan(c, g)
                            st_["sql"] = sql
                            if sql in seen:
                                st_.update({k: v for k, v in cand_log[seen[sql]].items() if k in ("runs", "rows", "nonnull", "ok")})
                                st_["dup"] = True
                            else:
                                seen[sql] = rank
                                got = con.rows(sql)
                                st_.update(runs=True, rows=len(got), nonnull=any(v is not None for r in got for v in r),
                                           ok=set(map(tuple, got)) == ref)
                        except Exception:
                            _LOG.debug("%s: candidate %d failed to compile or run", e.id, rank, exc_info=True)
                        cand_log.append(st_)
                    if not cand_log:  # nothing decoded (uncovered questions can fail): scored 0
                        cand_log.append({"rank": 0, "cost": 0.0, "runs": False, "rows": 0, "nonnull": False, "ok": False})
                    greedy_ok, oracle = cand_log[0]["ok"], any(c["ok"] for c in cand_log)
                    gsql = cand_log[0].get("sql")
                    g0 = cand_log[0]
                    pick = g0
                    if not (g0["runs"] and g0["nonnull"]):  # greedy errored or returned nothing: try close alternatives
                        pick = next((c for c in cand_log[1:] if c["runs"] and c["nonnull"] and c["cost"] <= a.max_cost), g0)
                    topk_ok, sql, rank = pick["ok"], pick.get("sql"), pick["rank"]
                    tot["n"] += 1
                    tot["greedy"] += greedy_ok
                    tot["topk"] += topk_ok
                    tot["oracle"] += oracle
                    tot["in_slots"] += e.t is not None
                    tot["covered"] += gold[e.id].get("outcome") == "plan"
                    out.append({"id": e.id, "db_id": db, "question": e.question, "greedy_sql": gsql, "topk_sql": sql,
                                "rank": rank, "greedy_ok": greedy_ok, "topk_ok": topk_ok, "oracle_ok": oracle,
                                "gold_sql": gold_sql(gold[e.id]), "covered": gold[e.id].get("outcome") == "plan",
                                "in_slots": e.t is not None, "decode_s": t_dec, "cands": cand_log,
                                "ref_rows": len(ref)})
            n = tot["n"]
            print(f"{db}: n={n} greedy {tot['greedy'] / n:.1%} topk {tot['topk'] / n:.1%} oracle@{a.k} "
                  f"{tot['oracle'] / n:.1%}  {(time.time() - t0) / n:.2f}s/q", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    n = max(tot["n"], 1)
    print(f"FINAL n={tot['n']} (covered {tot['covered']}) greedy {tot['greedy'] / n:.1%} topk {tot['topk'] / n:.1%} oracle@{a.k} {tot['oracle'] / n:.1%}")


if __name__ == "__main__":
    main()
