"""SQL writer candidates on held-out dev databases, merged with the slot model's candidates (round 4).
Runs on the GPU pod in /workspace/venv2 (vLLM).

  python scripts/sqlw_gen.py --lora runs/sqlw_sft/lora --slot-cands runs/sq3_pilot/cands_dev32.jsonl \
      --cards /root/d/plans/spider_train /root/d/plans/bird_train --db-roots /root/d/db BT --out runs/sqlw_sft/union_dev.jsonl

Per question: the writer's greedy SQL + --n samples. Every candidate is executed against the gold result.
Writes the merged candidate list (slot first, then writer: src "slot" / "w") and prints:
  writer greedy / writer oracle / writer vote (most common result among writer SQL that runs and returns rows),
  union oracle / union vote, and the slot model's numbers on the same questions.
Only held-out TRAINING databases; no sealed set is touched.
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sqlw.execute import check_db  # noqa: E402
from luthor.sq.candidates import ValueIndex  # noqa: E402
from luthor.sqlw.prompt import clean_sql, messages, used_tables, value_hints  # noqa: E402


def vote(cands: list[dict]) -> dict | None:
    """Most common result among candidates that run and return a non-null row; ties -> earliest."""
    live = [c for c in cands if c.get("runs") and c.get("nonnull") and c.get("sig")]
    if not live:
        return cands[0] if cands else None
    n = collections.Counter(c["sig"] for c in live)
    best = max(n.values())
    return next(c for c in live if n[c["sig"]] == best)


def metrics(recs: list[dict]) -> dict:
    ok = lambda c: bool(c and c.get("ok"))
    n = max(len(recs), 1)
    W = lambda r: [c for c in r["cands"] if c.get("src") == "w"]
    S = lambda r: [c for c in r["cands"] if c.get("src") != "w"]
    m = {"n": len(recs),
         "slot_greedy": sum(ok(S(r)[0] if S(r) else None) for r in recs) / n,
         "slot_oracle": sum(any(c.get("ok") for c in S(r)) for r in recs) / n,
         "writer_greedy": sum(ok(W(r)[0] if W(r) else None) for r in recs) / n,
         "writer_oracle": sum(any(c.get("ok") for c in W(r)) for r in recs) / n,
         "writer_vote": sum(ok(vote(W(r))) for r in recs) / n,
         "union_oracle": sum(any(c.get("ok") for c in r["cands"]) for r in recs) / n,
         "union_vote": sum(ok(vote(r["cands"])) for r in recs) / n,
         "union8_oracle": sum(any(c.get("ok") for c in S(r)[:8] + W(r)) for r in recs) / n}
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="base, or a merged checkpoint")
    ap.add_argument("--slot-cands", type=Path, help="slot candidate file (questions + slot candidates)")
    ap.add_argument("--questions", type=Path, help="question file instead (e.g. data/eval/dev_frozen.jsonl)")
    ap.add_argument("--hints", action="store_true", help="add value hints to the prompt (round 5)")
    ap.add_argument("--seconds", type=float, default=30.0, help="time limit per query when scoring")
    ap.add_argument("--cards", nargs="+", type=Path, required=True)
    ap.add_argument("--db-roots", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    if a.slot_cands:
        recs = [json.loads(l) for l in open(a.slot_cands)]
    else:
        recs = [{"id": r["id"], "db_id": r["db_id"], "question": r["question"], "context": r.get("context", ""),
                 "gold_sql": r["gold_sql"], "in_slots": r.get("in_slots"), "cands": []} for r in map(json.loads, open(a.questions))]
    if a.limit:
        recs = recs[:a.limit]
    paths = {}
    for root in a.db_roots:
        for p in Path(root).rglob("*.sqlite"):
            paths.setdefault(p.stem, str(p))
    cards = {}
    for d in a.cards:
        for p in (d / "cards").glob("*.json"):
            cards.setdefault(p.stem, p)
    recs = [r for r in recs if r["db_id"] in cards and r["db_id"] in paths]
    loaded, vidx = {}, {}
    prompts = []
    for r in recs:
        c = loaded.setdefault(r["db_id"], SchemaCard.load(cards[r["db_id"]]))
        must = sorted({t for x in r["cands"][:3] if x.get("sql") for t in used_tables(c, x["sql"])})
        hints = None
        if a.hints:
            if r["db_id"] not in vidx:
                con = sqlite3.connect(f"file:{paths[r['db_id']]}?mode=ro", uri=True)
                con.text_factory = lambda b: b.decode("utf-8", "replace")
                vidx[r["db_id"]] = ValueIndex.build(con, c, skip_card_values=False)
                con.close()
            hints = value_hints(vidx[r["db_id"]], r["question"], r.get("context", ""))
        prompts.append(messages(c, r["question"], r.get("context", ""), must=must, hints=hints))
    t0 = time.time()
    llm = LLM(a.model, enable_lora=a.lora is not None, max_lora_rank=64, max_model_len=8192,
              gpu_memory_utilization=a.gpu_mem, seed=0)
    lr = LoRARequest("sqlw", 1, str(a.lora)) if a.lora else None
    greedy = llm.chat(prompts, SamplingParams(temperature=0.0, max_tokens=384), lora_request=lr)
    samp = llm.chat(prompts, SamplingParams(temperature=a.temp, top_p=0.95, n=a.n, max_tokens=384, seed=0),
                    lora_request=lr) if a.n else [None] * len(prompts)
    print(f"generated in {(time.time() - t0) / 60:.1f} min", flush=True)
    by_db = collections.defaultdict(list)
    for i, r in enumerate(recs):
        for c in r["cands"]:
            c.setdefault("src", "slot")
        sqls = [clean_sql(greedy[i].outputs[0].text)] + ([clean_sql(o.text) for o in samp[i].outputs] if samp[i] else [])
        cnt = collections.Counter(sqls)
        r["cands"] += [{"src": "w", "rank": j, "sql": s, "cost": round(1 - cnt[s] / len(sqls), 4)} for j, s in enumerate(sqls)]
        by_db[r["db_id"]].append(i)
    tasks, order = [], []
    for db, idx in by_db.items():
        tasks.append((paths[db], [(recs[i]["gold_sql"], [c["sql"] for c in recs[i]["cands"]]) for i in idx], a.seconds))
        order.append(idx)
    with Pool(a.workers) as pool:
        for idx, res in zip(order, pool.map(check_db, tasks)):
            for i, rr in zip(idx, res):
                if rr is None:
                    recs[i]["gold_error"] = True
                    continue
                for c, (runs, nonnull, ok, sig) in zip(recs[i]["cands"], rr):
                    c.update(runs=runs, nonnull=nonnull, ok=ok, sig=sig)
    # questions whose gold fails stay in the file (flagged), so the denominator is visible downstream
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    m = metrics(recs)
    print(f"UNION DEV {json.dumps(m)}", flush=True)
    print(f"UNION DEV (slot-expressible only) {json.dumps(metrics([r for r in recs if r.get('in_slots')]))}", flush=True)
    (a.out.parent / (a.out.stem + "_metrics.json")).write_text(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
