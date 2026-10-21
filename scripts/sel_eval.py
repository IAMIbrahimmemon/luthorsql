"""Apply the group selector and the reviser to writer candidates on the frozen dev list (round 5, session A).

For each question the writer candidates (one or more files, pooled) are executed and grouped by result.
When the two largest groups disagree, the selector picks one; the reviser writes a new query, which is
executed. Reported on the full frozen list against plain voting, with database-resampled ranges.

  python scripts/sel_eval.py --cands runs/r5/dev_sft.jsonl runs/r5/dev_rl.jsonl --lora runs/r5/sel/lora --out runs/r5/sel_dev.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sqlw.evalkit import db_paths, paired, pick_vote  # noqa: E402
from luthor.sqlw.prompt import clean_sql, messages  # noqa: E402
from luthor.sqlw.select import revise_messages, run_preview, select_messages, top_groups  # noqa: E402


def connect(path: str, seconds: float):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    box = {"t": 0.0}
    con.set_progress_handler(lambda: 1 if time.time() > box["t"] else 0, 10_000)

    def run(sql):
        box["t"] = time.time() + seconds
        return run_preview(con, sql)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", type=Path, default=Path("data/eval/dev_frozen.jsonl"))
    ap.add_argument("--cands", nargs="+", type=Path, required=True, help="sqlw_gen outputs on the frozen list")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--lora", type=Path, help="selector + reviser LoRA (omit: vote and repair only)")
    ap.add_argument("--writer-lora", type=Path, help="writer LoRA on --model, for one error-repair retry")
    ap.add_argument("--cards", nargs="+", type=Path, default=[Path("data/plans/spider_train/cards"), Path("data/plans/bird_train/cards")])
    ap.add_argument("--db-roots", nargs="+", default=["data/raw/spider_data/database", "data/raw/bird_train/train_databases"])
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    frozen = {json.loads(l)["id"]: json.loads(l) for l in open(a.frozen)}
    # candidates with the execution results sqlw_gen already stored (runs / nonnull / ok / sig); only the
    # pair representatives shown to the selector and new revised or repaired queries are executed here
    pool = collections.defaultdict(list)
    for f in a.cands:
        for l in open(f):
            r = json.loads(l)
            for c in r["cands"]:
                if c.get("src", "w") == "w" and c.get("sql"):
                    pool[r["id"]].append({"sql": c["sql"], "runs": bool(c.get("runs")), "empty": not c.get("nonnull"),
                                          "ok": bool(c.get("ok")), "sig": c.get("sig") or ""})
    paths = db_paths(a.db_roots)
    cards = {}
    for d in a.cards:
        for p in d.glob("*.json"):
            cards.setdefault(p.stem, p)
    loaded, runs, recs, cache = {}, {}, [], {}

    def execute(db, sql):
        if (db, sql) not in cache:
            run = runs.setdefault(db, connect(paths[db], a.seconds))
            cache[(db, sql)] = run(sql)
        return cache[(db, sql)]

    def judge(fr, sql):
        p = execute(fr["db_id"], sql)
        return {"sql": sql, "runs": p["runs"], "empty": p["empty"], "preview": {k: v for k, v in p.items() if k != "set"},
                "ok": p["runs"] and fr.get("gold_status", "ok") == "ok" and p["sig"] == fr["gold_sig"]}

    for qid, fr in frozen.items():
        recs.append({"id": qid, "db_id": fr["db_id"], "fr": fr, "cands": pool.get(qid, [])})
    llm = LLM(a.model, enable_lora=True, max_lora_rank=64, max_loras=2, max_model_len=8192,
              gpu_memory_utilization=0.85, seed=0)
    lr = LoRARequest("sel", 1, str(a.lora)) if a.lora else None
    wr = LoRARequest("writer", 2, str(a.writer_lora)) if a.writer_lora else None
    todo = []
    for r in recs:
        g = top_groups(r["cands"])
        r["vote"] = pick_vote(r["cands"])  # round-4 rule: empty results do not win (step 0)
        if len(g) >= 2:
            card = loaded.setdefault(r["db_id"], SchemaCard.load(cards[r["db_id"]]))
            r["pair"] = tuple({**c, "preview": judge(r["fr"], c["sql"])["preview"]} for c in (g[0][0], g[1][0]))
            todo.append((r, card))
    if lr is None:
        todo = []
    sel_out = llm.chat([select_messages(c, r["fr"]["question"], r["fr"]["context"], *r["pair"]) for r, c in todo],
                       SamplingParams(temperature=0.0, max_tokens=2), lora_request=lr) if todo else []
    rev_out = llm.chat([revise_messages(c, r["fr"]["question"], r["fr"]["context"], *r["pair"]) for r, c in todo],
                       SamplingParams(temperature=0.0, max_tokens=384), lora_request=lr) if todo else []
    for (r, _), so, ro in zip(todo, sel_out, rev_out):
        letter = so.outputs[0].text.strip()[:1].upper()
        r["select"] = r["pair"][1] if letter == "B" else r["pair"][0]
        r["revise"] = judge(r["fr"], clean_sql(ro.outputs[0].text))
    # one error-repair retry: when the picked answer fails or returns nothing, show the writer the error
    for pick in ("vote", "select"):
        rep = []
        for r in recs:
            c = r.get(pick, r["vote"])
            if wr is None or c is None or (c.get("runs") and not c.get("empty", False)):
                continue
            card = loaded.setdefault(r["db_id"], SchemaCard.load(cards[r["db_id"]]))
            err = judge(r["fr"], c["sql"])["preview"].get("error", "") if not c.get("runs") else ""
            note = (f"That query failed: {err}" if not c.get("runs")
                    else "That query returned no rows.")
            rep.append((r, messages(card, r["fr"]["question"], r["fr"]["context"])
                        + [{"role": "assistant", "content": c["sql"]},
                           {"role": "user", "content": note + " Write a corrected query. Reply with the SQL only."}]))
        outs = llm.chat([m for _, m in rep], SamplingParams(temperature=0.0, max_tokens=384), lora_request=wr) if rep else []
        for (r, _), o in zip(rep, outs):
            fixed = judge(r["fr"], clean_sql(o.outputs[0].text))
            # keep the retry only if it runs and returns rows; otherwise keep the original answer
            r[pick + "_repair"] = fixed if fixed["runs"] and not fixed["empty"] else r.get(pick, r["vote"])
        r_n = len(rep)
        print(f"repair after {pick}: {r_n} retries", flush=True)
    ok = lambda c: int(bool(c and c.get("ok")))
    ids = [r["id"] for r in recs]
    db_of = {r["id"]: r["db_id"] for r in recs}
    sysm = {
        "greedy": {r["id"]: ok(r["cands"][0] if r["cands"] else None) for r in recs},
        "vote": {r["id"]: ok(r["vote"]) for r in recs},
        "select": {r["id"]: ok(r.get("select", r["vote"])) for r in recs},
        "revise": {r["id"]: ok(r.get("revise", r["vote"])) for r in recs},
        "vote_repair": {r["id"]: ok(r.get("vote_repair", r["vote"])) for r in recs},
        "select_repair": {r["id"]: ok(r.get("select_repair", r.get("select", r["vote"]))) for r in recs},
        "oracle": {r["id"]: int(any(c["ok"] for c in r["cands"])) for r in recs},
    }
    groups = {"all": ids, "clean": [i for i in ids if frozen[i].get("gold_clean", True)]}
    for src in sorted({frozen[i]["source"] for i in ids}):
        groups[src] = [i for i in ids if frozen[i]["source"] == src]
    rep = {g: {s: round(100 * sum(v[i] for i in gi) / len(gi), 1) for s, v in sysm.items()} for g, gi in groups.items()}
    rep["disagreements"] = len(todo)
    rep["paired"] = {f"{s} - vote": paired(sysm[s], sysm["vote"], db_of)
                     for s in ("select", "revise", "vote_repair", "select_repair")}
    rep["revise_broke_right_vote"] = sum(1 for r in recs if "revise" in r and ok(r["vote"]) and not r["revise"]["ok"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    print("SEL_EVAL", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
