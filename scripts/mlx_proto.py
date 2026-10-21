"""Mac prototype of agent-style prompting (round 5, step 2), with MLX and a 4-bit base Qwen2.5-Coder-7B.

Compares, on the same frozen dev questions (default: 100 clean-gold BIRD-style questions):
  plain   - the writer prompt, greedy
  plan    - write a short plan (tables, joins, filters, grouping, output), then the SQL
  repair  - plain, then one retry with the SQLite error (or "no rows") when the query fails or is empty
  hints   - plain prompt plus value hints
This is the base model, not the trained writer: only differences between modes are informative.

  python scripts/mlx_proto.py --n 100
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sq.candidates import ValueIndex  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sqlw.evalkit import db_paths, paired  # noqa: E402
from luthor.sqlw.prompt import clean_sql, messages, value_hints  # noqa: E402

ROOTS = ["data/raw/spider_data/database", "data/raw/bird_train/train_databases"]
CARD_DIRS = [Path("data/plans/spider_train/cards"), Path("data/plans/bird_train/cards")]
PLAN_SYSTEM = ("You write one SQLite query that answers the question exactly. First write a short plan: the "
               "tables and joins, the filters with exact values, any grouping or ordering, and the output "
               "columns. Then give the final query in a ```sql block.")


def run(path: str, sql: str, seconds: float = 30.0):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    t = time.time() + seconds
    con.set_progress_handler(lambda: 1 if time.time() > t else 0, 10_000)
    try:
        return con.execute(sql).fetchmany(100_000), None
    except Exception as e:
        return None, str(e)[:200]
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    ap.add_argument("--frozen", type=Path, default=Path("data/eval/dev_frozen.jsonl"))
    ap.add_argument("--source", default="bird")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--modes", nargs="+", default=["plain", "plan", "repair", "hints"])
    ap.add_argument("--out", type=Path, default=Path("data/eval/mlx_proto.jsonl"))
    a = ap.parse_args()
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(a.model)
    greedy = make_sampler(temp=0.0)

    def chat(msgs, max_tokens):
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=greedy, verbose=False)

    rows = [json.loads(l) for l in open(a.frozen)]
    rows = [r for r in rows if r["source"] == a.source and r.get("gold_clean", True)]
    random.Random(0).shuffle(rows)
    rows = rows[:a.n]
    paths = db_paths(ROOTS)
    cards = {}
    for d in CARD_DIRS:
        for p in d.glob("*.json"):
            cards.setdefault(p.stem, p)
    done = {}
    if a.out.exists():
        for l in open(a.out):
            x = json.loads(l)
            done[(x["id"], x["mode"])] = x
    vidx, cc = {}, {}
    f = open(a.out, "a")
    t0 = time.time()
    for k, r in enumerate(rows):
        db = r["db_id"]
        card = cc.setdefault(db, SchemaCard.load(cards[db]))
        gold, _ = run(paths[db], r["gold_sql"], 60)
        ref = set(map(tuple, gold))
        for mode in a.modes:
            if (r["id"], mode) in done:
                continue
            q, ctx = r["question"], r["context"]
            if mode == "plan":
                msgs = messages(card, q, ctx)
                msgs[0] = {"role": "system", "content": PLAN_SYSTEM}
                text = chat(msgs, 700)
            else:
                hints = None
                if mode == "hints":
                    if db not in vidx:
                        con = sqlite3.connect(f"file:{paths[db]}?mode=ro", uri=True)
                        con.text_factory = lambda b: b.decode("utf-8", "replace")
                        vidx[db] = ValueIndex.build(con, card, skip_card_values=False)
                        con.close()
                    hints = value_hints(vidx[db], q, ctx)
                msgs = messages(card, q, ctx, hints=hints)
                text = chat(msgs, 400)
            sql = clean_sql(text)
            got, err = run(paths[db], sql)
            retried = False
            if mode == "repair" and (err or not any(v is not None for row in got for v in row)):
                note = f"That query failed: {err}" if err else "That query returned no rows."
                msgs = msgs + [{"role": "assistant", "content": sql},
                               {"role": "user", "content": note + " Write a corrected query. Reply with the SQL only."}]
                sql = clean_sql(chat(msgs, 400))
                got, err = run(paths[db], sql)
                retried = True
            ok = got is not None and set(map(tuple, got)) == ref
            x = {"id": r["id"], "db_id": db, "mode": mode, "sql": sql, "ok": ok, "error": err, "retried": retried}
            f.write(json.dumps(x) + "\n")
            f.flush()
            done[(r["id"], mode)] = x
        print(f"{k + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    f.close()
    db_of = {r["id"]: r["db_id"] for r in rows}
    ids = [r["id"] for r in rows]
    acc = {m: {i: int(done[(i, m)]["ok"]) for i in ids if (i, m) in done} for m in a.modes}
    rep = {m: round(100 * sum(v.values()) / max(len(v), 1), 1) for m, v in acc.items()}
    rep["paired_vs_plain"] = {m: paired(acc[m], acc["plain"], db_of) for m in a.modes if m != "plain"}
    rep["repair_retries"] = sum(done[(i, "repair")]["retried"] for i in ids if (i, "repair") in done)
    print("MLX_PROTO", json.dumps(rep), flush=True)
    a.out.with_suffix(".summary.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
