"""Selector and reviser training items from writer candidates (round 5, step 2).

Input: candidate files in the sqlw_gen format (id, db_id, question, context, gold_sql, cands[{sql, src, rank}]).
Each distinct candidate is executed once with a preview; groups are formed by result; see
luthor.sqlw.select.build_items for which items are made. Training questions only: --exclude-dbs drops the
held-out dev databases (and the script refuses sealed databases).

  python scripts/sel_data.py --cands writer_train.jsonl --out data/sel/train.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sqlw.evalkit import db_paths  # noqa: E402
from luthor.sqlw.select import build_items, run_preview, top_groups  # noqa: E402


def _connect(path: str, seconds: float):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    box = {"t": 0.0}
    con.set_progress_handler(lambda: 1 if time.time() > box["t"] else 0, 10_000)
    return con, box


def work(task):
    path, card_path, recs, seconds, seed = task
    con, box = _connect(path, seconds)
    card = SchemaCard.load(card_path)
    rnd, out = random.Random(seed), []
    for r in recs:
        if r.get("gold_error"):
            continue
        # sqlw_gen already executed every candidate (runs / nonnull / ok / sig); only the two group
        # representatives the selector will see are run again, for their result previews
        for c in r["cands"]:
            c.update(runs=bool(c.get("runs")), ok=bool(c.get("ok")), sig=c.get("sig") or "", empty=not c.get("nonnull"))
        for g in top_groups(r["cands"]):
            box["t"] = time.time() + seconds
            p = run_preview(con, g[0]["sql"])
            g[0]["preview"] = {k: v for k, v in p.items() if k != "set"}
        out += build_items(card, r, rnd)
    con.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", type=Path, required=True)
    ap.add_argument("--cards", nargs="+", type=Path, default=[Path("data/plans/spider_train/cards"), Path("data/plans/bird_train/cards")])
    ap.add_argument("--db-roots", nargs="+", default=["data/raw/spider_data/database", "data/raw/bird_train/train_databases"])
    ap.add_argument("--exclude-dbs", type=Path, default=Path("data/mix2_dev_dbs.json"))
    ap.add_argument("--allow-dev", action="store_true", help="format check on dev only; never for training")
    ap.add_argument("--writer-only", action="store_true", help="use only writer candidates (src w)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dev = set(json.loads(a.exclude_dbs.read_text()))
    paths = db_paths(a.db_roots)
    cards = {}
    for d in a.cards:
        for p in d.glob("*.json"):
            cards.setdefault(p.stem, p)
    pooled = {}  # one record per question; candidates from every file, in file order
    for f in a.cands:
        for l in open(f):
            r = json.loads(l)
            if (r["db_id"] in dev) != a.allow_dev or r["db_id"] not in paths or r["db_id"] not in cards:
                continue
            cs = [c for c in r["cands"] if not a.writer_only or c.get("src", "w") == "w"]
            if r["id"] in pooled:
                pooled[r["id"]]["cands"] += cs
            else:
                pooled[r["id"]] = {**r, "cands": cs}
    by_db = collections.defaultdict(list)
    for r in pooled.values():
        by_db[r["db_id"]].append(r)
    tasks = [(paths[db], str(cards[db]), rs, a.seconds, i) for i, (db, rs) in enumerate(by_db.items())]
    with Pool(a.workers) as pool:
        items = [x for part in pool.imap_unordered(work, tasks) for x in part]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(x) + "\n" for x in items))
    k = collections.Counter(x["kind"] for x in items)
    sel = [x for x in items if x["kind"] == "select"]
    print(f"SEL_DATA {dict(k)}; select labels {dict(collections.Counter(x['sql'] for x in sel))}; "
          f"revise both-wrong {sum(x.get('both_wrong', False) for x in items)} from {sum(map(len, by_db.values()))} questions", flush=True)


if __name__ == "__main__":
    main()
