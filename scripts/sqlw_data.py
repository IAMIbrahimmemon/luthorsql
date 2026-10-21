"""Training data for the SQL writer (round 4). Public data only.

  python scripts/sqlw_data.py --out data/sqlw --bird-list bird_small.txt

  sft.jsonl  supervised rows {id, db_id, source, messages, sql}:
             every human-written row of the public train sets (Spider, Spider-Syn, BIRD, SParC/CoSQL), INCLUDING
             queries the slot grammar cannot express; plus checked mix3 rows (SynSQL spread over many databases,
             generated questions, WikiSQL), target = the original SQL where there is one.
  rl.jsonl   reward-learning prompts {id, db_id, messages, gold_sql} on Spider / BIRD train databases the pod has.
Held out: --dev-dbs and every sealed database; questions with >= 50% word overlap with a sealed question.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_mix3 import EVAL, HUMAN, V3  # noqa: E402
from prepare_plans import dedup_against  # noqa: E402

from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sq.candidates import ValueIndex  # noqa: E402
from luthor.sqlw.evalkit import db_paths  # noqa: E402
from luthor.sqlw.prompt import messages, used_tables, value_hints  # noqa: E402

BIRD_FILTERED = Path("data/raw/bird_filtered/data/train-00000-of-00001.jsonl")  # birdsql/bird23-train-filtered


def spread(rows: list[dict], n: int, rnd: random.Random) -> list[dict]:
    """Up to n rows, round-robin over databases (as many distinct databases as possible)."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["db_id"]].append(r)
    for v in by.values():
        rnd.shuffle(v)
    dbs = list(by)
    rnd.shuffle(dbs)
    out, i = [], 0
    while len(out) < n and any(by.values()):
        db = dbs[i % len(dbs)]
        if by[db]:
            out.append(by[db].pop())
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/sqlw"))
    ap.add_argument("--mix3", type=Path, default=Path("data/mix3"))
    ap.add_argument("--dev-dbs", type=Path, default=Path("data/mix2_dev_dbs.json"))
    ap.add_argument("--bird-list", type=Path, required=True, help="BIRD train databases present on the pod")
    ap.add_argument("--synsql", type=int, default=12000)
    ap.add_argument("--gen", type=int, default=6000)
    ap.add_argument("--wiki", type=int, default=2000)
    ap.add_argument("--rl", type=int, default=8000)
    # round 5
    ap.add_argument("--bird-clean", action="store_true",
                    help="drop BIRD train questions removed by BIRD's official quality filter (SFT and RL)")
    ap.add_argument("--curr", type=Path, help="verified direct-SQL rows (scripts/curriculum_prep.py --collect)")
    ap.add_argument("--hints", action="store_true", help="value hints in the prompt (needs the databases)")
    ap.add_argument("--db-roots", nargs="+", default=["data/raw/spider_data/database",
                                                      "data/raw/bird_train/train_databases", "data/raw/synsql/databases"])
    a = ap.parse_args()
    rnd = random.Random(0)
    dev = set(json.loads(a.dev_dbs.read_text()))
    eval_q, eval_dbs = [], set()
    for d in EVAL.iterdir():
        for l in open(d / "plans.jsonl"):
            r = json.loads(l)
            eval_q.append(r["question"])
            eval_dbs.add(r["db_id"])
    keep_db = lambda db: db not in eval_dbs and db not in dev
    cards: dict[str, SchemaCard] = {}

    def card(src_dir: Path, db: str) -> SchemaCard | None:
        if db not in cards:
            f = src_dir / f"{db}.json"
            cards[db] = SchemaCard.load(f) if f.exists() else None
        return cards[db]

    human = []
    for name in HUMAN:
        for l in open(V3 / name / "plans.jsonl"):
            r = json.loads(l)
            if r.get("twin_of") or not r.get("gold_sql") or not keep_db(r["db_id"]):
                continue
            human.append({**r, "source": name, "_cards": V3 / name / "cards"})
    if a.bird_clean:
        norm = lambda s: " ".join(s.lower().split())
        clean = {(r["db_id"], norm(r["question"])) for r in map(json.loads, open(BIRD_FILTERED))}
        nb = len(human)
        human = [r for r in human if r["source"] != "bird_train" or (r["db_id"], norm(r["question"])) in clean]
        print(f"bird-clean: dropped {nb - len(human)} BIRD rows removed by the official filter", flush=True)
    n0 = len(human)
    human = dedup_against(human, eval_q)
    print(f"human rows: {len(human)} (dropped {n0 - len(human)} for sealed overlap); "
          f"outside the slot grammar: {sum(r.get('outcome') != 'plan' for r in human)}", flush=True)

    mix = [json.loads(l) for l in open(a.mix3 / "plans.jsonl")]
    mix = [r for r in mix if "#" not in r["id"] and not r.get("twin_of")]
    pick = (spread([r for r in mix if r["source"] == "synsql_train"], a.synsql, rnd)
            + spread([r for r in mix if r["source"].startswith("gen_")], a.gen, rnd)
            + spread([r for r in mix if r["source"] == "wikisql_train"], a.wiki, rnd))
    for r in pick:
        r["_cards"] = a.mix3 / "cards"
    if a.curr:
        curr = [json.loads(l) for l in open(a.curr)]
        curr = [{**r, "source": "curr_synsql", "_cards": V3 / "synsql_train" / "cards"} for r in curr if keep_db(r["db_id"])]
        pick += curr
        print(f"curriculum rows: {len(curr)}", flush=True)
    paths = db_paths(a.db_roots) if a.hints else {}
    vidx = {}

    def hints_for(r, c):
        if not a.hints or r["db_id"] not in paths:
            return None
        if r["db_id"] not in vidx:
            if len(vidx) > 64:
                vidx.clear()
            con = sqlite3.connect(f"file:{paths[r['db_id']]}?mode=ro", uri=True)
            con.text_factory = lambda b: b.decode("utf-8", "replace")
            vidx[r["db_id"]] = ValueIndex.build(con, c, skip_card_values=False)
            con.close()
        return value_hints(vidx[r["db_id"]], r["question"], r.get("context", ""))

    a.out.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    with open(a.out / "sft.jsonl", "w") as f:
        rows = human + pick
        rnd.shuffle(rows)
        for r in rows:
            c = card(r["_cards"], r["db_id"])
            if c is None:
                stats["no_card"] += 1
                continue
            sql = r.get("gold_sql") or r["compiled_sql"]
            msg = messages(c, r["question"], r.get("context", ""), must=used_tables(c, sql), hints=hints_for(r, c))
            f.write(json.dumps({"id": r["id"], "db_id": r["db_id"], "source": r["source"], "messages": msg,
                                "sql": sql}) + "\n")
            stats[r["source"]] += 1
    print("sft", dict(stats), "databases", len({r["db_id"] for r in rows}), flush=True)

    bird_ok = set(a.bird_list.read_text().split())
    spider_dbs = {p.name for p in Path("data/raw/spider_data/database").iterdir()}
    rl = [r for r in human if r["source"] in ("spider_train", "bird_train")
          and ((r["source"] == "spider_train" and r["db_id"] in spider_dbs) or r["db_id"] in bird_ok)]
    rl = spread(rl, a.rl, rnd)
    with open(a.out / "rl.jsonl", "w") as f:
        for r in rl:
            c = card(r["_cards"], r["db_id"])
            if c is None:
                continue
            f.write(json.dumps({"id": r["id"], "db_id": r["db_id"], "source": r["source"],
                                "messages": messages(c, r["question"], r.get("context", ""), hints=hints_for(r, c)),
                                "gold_sql": r["gold_sql"]}) + "\n")
    print("rl", len(rl), collections.Counter(r["source"] for r in rl), flush=True)


if __name__ == "__main__":
    main()
