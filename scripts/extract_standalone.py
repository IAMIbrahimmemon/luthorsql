"""SParC + CoSQL (train splits) -> Spider-format standalone questions.

Only questions that stand alone are kept: each interaction's FINAL goal question and its FIRST turn.
Later turns ("what about the older ones?") need the conversation and would be noise for a
single-question model. Output: data/raw/sparc_cosql/ laid out like spider_data (train_spider.json,
train_others.json, tables.json, database -> Spider's databases), so prepare_plans.py --format spider reads it.
"""
import json
from pathlib import Path

OUT = Path("data/raw/sparc_cosql")
SPIDER = Path("data/raw/spider_data").resolve()

rows, seen = [], set()
spider_q = {q["question"].strip().lower() for f in ("train_spider.json", "train_others.json")
            for q in json.load(open(SPIDER / f))}


def add(db, q, sql, src):
    q, sql = (q or "").strip(), (sql or "").strip().rstrip(";")
    if not q or not sql or not sql.lower().startswith("select"):
        return
    key = (db, q.lower())
    if key in seen or q.lower() in spider_q:  # exact duplicates of Spider train questions add nothing
        return
    seen.add(key)
    rows.append({"db_id": db, "question": q, "query": sql, "source": src})


for x in json.load(open("data/raw/sparc/train.json")):
    add(x["db_id"], x["final_utterance"], x["final_query"], "sparc_final")
    if x["interaction_utterance"] and x["interaction_query"]:
        add(x["db_id"], x["interaction_utterance"][0], x["interaction_query"][0], "sparc_first")
for x in json.load(open("data/raw/cosql/cosql_dataset/sql_state_tracking/cosql_train.json")):
    add(x["database_id"], x["final"]["utterance"], x["final"]["query"], "cosql_final")
    if x["interaction"]:
        add(x["database_id"], x["interaction"][0]["utterance"], x["interaction"][0]["query"], "cosql_first")

OUT.mkdir(parents=True, exist_ok=True)
sd = OUT / "spider_data"
sd.mkdir(exist_ok=True)
(sd / "train_spider.json").write_text(json.dumps(rows, indent=1))
(sd / "train_others.json").write_text("[]")
for name in ("tables.json", "database"):
    if not (sd / name).exists():
        (sd / name).symlink_to(SPIDER / name)
from collections import Counter
print(len(rows), Counter(r["source"] for r in rows))
