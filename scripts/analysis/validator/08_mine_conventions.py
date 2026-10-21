"""08 gold_conventions: mine BIRD-train gold SQL for wording-conditional conventions.
Train = train.json minus held-out DBs (data/sys1/dev_dbs_bird.json). Also tabulates DECISION golds (BIRD-train gold
of the DECISION ids; diagnostic) to see whether the held-out DBs follow the train prior. No BIRD dev data is read.
  .venv/bin/python scripts/analysis/validator/08_mine_conventions.py > runs/analysis/validator/08_mine.out
"""
import json, importlib.util, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("cf", ROOT / "scripts/analysis/validator/08_conv_feats.py")
cf = importlib.util.module_from_spec(spec); spec.loader.exec_module(cf)

train = json.load(open(ROOT / "data/raw/bird_train/train.json"))
held = set(json.load(open(ROOT / "data/sys1/dev_dbs_bird.json")))
dec_ids = set(json.load(open(ROOT / "data/sys1/heldout_v2/decision.json"))["ids"])
rows = {"train": [], "decision": []}
for i, r in enumerate(train):
    rid = f"bird_train_train_{i}"
    split = "decision" if rid in dec_ids else ("train" if r["db_id"] not in held else None)
    if split is None:
        continue
    w = cf.wording(r["question"], r.get("evidence", ""))
    c = cf.sql_conv(r["SQL"], r["db_id"])
    rows[split].append({"i": i, "db": r["db_id"], "w": w, "c": c})
print("n train", len(rows["train"]), "dbs", len({x["db"] for x in rows["train"]}), "| n decision", len(rows["decision"]))

TABLES = {}
def tab(name, cond, key):
    out = {}
    for split in ("train", "decision"):
        R = [x for x in rows[split] if cond(x)]
        cnt = Counter(key(x) for x in R)
        n = sum(cnt.values())
        bydb = defaultdict(Counter)
        for x in R: bydb[x["db"]][key(x)] += 1
        out[split] = (n, cnt, bydb)
    n, cnt, bydb = out["train"]
    dn, dcnt, dbydb = out["decision"]
    print(f"\n## {name}\n train n={n}: " + ", ".join(f"{k}={v/n:.2f}" for k, v in cnt.most_common()) if n else f"\n## {name} train n=0")
    if n:
        top = cnt.most_common(1)[0][0]
        rates = [(db, c[top] / sum(c.values()), sum(c.values())) for db, c in bydb.items() if sum(c.values()) >= 5]
        flip = [r for r in rates if r[1] < 0.5]
        print(f"  majority '{top}': train DBs with >=5 q: {len(rates)}, DBs where majority<0.5: {len(flip)}; "
              f"per-DB rate min/median/max {min(r[1] for r in rates) if rates else 0:.2f}/"
              f"{sorted(r[1] for r in rates)[len(rates)//2] if rates else 0:.2f}/{max(r[1] for r in rates) if rates else 0:.2f}")
    if dn:
        print(f" DECISION gold n={dn}: " + ", ".join(f"{k}={v/dn:.2f}" for k, v in dcnt.most_common())
              + " | per DB: " + "; ".join(f"{db}:" + ",".join(f"{k}{v}" for k, v in c.most_common(3)) for db, c in dbydb.items()))
    TABLES[name] = {"train": dict(cnt), "decision": dict(dcnt)}

W = lambda x, k: x["w"][k]
tab("superlative wording -> tie form", lambda x: W(x, "super") and not W(x, "howmany") and not W(x, "pct"), lambda x: x["c"]["tie"])
tab("superlative, evidence has MAX/MIN( -> tie form", lambda x: W(x, "super") and W(x, "ev_max") and not W(x, "howmany"), lambda x: x["c"]["tie"])
tab("superlative, evidence no MAX/MIN -> tie form", lambda x: W(x, "super") and not W(x, "ev_max") and not W(x, "howmany") and not W(x, "pct"), lambda x: x["c"]["tie"])
tab("top-N wording -> tie form", lambda x: W(x, "topn"), lambda x: x["c"]["tie"])
tab("no superlative wording -> tie form", lambda x: not W(x, "super") and not W(x, "topn"), lambda x: x["c"]["tie"])
tab("how many -> count form", lambda x: W(x, "howmany"), lambda x: x["c"]["cnt"])
tab("how many, gold has JOIN -> count form", lambda x: W(x, "howmany") and x["c"]["join"], lambda x: x["c"]["cnt"])
tab("how many, no JOIN -> count form", lambda x: W(x, "howmany") and not x["c"]["join"], lambda x: x["c"]["cnt"])
tab("how many, JOIN, counted col is a PK -> count form", lambda x: W(x, "howmany") and x["c"]["join"] and x["c"]["cnt_pk"], lambda x: x["c"]["cnt"])
tab("how many, JOIN, counted col not PK -> count form", lambda x: W(x, "howmany") and x["c"]["join"] and x["c"]["cnt_pk"] is False, lambda x: x["c"]["cnt"])
tab("how many, no JOIN, counted col not PK -> count form", lambda x: W(x, "howmany") and not x["c"]["join"] and x["c"]["cnt_pk"] is False, lambda x: x["c"]["cnt"])
tab("how many, evidence says DISTINCT -> count form", lambda x: W(x, "howmany") and W(x, "ev_distinct"), lambda x: x["c"]["cnt"])
tab("how many, evidence has COUNT( no DISTINCT -> count form", lambda x: W(x, "howmany") and W(x, "ev_count") and not W(x, "ev_distinct"), lambda x: x["c"]["cnt"])
tab("percent wording -> x100", lambda x: W(x, "pct"), lambda x: x["c"]["x100"])
tab("percent wording, evidence has *100 -> x100", lambda x: W(x, "pct") and W(x, "ev_mult100"), lambda x: x["c"]["x100"])
tab("percent wording, evidence no *100 -> x100", lambda x: W(x, "pct") and not W(x, "ev_mult100"), lambda x: x["c"]["x100"])
tab("percent wording -> CAST AS REAL", lambda x: W(x, "pct"), lambda x: x["c"]["cast_real"])
tab("ratio/average non-pct division -> CAST AS REAL", lambda x: not W(x, "pct") and "/" in json.dumps(x["c"]) is False, lambda x: x["c"]["cast_real"])
tab("superlative + LIMIT 1 gold -> ORDER BY expr in SELECT", lambda x: W(x, "super") and x["c"]["tie"] == "limit1" and x["c"]["order_in_sel"] is not None, lambda x: x["c"]["order_in_sel"])
tab("list-style wording -> SELECT DISTINCT", lambda x: W(x, "list"), lambda x: x["c"]["sel_distinct"])
tab("non-list wording -> SELECT DISTINCT", lambda x: not W(x, "list") and not W(x, "howmany"), lambda x: x["c"]["sel_distinct"])
tab("'full name' wording -> n select cols", lambda x: W(x, "fullname"), lambda x: x["c"]["ncol"])
tab("'name' (not full) list wording -> n select cols", lambda x: W(x, "name") and not W(x, "fullname") and W(x, "list"), lambda x: x["c"]["ncol"])
tab("how many -> n select cols", lambda x: W(x, "howmany"), lambda x: x["c"]["ncol"])
json.dump(TABLES, open(ROOT / "runs/analysis/validator/08_conv_tables.json", "w"), indent=1, default=str)
json.dump(rows["train"], open(ROOT / "runs/analysis/validator/08_train_rows.json", "w"), default=str)
