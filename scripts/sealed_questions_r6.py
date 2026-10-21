"""Sealed question files for round 6: KaggleDBQA (test split), Spider-DK and Spider-Realistic.
Same fields and gold procedure as scripts/sealed_questions.py (gold run once, 60 s limit, result signature
stored). Never used for training, tuning or prompt design; the KaggleDBQA few-shot split is not used anywhere.

  python scripts/sealed_questions_r6.py                    # build all r6 sets -> data/eval/sealed_<set>.jsonl
  python scripts/sealed_questions_r6.py --sets spider_dk   # build only some
  python scripts/sealed_questions_r6.py --overlap          # report question/db overlap with training data (no writes)

Raw data (data/raw is the external-SSD symlink):
  data/raw/kaggledbqa/repo        git clone https://github.com/Chia-Hsuan-Lee/KaggleDBQA
  data/raw/kaggledbqa/databases   Google Drive folder 1g-Y9Up2_mtQijqUXBNcTWmsFIvsFnV7f (databases.zip, unzipped)
  data/raw/spider_dk/repo         git clone https://github.com/ygan/Spider-DK
  data/raw/spider_realistic       https://zenodo.org/records/5205322 (spider-realistic.json)
"""
from __future__ import annotations

import argparse
import collections
import difflib
import glob
import json
import re
import string
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sqlw.evalkit import db_paths, run_gold  # noqa: E402

SPIDER_DB = "data/raw/spider_data/database"
KAGGLE = "data/raw/kaggledbqa"


def _kaggle(split):
    out = []
    for f in sorted(glob.glob(f"{KAGGLE}/repo/examples/*_{split}.json")):
        out += json.load(open(f))
    return out


SETS = {  # name: (loader -> raw rows, database roots in priority order, question key, gold key)
    "kaggledbqa_test": (lambda: _kaggle("test"), [f"{KAGGLE}/databases"], "question", "query"),
    "spider_dk": (lambda: json.load(open("data/raw/spider_dk/repo/Spider-DK.json")),
                  ["data/raw/spider_dk/repo/database", SPIDER_DB], "question", "query"),
    "spider_realistic": (lambda: json.load(open("data/raw/spider_realistic/spider-realistic.json")),
                         [SPIDER_DB], "question", "query"),
}


def build(name):
    load, roots, qk, gk = SETS[name]
    paths = {k: v for k, v in db_paths(roots).items() if not k.startswith("._")}  # skip exFAT AppleDouble files
    rows = [{"id": f"{name}_{i}", "set": name, "source": name, "db_id": r["db_id"], "question": r[qk].strip(),
             "context": "", "gold_sql": r[gk]} for i, r in enumerate(load())]
    missing = sorted({r["db_id"] for r in rows if r["db_id"] not in paths})
    by_db = collections.defaultdict(list)
    for r in rows:
        if r["db_id"] in paths:
            by_db[r["db_id"]].append(r)
    tasks = [(paths[db], [r["gold_sql"] for r in rs], 60.0) for db, rs in by_db.items()]
    with Pool(8) as pool:
        for rs, res in zip(by_db.values(), pool.map(run_gold, tasks)):
            for r, (status, n, empty, sig) in zip(rs, res):
                r.update(gold_status=status, gold_rows=n, gold_empty=empty, gold_sig=sig)
    for r in rows:
        r.setdefault("gold_status", "no_db")
    Path(f"data/eval/sealed_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(name, len(rows), "dbs", len({r["db_id"] for r in rows}), "missing dbs", missing,
          dict(collections.Counter(r["gold_status"] for r in rows)),
          "empty", sum(r.get("gold_empty", False) for r in rows), flush=True)
    if name == "kaggledbqa_test":
        print("  kaggledbqa fewshot split (recorded only, not used):", len(_kaggle("fewshot")), flush=True)


# ---------- overlap report ----------
_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def norm(q):
    return " ".join(q.lower().translate(_PUNCT).split())


def training_questions():
    """(source, normalized question) for every training question, plus training db ids per source."""
    qs, dbs = [], collections.defaultdict(set)

    def add(src, q, db):
        if q:
            qs.append((src, norm(q)))
        dbs[src].add(db)

    for f in ["data/raw/spider_data/train_spider.json", "data/raw/spider_data/train_others.json"]:
        if Path(f).exists():
            for r in json.load(open(f)):
                add("spider_train", r["question"], r["db_id"])
    if Path("data/raw/spider_syn/train_spider.json").exists():
        for r in json.load(open("data/raw/spider_syn/train_spider.json")):
            add("spider_syn_train", r.get("SpiderSynQuestion") or r.get("question"), r["db_id"])
    for r in json.load(open("data/raw/bird_train/train.json")):
        add("bird_train", r["question"], r["db_id"])
    for r in json.load(open("data/raw/sparc/train.json")):
        for u in r["interaction_utterance"] + [r["final_utterance"]]:
            add("sparc_train", u, r["db_id"])
    for r in json.load(open("data/raw/cosql/cosql_dataset/sql_state_tracking/cosql_train.json")):
        for t in r["interaction"]:
            add("cosql_train", t["utterance"], r["database_id"])
        add("cosql_train", (r.get("final") or {}).get("utterance"), r["database_id"])
    sft_schemas = {}
    for line in open("data/sqlw5/sft.jsonl"):
        r = json.loads(line)
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        q = user.rsplit("Question:", 1)[-1].strip() if "Question:" in user else ""
        add("sqlw5_sft:" + r["source"], q, r["db_id"])
        sft_schemas.setdefault(r["db_id"], {t.lower() for t in re.findall(r"CREATE TABLE\s+[`\"]?(\w+)", user)})
    return qs, dbs, sft_schemas


def shingles(q, n=3):
    w = q.split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def overlap():
    train, train_dbs, sft_schemas = training_questions()
    print("training questions:", len(train), dict(collections.Counter(s.split(":")[0] for s, _ in train)))
    exact = collections.defaultdict(set)
    index = collections.defaultdict(list)  # word 3-gram -> training question ids
    for i, (src, q) in enumerate(train):
        exact[q].add(src)
        for s in shingles(q):
            index[s].append(i)
    all_train_dbs = set().union(*train_dbs.values())
    spider_dev_dbs = {r["db_id"] for r in json.load(open("data/raw/spider_data/dev.json"))}
    print("spider dev dbs in any training source:", sorted(spider_dev_dbs & all_train_dbs) or "none")
    for name in SETS:
        f = Path(f"data/eval/sealed_{name}.jsonl")
        rows = [json.loads(line) for line in open(f)] if f.exists() else \
            [{"question": r[SETS[name][2]], "db_id": r["db_id"]} for r in SETS[name][0]()]
        n_exact, near, ex_examples = 0, [], []
        for r in rows:
            q = norm(r["question"])
            if q in exact:
                n_exact += 1
                ex_examples.append((r["question"], sorted(exact[q])))
                continue
            cand = collections.Counter(i for s in shingles(q) for i in index.get(s, ()))
            best = (0.0, None)
            for i, _ in cand.most_common(200):
                ratio = difflib.SequenceMatcher(None, q, train[i][1]).ratio()
                if ratio > best[0]:
                    best = (ratio, i)
            if best[0] >= 0.9:
                near.append((r["question"], train[best[1]][1], train[best[1]][0], round(best[0], 3)))
        dbs = sorted({r["db_id"] for r in rows})
        base = {d: d.removeprefix("new_") for d in dbs}
        hit = {d: sorted(s for s, v in train_dbs.items() if base[d] in v) for d in dbs}
        print(f"\n== {name}: {len(rows)} questions, {len(dbs)} dbs")
        print("  exact matches:", n_exact, " near (ratio>=0.9):", len(near))
        for e in ex_examples[:3]:
            print("   exact:", e)
        for e in near[:3]:
            print("   near:", e)
        print("  dbs:", dbs)
        print("  dbs (or their Spider base) in training data:", {d: v for d, v in hit.items() if v} or "none")
        if name == "kaggledbqa_test":  # renamed DBs escape the name check: compare (table, column) sets too
            def cols(t):
                return {(t["table_names_original"][ti].lower(), c.lower()) for ti, c in t["column_names_original"] if ti >= 0}
            kag = {t["db_id"]: cols(t) for t in json.load(open(f"{KAGGLE}/repo/KaggleDBQA_tables.json"))}
            train_schemas = {}  # spider/bird train schemas (column level); r5 SFT-only schemas by table name below
            for tf in ["data/raw/bird_train/train_tables.json", "data/raw/spider_data/tables.json"]:
                for t in json.load(open(tf)):
                    if t["db_id"] in all_train_dbs:
                        train_schemas.setdefault(t["db_id"], cols(t))
            n_q = collections.Counter(r["db_id"] for r in rows)
            for d in dbs:
                cov, jac, best = max((len(kag[d] & s) / len(kag[d]), len(kag[d] & s) / len(kag[d] | s), k)
                                     for k, s in train_schemas.items())
                tabs = {t for t, _ in kag[d]}
                tj, tbest = max((len(tabs & s) / len(tabs | s), k) for k, s in sft_schemas.items() if s)
                flag = "  <-- SAME/SUPERSET SCHEMA IN TRAINING" if cov >= 0.9 else ""
                print(f"  {d} ({n_q[d]} q): nearest train schema {best} column coverage {cov:.2f} jaccard {jac:.2f};"
                      f" nearest r5-SFT schema {tbest} table jaccard {tj:.2f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=list(SETS), choices=list(SETS))
    ap.add_argument("--overlap", action="store_true", help="only print the overlap report")
    a = ap.parse_args()
    if a.overlap:
        overlap()
        return
    for name in a.sets:
        build(name)


if __name__ == "__main__":
    main()
