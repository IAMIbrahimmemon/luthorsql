"""Decision-model fine-tuning data for candidate selection (Perplexity Decider v1.1 27B; autojev DecisionInput rows).

  python scripts/decider/dataset.py train     # -> data/decider/{train,val}.jsonl  (BIRD-train lists, never held-out)
  python scripts/decider/dataset.py eval      # -> data/decider/eval_{decision,reserved}.jsonl (wide-pool shortlists)

Every row: {"id", "state": str, "question": {"type": "choice", "instructions", "criteria": {A..: candidate text}},
"target": [p per option] (train/val only), "order": [candidate indices] (eval only)}. One builder, `row()`, makes
the train and the eval rows, so the model sees the same format in both.

Train/val: box W's k=32 BIRD-train lists from both runs (runs/phase1/p1w/p1w/{r1,r2}/cands_train.jsonl), pooled per
question by result signature; questions with >= 1 right and >= 1 wrong distinct live result; gold-noise ids and
contaminated DBs dropped. Per question N_LISTS shortlists of K options: one right result plus K-1 wrong ones drawn
with probability falling with the generator's rank (hard negatives first), positions shuffled. Val = every question
of VAL_DBS (whole databases held out from train) for checkpoint choice and temperature fitting; never DECISION.
Eval: exactly the jev_judge.py shortlist (top-4 distinct live results of the wide pool by sum_z, shuffled by id).
"""
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
OUT = ROOT / "data/decider"
W = ROOT / "runs/phase1/p1w/p1w"
K, SEED = 4, 20261009
N_LISTS = int(sys.argv[sys.argv.index("--lists") + 1]) if "--lists" in sys.argv else 3
NO_EXT = "--no-ext" in sys.argv
FIXED_VAL_DBS = {"citeseer", "disney", "professional_basketball", "trains"}  # chosen by the first build (seed above)
LETTERS = "ABCDEFGH"
INSTR = ("Exactly one candidate SQL query is intended to answer the question correctly on this SQLite database. "
         "Pick the candidate whose result best answers the question as the evidence defines it: the right columns "
         "in the right order (only what is asked), the filters and joins the evidence implies, aggregation and "
         "grouping, DISTINCT vs duplicates, ordering/LIMIT for highest/top/most questions.")


def row(qid, db, question, evidence, cands, card, preview):
    """cands: list of SQL strings in display order -> DecisionInput (state text + choice question)."""
    from luthor.sqlw.prompt6 import ddl6
    question, evidence = question or "", evidence or ""
    state = (f"Database schema:\n{ddl6(card, question, evidence)}\n\nEvidence: {(evidence or '').strip() or '(none)'}"
             f"\nQuestion: {question}")
    crit = {LETTERS[n]: f"SQL: {sql}\nResult:\n{preview(db, sql)}" for n, sql in enumerate(cands)}
    return {"id": qid, "state": state, "question": {"type": "choice", "instructions": INSTR, "criteria": crit}}


def cards_for():
    from luthor.sq.schema_card import SchemaCard
    cache = {}

    def get(db):
        if db not in cache:
            cache[db] = SchemaCard.load(ROOT / "data/sys1/bird_train/cards" / f"{db}.json")
        return cache[db]
    return get


def _rows_for(args):
    """All rows (train: N_LISTS shortlists, val: 1) for one question; top-level so a process pool can run it."""
    i, m, opts_pool, is_val = args
    from llm_judge import preview
    card = cards_for()
    right = sorted((v for v in opts_pool if v[2]), key=lambda v: v[0])
    wrong = sorted((v for v in opts_pool if not v[2]), key=lambda v: v[0])
    out = []
    for t in range(1 if is_val else N_LISTS):
        r_ = random.Random(f"{SEED}:{i}:{t}")
        pos = right[0] if t == 0 else r_.choice(right)
        pick, cand = [], list(wrong)
        while cand and len(pick) < K - 1:  # hard negatives first: weight 1/(1+position in rank order)
            w = [1.0 / (1 + j) for j in range(len(cand))]
            j = r_.choices(range(len(cand)), weights=w)[0]
            pick.append(cand.pop(j))
        opts = [pos] + pick
        r_.shuffle(opts)
        rec = row(f"{i}#{t}", m["db_id"], m["question"], m.get("context", ""), [o[1] for o in opts],
                  card(m["db_id"]), preview)
        rec["target"] = [1.0 if o[2] else 0.0 for o in opts]
        out.append(rec)
    return is_val, out


def build_train():
    from llm_judge import preview
    noise = set(json.load(open(ROOT / "data/sys1/gold_noise_drop.json"))["ids"])
    cont = set(json.load(open(ROOT / "data/sys1/contaminated_dbs.json")))
    held = set(json.load(open(ROOT / "data/sys1/dev_dbs_bird.json")))
    meta, pool = {}, collections.defaultdict(dict)   # pool[id][sig] = (best rank, sql, ok)
    for role in ("r1", "r2"):
        for ln in open(W / role / "cands_train.jsonl"):
            r = json.loads(ln)
            meta[r["id"]] = r
            for c in r["cands"]:
                if not (c.get("sql") and c.get("runs") and c.get("nonnull")):
                    continue
                k = c.get("sig") or c["sql"]
                if k not in pool[r["id"]] or c["rank"] < pool[r["id"]][k][0]:
                    pool[r["id"]][k] = (c["rank"], c["sql"], bool(c.get("ok")))
    # external candidates (scripts/decider/ext_pools.py: SENSE + SynSQL-Merge, re-executed and re-labelled here) join
    # the same pools by result signature; their questions are BIRD-train questions on non-held-out DBs
    ext = OUT / "ext_pools.jsonl"
    ext_ids = set()
    if ext.exists() and not NO_EXT:
        for ln in open(ext):
            r = json.loads(ln)
            ext_ids.add(r["id"])
            if r.get("gold_corrected"):  # pool relabelled against BIRD-Verified gold (incl. W's own candidates)
                pool[r["id"]] = {}
                meta[r["id"]] = r
            meta.setdefault(r["id"], r)
            for c in r["cands"]:
                if not (c.get("sql") and c.get("nonnull")):
                    continue
                k = c.get("sig") or c["sql"]
                if k not in pool[r["id"]]:
                    pool[r["id"]][k] = (100 + c["rank"], c["sql"], bool(c.get("ok")))  # after W's own ranks
    ids = sorted(i for i, d in pool.items() if any(v[2] for v in d.values()) and any(not v[2] for v in d.values())
                 and i not in noise and meta[i]["db_id"] not in cont and meta[i]["db_id"] not in held)
    dbs = sorted({meta[i]["db_id"] for i in ids})
    rnd = random.Random(SEED)
    val_dbs = set(rnd.sample(dbs, max(3, len(dbs) // 10)))
    val_dbs = FIXED_VAL_DBS or val_dbs  # pinned: the val split must not move when more data joins
    OUT.mkdir(parents=True, exist_ok=True)
    # resume: keep questions whose rows are complete, drop partial ones
    done, keep = set(), {}
    for name, need in (("train.jsonl", N_LISTS), ("val.jsonl", 1)):
        rows_ = [json.loads(ln) for ln in open(OUT / name)] if (OUT / name).exists() else []
        cnt = collections.Counter(r["id"].split("#")[0] for r in rows_)
        # rows of questions that gained external candidates are rebuilt from the merged pool
        # with fewer lists than an earlier build, its first lists (#0..#need-1) are reused as they are
        rows_ = [r for r in rows_ if int(r["id"].split("#")[1]) < need]
        cnt = collections.Counter(r["id"].split("#")[0] for r in rows_)
        keep[name] = [r for r in rows_ if cnt[r["id"].split("#")[0]] == need and r["id"].split("#")[0] not in ext_ids]
        done |= {q for q, c in cnt.items() if c == need and q not in ext_ids}
    for name, rs in keep.items():
        with open(OUT / name, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)
    todo = [(i, meta[i], list(pool[i].values()), meta[i]["db_id"] in val_dbs) for i in ids if i not in done]
    print(f"{len(done)} questions already built, {len(todo)} to build", flush=True)
    n = collections.Counter({"train": len(keep["train.jsonl"]), "val": len(keep["val.jsonl"])})
    from multiprocessing import get_context
    with open(OUT / "train.jsonl", "a") as ft, open(OUT / "val.jsonl", "a") as fv, \
            get_context("spawn").Pool(8) as pool_:
        for k_, (is_val, recs) in enumerate(pool_.imap_unordered(_rows_for, todo, chunksize=4), 1):
            for rec in recs:
                (fv if is_val else ft).write(json.dumps(rec) + "\n")
                n["val" if is_val else "train"] += 1
            if k_ % 100 == 0:
                print(f"  {k_}/{len(todo)}", flush=True)
    stats = {"questions": len(ids), "dbs": len(dbs), "val_dbs": sorted(val_dbs), "rows": dict(n), "K": K,
             "lists_per_question": N_LISTS, "seed": SEED}
    (OUT / "train_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


def build_eval():
    import os
    from llm_judge import load, preview, shortlist
    card = cards_for()
    b1, b2 = load(W / "heldout_union_by_r1.jsonl"), load(W / "heldout_union_by_r2.jsonl")
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("decision", "reserved"):
        ids = json.load(open(ROOT / f"data/sys1/heldout_v2/{split}.json"))["ids"]
        n = 0
        with open(OUT / f"eval_{split}.jsonl", "w") as f:
            for i in ids:
                if i not in b2:
                    continue
                top = shortlist(b1[i], b2[i], K)
                if len(top) < 2:
                    continue
                order = top[:]
                random.Random(int(hashlib.md5(i.encode()).hexdigest()[:8], 16)).shuffle(order)
                r = b2[i]
                rec = row(i, r["db_id"], r["question"], r.get("context", ""), [r["cands"][j]["sql"] for j in order],
                          card(r["db_id"]), preview)
                rec["order"] = order
                f.write(json.dumps(rec) + "\n")
                n += 1
        print(split, n, "rows")
    _ = os


if __name__ == "__main__":
    {"train": build_train, "eval": build_eval}[sys.argv[1]]()
