"""Calibrate luthor.contamination.near_dup's threshold (CPU, ~1-2 min). Reads ORIGINAL BIRD dev questions only as
the reference side of the filter (no model, nothing fitted to dev results).

  python scripts/analysis/contamination_calib.py > scripts/analysis/contamination_calib.out

1. False-positive side: BIRD-train (9,428 questions on 69 DBs, none of them a dev DB) vs dev. True copies are not
   expected here, so the flag rate at each threshold is an upper bound on the false-positive rate for BIRD-style text.
2. Recall side: synthetic near-copies of every dev question (exact, case/punctuation, one word dropped, two words
   dropped, one content word replaced, a prefix added, evidence removed).
3. Scale/timing: the SynSQL and Spider training rows that System-1 trains on (data/sys1/*_train/plans.jsonl), plus the
   database-level check (dev db_id names and table-name overlap).
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from luthor import contamination as C  # noqa: E402

THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def perturb(q: str, kind: str, rnd: random.Random) -> str:
    w = q.split()
    if kind == "exact":
        return q
    if kind == "case_punct":
        return q.lower().rstrip("?.") + " ?"
    if kind == "drop1" and len(w) > 3:
        del w[rnd.randrange(len(w))]
    elif kind == "drop2" and len(w) > 4:
        for _ in range(2):
            del w[rnd.randrange(len(w))]
    elif kind == "swap1":
        long_ = [i for i, x in enumerate(w) if len(x) > 4]
        if long_:
            w[rnd.choice(long_)] = rnd.choice(["average", "Seattle", "1999", "employees", "highest", "customers"])
    elif kind == "prefix":
        return "Can you tell me: " + q
    return " ".join(w)


def main():
    t0 = time.time()
    dev = json.load(open(C.DEV_JSON))
    T = json.load(open(REPO / "data/raw/bird_train/train.json"))
    ref_qe = C.load_dev_questions(True)
    ref_q = C.load_dev_questions(False)
    tr_qe = [(t["question"] + " " + (t.get("evidence") or "")).strip() for t in T]
    tr_q = [t["question"] for t in T]
    print(f"reference: ORIGINAL BIRD dev (2024-06-27), {len(ref_qe)} questions; query: BIRD-train {len(T)} questions")
    print("shingles: normalised character 5-grams (default) and word 3-grams; Jaccard; exact (prefix-filtered) search\n")
    print("== 1. BIRD-train vs dev: questions flagged (no train DB is a dev DB, so these are false positives) ==")
    res = {}
    for name, qs, ref, n, unit in (("q+evidence, char5", tr_qe, ref_qe, 5, "char"), ("question only, char5", tr_q, ref_q, 5, "char"),
                                   ("q+evidence, word3", tr_qe, ref_qe, 3, "word"), ("question only, word3", tr_q, ref_q, 3, "word")):
        t1 = time.time()
        sims = C.max_similarity(qs, ref, floor=min(THRESHOLDS), n=n, unit=unit)
        res[name] = sims
        row = "  ".join(f">={t:.1f}: {sum(s >= t for s, _ in sims):4d}" for t in THRESHOLDS)
        print(f"  {name:22s} {row}   ({time.time() - t1:.1f}s)")
    print("\n  highest-similarity BIRD-train/dev pairs (q+evidence, char5):")
    sims = res["q+evidence, char5"]
    top = sorted(range(len(T)), key=lambda i: -sims[i][0])[:8]
    for i in top:
        s, j = sims[i]
        print(f"   J={s:.3f}  train[{i}] {T[i]['db_id']}: {T[i]['question'][:90]!r}")
        print(f"            dev[{j}] {dev[j]['db_id']}: {dev[j]['question'][:90]!r}")
    print("\n  examples near 0.5 (q+evidence, char5):")
    mid = [i for i in range(len(T)) if 0.45 <= sims[i][0] < 0.55][:3]
    for i in mid:
        s, j = sims[i]
        print(f"   J={s:.3f}  train: {tr_qe[i][:100]!r}\n            dev:   {ref_qe[j][:100]!r}")

    print("\n== 2. recall on synthetic near-copies of all 1,534 dev questions (question only vs question-only ref; "
          "'no_evidence' = q-only query vs q+evidence ref) ==")
    rnd = random.Random(0)
    for kind in ("exact", "case_punct", "drop1", "drop2", "swap1", "prefix", "no_evidence"):
        if kind == "no_evidence":
            qs, ref = ref_q, ref_qe
        else:
            qs, ref = [perturb(q, kind, rnd) for q in ref_q], ref_q
        sims = C.max_similarity(qs, ref, floor=min(THRESHOLDS))
        print(f"  {kind:12s} " + "  ".join(f">={t:.1f}: {100 * sum(s >= t for s, _ in sims) / len(qs):5.1f}%" for t in THRESHOLDS))

    print("\n== 3. System-1 training sources vs dev at the default threshold "
          f"{C.DEFAULT_THRESHOLD} (q+evidence 'context' where present) ==")
    for src in ("spider_train", "synsql_train", "bird_train"):
        rows = [json.loads(l) for l in open(REPO / f"data/sys1/{src}/plans.jsonl")]
        texts = [(r.get("question", "") + " " + (r.get("context") or "")).strip() for r in rows]
        t1 = time.time()
        flags = C.near_dup(texts, ref_qe)
        dt = time.time() - t1
        dbf = C.dev_db_flags([r.get("db_id", "") for r in rows])
        print(f"  {src:13s} rows {len(rows):7d}  near-dup {sum(flags):4d}  dev db_id {sum(dbf):5d} "
              f"({dict(Counter(r['db_id'] for r, f in zip(rows, dbf) if f))})  {dt:.1f}s ({1e6 * dt / max(1, len(rows)):.0f} us/row)")
        for r, f in list(zip(rows, flags)):
            if f:
                print(f"     flagged {r['id']} {r['db_id']}: {r['question'][:100]!r}")
    tabs = json.load(open(REPO / "data/raw/bird_train/train_tables.json"))
    near = [(t["db_id"], C.schema_near_dev(t["table_names_original"], 0.5)) for t in tabs]
    print(f"  BIRD-train schemas with table-name Jaccard >= 0.5 to a dev schema: {[x for x in near if x[1]]}")
    sp = REPO / "data/eval_raw/spider_data/tables.json"
    if sp.exists():
        stabs = json.load(open(sp))
        hits = [(t["db_id"], C.schema_near_dev(t["table_names_original"], 0.5)) for t in stabs]
        print(f"  Spider schemas (tables.json) named like a dev DB: {[t['db_id'] for t in stabs if C.is_dev_db(t['db_id'])]}; "
              f"table-name Jaccard >= 0.5: {[h for h in hits if h[1]]}")
        sp_rows = Counter(json.loads(l)["db_id"] for l in open(REPO / "data/sys1/spider_train/plans.jsonl"))
        for db, dv in hits:
            if dv:
                s = frozenset(C._norm_db(x) for x in next(t for t in stabs if t["db_id"] == db)["table_names_original"])
                print(f"    Spider {db} ~ dev {dv}: table-name Jaccard {C.jaccard(s, C.dev_table_sets()[dv]):.2f}, "
                      f"{sp_rows[db]} rows in data/sys1/spider_train/plans.jsonl (a System-1 training source)")
    syn = REPO / "data/raw/synsql/tables.json"
    if syn.exists():
        used = {json.loads(l)["db_id"] for l in open(REPO / "data/sys1/synsql_train/plans.jsonl")}
        hits = [(t["db_id"], C.schema_near_dev(t["table_names_original"], 0.4)) for t in json.load(open(syn)) if t["db_id"] in used]
        print(f"  SynSQL training schemas ({len(used)} DBs) with table-name Jaccard >= 0.4 to a dev schema: {[h for h in hits if h[1]]}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
