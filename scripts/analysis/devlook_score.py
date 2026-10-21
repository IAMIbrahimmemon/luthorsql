"""BIRD dev look #1 (runs/phase1/devlook/FROZEN.md): pick VALHARD's top candidate per question, execute it against the
gold ONCE, print strict EX. Mac, CPU, < 2 GB RAM.

  .venv/bin/python scripts/analysis/devlook_score.py --scored runs/phase1/devlook/devlook/poolA_by_valhard.jsonl --variant A
  .venv/bin/python scripts/analysis/devlook_score.py --scored runs/phase1/devlook/devlook/poolB_by_valhard.jsonl --variant B
  # dry run on DECISION (BIRD-train gold, 796 q; RESERVED rows are skipped by id, never parsed further; no lock):
  .venv/bin/python scripts/analysis/devlook_score.py --practice-decision --scored runs/phase1/valhard/valhard/heldout_by_new.jsonl

Conventions (as the DECISION evals, scripts/analysis/valhard_eval.py, and the 59.06 pooled number, docs/SYSTEM1_ENS1_RESULTS.md):
  - Pick = max VALHARD rr over live candidates (runs and returns a non-null value; all candidates with SQL if none is
    live). No VAL+DROP, no second validator, no vote.
  - Denominator = all 1,534 ORIGINAL BIRD dev questions. A question with no candidate (or missing from the scored file)
    counts wrong. Uncovered (grammar-inexpressible) questions are in the pool like every other question (the generator
    never sees coverage) and are scored the same way, as in 59.06.
  - Strict EX: the pick's result rows and the gold's as SETS of tuples are equal; the same execution as the box-side
    scorer that produced 59.06 (rerank_sq2._exec_db: read-only, 3.0 s per query, at most 100,000 rows, text decoded
    with replacement). A gold that fails to run scores 0 for that question (rerank_sq2 gen --all-questions).
  - Headline: ORIGINAL gold (data/bench/dev_20240627/dev.json). Info: REVISED gold (2025-11-06; questions / evidence
    the model saw are the original ones, so this is a proxy, as in SYSTEM1_ENS1_RESULTS.md).
One look: the first real run writes runs/phase1/devlook/LOOK_USED BEFORE executing anything; a second run is refused
unless --confirm-second-run (and is recorded in the lock file).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIG = ROOT / "data/bench/dev_20240627/dev.json"
REVISED = ROOT / "data/raw/bird_dev_20251106/data/dev_20251106-00000-of-00001.json"
DEV_DBS = ROOT / "data/bench/dev_20240627/dev_databases"
LOCK = ROOT / "runs/phase1/devlook/LOOK_USED"
DECISION = ROOT / "data/sys1/heldout_v2/decision.json"
TRAIN = ROOT / "data/raw/bird_train/train.json"
TRAIN_DBS = ROOT / "data/raw/bird_train/train_databases"
REFERENCE = ("reference (not this system): 59.06 = R1 u R2 pooled, both validators sum_z, + frozen VAL+DROP repair; "
             "58.21 = the same pooled validator pick without VAL+DROP (ORIGINAL dev, all 1,534; SYSTEM1_ENS1_RESULTS.md)")


# ---- pick (scripts/analysis/valhard_eval.py, verbatim rule) -----------------------------------------------------
def live(cands):
    L = [c for c in cands if c.get("runs") and c.get("nonnull")]
    return L or cands


def pick(cands):
    u = [c for c in cands if c.get("sql")]
    if not u:
        return None
    return max(live(u), key=lambda c: c.get("rr", -1e9))


# ---- execution (rerank_sq2._connect / _exec_db semantics) ------------------------------------------------------
def _connect(path: str, seconds: float):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    box = {"t": 0.0}
    con.set_progress_handler(lambda: 1 if time.time() > box["t"] else 0, 10_000)

    def rows(sql):
        box["t"] = time.time() + seconds
        return con.execute(sql).fetchmany(100_000)
    return rows


def _run(rows, sql):
    if not sql:
        return None
    try:
        return set(map(tuple, rows(sql)))
    except Exception:
        return None


def _exec_db(task):
    """(db path, [(qid, [gold sql per gold version], pick sql, [other sqls for the oracle])], seconds) ->
    [(qid, [hit per gold], [gold failed per gold], [oracle hit per gold] | None)]"""
    path, qs, seconds = task
    rows = _connect(path, seconds)
    out = []
    for qid, golds, psql, others in qs:
        refs = [_run(rows, g) for g in golds]
        got = _run(rows, psql)
        hit = [r is not None and got is not None and got == r for r in refs]
        orc = None
        if others is not None:
            res = [_run(rows, s) for s in others]
            orc = [r is not None and any(x is not None and x == r for x in res) for r in refs]
        out.append((qid, hit, [r is None for r in refs], orc))
    return out


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def take_lock(a) -> None:
    rec = {"time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "variant": a.variant,
           "scored": str(a.scored), "scored_sha256": sha(a.scored), "argv": sys.argv[1:]}
    if LOCK.exists():
        if not a.confirm_second_run:
            sys.exit(f"REFUSED: {LOCK} exists (the one approved BIRD dev look is used):\n{LOCK.read_text()}\n"
                     "Re-run with --confirm-second-run only with the owner's explicit approval.")
        with open(LOCK, "a") as f:
            f.write(json.dumps({**rec, "second_run": True}) + "\n")
        print(f"WARNING: second run on BIRD dev, recorded in {LOCK}")
        return
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "x") as f:  # exclusive create: two concurrent runs cannot both pass
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", type=Path, required=True, help="the box's pool<V>_by_valhard.jsonl")
    ap.add_argument("--variant", choices=["A", "B"], help="required for the real look")
    ap.add_argument("--confirm-second-run", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="also execute every candidate (pool ceiling; decide BEFORE the look)")
    ap.add_argument("--seconds", type=float, default=3.0, help="per-query time limit (3.0 = the 59.06 scorer)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--json", type=Path, help="per-question results (default runs/phase1/devlook/look_<V>.json)")
    ap.add_argument("--practice-decision", action="store_true",
                    help="dry run on DECISION (BIRD-train gold, held-out DBs); no lock, no revised gold")
    a = ap.parse_args()

    if a.practice_decision:
        dec = json.load(open(DECISION))
        ids = list(dec["ids"])
        tr = json.load(open(TRAIN))
        meta = {i: tr[int(i.rsplit("_", 1)[1])] for i in ids}
        golds = {i: [meta[i]["SQL"]] for i in ids}
        gold_names = ["DECISION (BIRD-train gold)"]
        db_root = TRAIN_DBS
        diff = {i: "n/a" for i in ids}
        name = "practice"
    else:
        if not a.variant:
            sys.exit("--variant A|B required")
        if "decision" in str(a.scored).lower() or "heldout" in str(a.scored).lower():
            sys.exit("this file looks like a held-out pool; use --practice-decision")
        orig = json.load(open(ORIG))
        rev = json.load(open(REVISED))
        assert [q["question_id"] for q in orig] == [q["question_id"] for q in rev]
        ids = [f"bird_dev_{q['question_id']}" for q in orig]
        meta = {i: q for i, q in zip(ids, orig)}
        golds = {i: [q["SQL"], r["SQL"]] for i, q, r in zip(ids, orig, rev)}
        gold_names = ["ORIGINAL (2024-06-27)", "REVISED (2025-11-06, info)"]
        db_root = DEV_DBS
        diff = {i: q.get("difficulty", "n/a") for i, q in zip(ids, orig)}
        name = a.variant
    idset = set(ids)

    # load the scored file: rows outside the id set are skipped by id (never parsed further, e.g. RESERVED)
    S, extra = {}, 0
    for line in open(a.scored):
        r = json.loads(line)
        if r["id"] not in idset:
            extra += 1
            continue
        S[r["id"]] = r
    if not a.practice_decision:
        assert extra == 0, f"{extra} rows in {a.scored} are not BIRD dev questions"
    n_rr = sum("rr" in c for r in S.values() for c in r["cands"])
    if not a.practice_decision and n_rr == 0:
        sys.exit("no VALHARD scores (rr) in the scored file: refusing")

    picks = {}
    for i in ids:
        p = pick(S[i]["cands"]) if i in S else None
        picks[i] = p
    no_cand = sum(p is None for p in picks.values())
    n_live = sum(1 for i in ids if picks[i] is not None and picks[i].get("runs") and picks[i].get("nonnull"))
    print(f"{name}: {len(ids)} questions; scored file has {len(S)} ({extra} other rows skipped), "
          f"{sum(len(r['cands']) for r in S.values())} candidates, {n_rr} with rr; no pick {no_cand}; live picks {n_live}")

    if not a.practice_decision:
        take_lock(a)  # BEFORE anything touches the gold

    tasks = collections.defaultdict(list)
    for i in ids:
        db = meta[i]["db_id"]
        others = None
        if a.oracle and i in S:
            others = list(dict.fromkeys(c["sql"] for c in S[i]["cands"] if c.get("sql")))
        tasks[db].append((i, golds[i], picks[i]["sql"] if picks[i] else None, others))
    jobs = [(str(db_root / db / f"{db}.sqlite"), qs, a.seconds) for db, qs in tasks.items()]
    t0 = time.time()
    res = {}
    with Pool(a.workers) as pool:
        for part in pool.imap_unordered(_exec_db, jobs):
            for qid, hit, gfail, orc in part:
                res[qid] = (hit, gfail, orc)
    n = len(ids)
    out = {"variant": name, "scored": str(a.scored), "scored_sha256": sha(a.scored), "n": n, "no_pick": no_cand,
           "seconds": a.seconds, "exec_minutes": round((time.time() - t0) / 60, 2), "gold": {}}
    print(f"executed in {(time.time() - t0) / 60:.1f} min")
    for gi, gname in enumerate(gold_names):
        hits = {i: res[i][0][gi] for i in ids}
        gf = sum(res[i][1][gi] for i in ids)
        ex = 100 * sum(hits.values()) / n
        tag = "HEADLINE " if gi == 0 else "info     "
        print(f"{tag}{gname}: strict EX {ex:.2f}  ({sum(hits.values())}/{n}; gold failed to run on {gf}, scored 0)")
        g = {"ex": round(ex, 2), "hits": sum(hits.values()), "gold_failed": gf, "by_difficulty": {}, "by_db": {}}
        for d in ("simple", "moderate", "challenging", "n/a"):
            q = [i for i in ids if diff[i] == d]
            if q:
                v = 100 * sum(hits[i] for i in q) / len(q)
                g["by_difficulty"][d] = {"n": len(q), "ex": round(v, 2)}
        if g["by_difficulty"]:
            print("    by difficulty: " + ", ".join(f"{d} {x['ex']:.2f} (n={x['n']})" for d, x in g["by_difficulty"].items()))
        dbs = sorted({meta[i]["db_id"] for i in ids})
        for d in dbs:
            q = [i for i in ids if meta[i]["db_id"] == d]
            g["by_db"][d] = {"n": len(q), "ex": round(100 * sum(hits[i] for i in q) / len(q), 2)}
        print("    by database: " + ", ".join(f"{d} {x['ex']:.1f}" for d, x in g["by_db"].items()))
        if a.oracle:
            o = sum(bool(res[i][2] and res[i][2][gi]) for i in ids)
            g["oracle"] = round(100 * o / n, 2)
            print(f"    pool oracle {100 * o / n:.2f}")
        g["per_question"] = {i: bool(hits[i]) for i in ids}
        out["gold"][gname] = g
    print(REFERENCE)
    jp = a.json or (ROOT / f"runs/phase1/devlook/look_{name}.json")
    if a.practice_decision and a.json is None:
        jp = Path("/dev/null")
    out["picks"] = {i: (picks[i]["sql"] if picks[i] else None) for i in ids}
    if str(jp) != "/dev/null":
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(out, indent=1))
        print(f"-> {jp}")


if __name__ == "__main__":
    main()
