"""Where round-5 writer v2 goes wrong on BIRD dev (now the round-6 dev set). For every question the vote's
pick is compared with the gold result and the miss is put into one shape-based bucket; the misses are
written out with question, evidence, gold and picked SQL for reading.

  python scripts/dev6_errors.py --cands runs/r5/sealed_bird_dev_v2.jsonl
  # -> data/eval/dev6_errors.json (counts), data/eval/dev6_errors.jsonl (one row per miss)
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sqlw.evalkit import db_paths, pick_vote  # noqa: E402
from luthor.sqlw.execute import connect  # noqa: E402


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def bucket(gold, got):
    """Shape of a wrong result against the gold one (both lists of tuples)."""
    if not got:
        return "empty"
    g, p = set(map(tuple, gold)), set(map(tuple, got))
    gc, pc = len(gold[0]) if gold else 0, len(got[0])
    if gc and pc != gc:
        # same answer with extra or missing columns?
        if pc > gc and any(all(any(x == y for y in row) for x in grow) for grow in gold for row in got[:50]):
            return "extra_columns"
        return "fewer_columns" if pc < gc else "more_columns_other"
    if p < g:
        return "subset_rows"
    if p > g:
        return "superset_rows"
    if len(g) == len(p) == 1:
        a, b = next(iter(g)), next(iter(p))
        na, nb = [num(x) for x in a], [num(x) for x in b]
        if all(x is not None for x in na + nb):
            ratios = [y / x for x, y in zip(na, nb) if x]
            if ratios and all(abs(r - 100) < 1e-6 or abs(r - 0.01) < 1e-9 for r in ratios):
                return "percent_scale"
            if all(abs(x - y) <= 1e-4 * max(1, abs(x)) for x, y in zip(na, nb)):
                return "rounding"
            if any(float(x).is_integer() != float(y).is_integer() for x, y in zip(na, nb)):
                return "int_vs_real"
            return "wrong_number"
        return "wrong_value"
    if g & p:
        return "overlap_rows"
    if len(p) == len(g):
        return "same_count_disjoint"
    return "disjoint"


def work(task):
    path, items = task
    run = connect(path, 30.0)
    out = []
    for it in items:
        try:
            gold = run(it["gold_sql"]) if it["id"] != "bird_dev_701" else None
        except Exception:
            gold = None
        try:
            got = run(it["pick"]) if it["pick"] else []
            err = ""
        except Exception as e:
            got, err = None, str(e)[:200]
        if got is None:
            b = "fails"
        elif gold is None:
            b = "gold_slow"
        else:
            b = bucket(gold, got)
        out.append({**it, "bucket": b, "error": err, "gold_rows_preview": (gold or [])[:3],
                    "pick_rows_preview": (got or [])[:3]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", type=Path, default=Path("runs/r5/sealed_bird_dev_v2.jsonl"))
    ap.add_argument("--dev", type=Path, default=Path("data/eval/dev6_bird.jsonl"))
    ap.add_argument("--db-root", default="data/bench/dev_20240627/dev_databases")
    ap.add_argument("--out", type=Path, default=Path("data/eval/dev6_errors"))
    a = ap.parse_args()
    dev = {json.loads(l)["id"]: json.loads(l) for l in open(a.dev)}
    cands = {}
    for l in open(a.cands):
        r = json.loads(l)
        cands[r["id"]] = [{"sql": c["sql"], "runs": bool(c.get("runs")), "empty": not c.get("nonnull"),
                           "sig": c.get("sig") or ""} for c in r["cands"] if c.get("src", "w") == "w" and c.get("sql")]
    misses, counts = [], collections.Counter()
    for qid, d in dev.items():
        cs = cands.get(qid, [])
        for c in cs:
            c["ok"] = c["runs"] and c["sig"] == d["gold_sig"]
        pick = pick_vote(cs) if cs else None
        if pick and pick["ok"]:
            counts["right"] += 1
            continue
        groups = collections.Counter(c["sig"] for c in cs if c["runs"] and not c["empty"])
        right = sum(c["ok"] for c in cs)
        misses.append({"id": qid, "db_id": d["db_id"], "difficulty": d["difficulty"], "question": d["question"],
                       "evidence": d["context"], "gold_sql": d["gold_sql"], "pick": pick["sql"] if pick else "",
                       "n_right": right, "n_cands": len(cs), "winner_votes": max(groups.values()) if groups else 0,
                       "n_groups": len(groups)})
    by_db = collections.defaultdict(list)
    for m in misses:
        by_db[m["db_id"]].append(m)
    paths = db_paths([a.db_root])
    with Pool(8) as pool:
        done = [x for part in pool.map(work, [(paths[db], ms) for db, ms in by_db.items()]) for x in part]
    n = len(dev)
    rep = {"n": n, "right": counts["right"], "acc": round(100 * counts["right"] / n, 1), "misses": len(done),
           "oracle_hit_vote_lost": sum(m["n_right"] > 0 for m in done),
           "no_candidate_right": sum(m["n_right"] == 0 for m in done),
           "buckets": dict(collections.Counter(m["bucket"] for m in done).most_common()),
           "buckets_when_no_candidate_right": dict(collections.Counter(m["bucket"] for m in done if m["n_right"] == 0).most_common()),
           "by_difficulty": {k: {"n": sum(d["difficulty"] == k for d in dev.values()),
                                 "miss": sum(m["difficulty"] == k for m in done)} for k in ("simple", "moderate", "challenging")},
           "by_db": {db: {"n": sum(d["db_id"] == db for d in dev.values()), "miss": len(ms)} for db, ms in sorted(by_db.items())}}
    Path(f"{a.out}.json").write_text(json.dumps(rep, indent=1))
    Path(f"{a.out}.jsonl").write_text("".join(json.dumps(m, default=str) + "\n" for m in done))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
