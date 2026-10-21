"""Re-score saved candidate files on the frozen dev list (round 5, step 0).

Every candidate is run again and compared with the ORIGINAL human gold SQL. Accuracy is reported over
all frozen questions: a question with no candidates in a file counts as wrong (and is reported as
"unscored"), so the denominator never shrinks silently.

  python scripts/dev_score.py --files sft=runs/sqlw_sft/union_dev_scored.jsonl rl=runs/sqlw_rl/union_dev_scored.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sqlw.evalkit import (check_cands, db_paths, oracle, paired, pick_greedy, pick_vote,  # noqa: E402
                                 pick_vote_empty_majority, top2_groups_ok)

ROOTS = ["data/raw/spider_data/database", "data/raw/bird_train/train_databases"]


def rescore(frozen: dict, recs: list[dict], seconds: float, workers: int) -> dict:
    """id -> list of candidates (src, rank, sql, rr) with runs/empty/ok/sig against the original gold."""
    paths = db_paths(ROOTS)
    out = {}
    for r in recs:
        if r["id"] in frozen:
            out[r["id"]] = [{"src": c["src"], "rank": c["rank"], "sql": c.get("sql"), "rr": c.get("rr")}
                            for c in r["cands"] if c["src"] == "w" or c["rank"] == 0]  # slot: first answer only
    by_db = collections.defaultdict(list)
    for i in out:
        by_db[frozen[i]["db_id"]].append(i)
    tasks = [(paths[db], [({"sig": frozen[i]["gold_sig"]}, [c["sql"] for c in out[i]]) for i in ids], seconds)
             for db, ids in by_db.items()]
    with Pool(workers) as pool:
        for ids, res in zip(by_db.values(), pool.imap(check_cands, tasks)):
            for i, rr in zip(ids, res):
                for c, x in zip(out[i], rr):
                    c.update(x)
    return out


def writer(cands):
    return sorted([c for c in cands if c["src"] == "w"], key=lambda c: c["rank"])


def slot(cands):
    return sorted([c for c in cands if c["src"] == "slot"], key=lambda c: c["rank"])


def pick_rerank(cands, allow_empty=False):
    live = [c for c in cands if c["runs"] and (allow_empty or not c["empty"]) and c.get("rr") is not None]
    return max(live, key=lambda c: c["rr"]) if live else pick_greedy(cands)


def systems(cands_of: dict) -> dict:
    """name -> function(id) -> 0/1 (missing question -> 0)."""
    def f(sel, pick):
        return lambda i: int(bool(i in cands_of and (p := pick(sel(cands_of[i]))) and p["ok"]))
    def g(sel, test):
        return lambda i: int(bool(i in cands_of and test(sel(cands_of[i]))))
    return {
        "slot_greedy": f(slot, pick_greedy),
        "writer_greedy": f(writer, pick_greedy),
        "writer_vote": f(writer, pick_vote),
        "writer_vote_empty": f(writer, lambda c: pick_vote(c, allow_empty=True)),
        "writer_vote_empty_major": f(writer, pick_vote_empty_majority),
        "writer_rerank": f(writer, pick_rerank),
        "writer_rerank_empty": f(writer, lambda c: pick_rerank(c, allow_empty=True)),
        "writer_top2": g(writer, top2_groups_ok),
        "writer_oracle": g(writer, oracle),
        "writer_plus_slot1_oracle": g(lambda c: c, oracle),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", type=Path, default=Path("data/eval/dev_frozen.jsonl"))
    ap.add_argument("--files", nargs="+", required=True, help="name=path of a union_dev(_scored).jsonl")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("data/eval/r4_rescore.json"))
    a = ap.parse_args()
    frozen = {json.loads(l)["id"]: json.loads(l) for l in open(a.frozen)}
    ids = sorted(frozen)
    db_of = {i: frozen[i]["db_id"] for i in ids}
    groups = {"all": ids, "in_slots": [i for i in ids if frozen[i]["in_slots"]],
              "not_in_slots": [i for i in ids if not frozen[i]["in_slots"]],
              "spider": [i for i in ids if frozen[i]["source"] == "spider"],
              "bird": [i for i in ids if frozen[i]["source"] == "bird"],
              "gold_empty": [i for i in ids if frozen[i]["gold_empty"]],
              "clean": [i for i in ids if frozen[i].get("gold_clean", True)],
              "bird_clean": [i for i in ids if frozen[i]["source"] == "bird" and frozen[i].get("gold_clean", True)]}
    report, per = {"groups": {k: len(v) for k, v in groups.items()}}, {}
    for spec in a.files:
        name, path = spec.split("=", 1)
        cache = a.out.parent / f"rescored_{name}.json"
        if cache.exists():
            cands_of = json.loads(cache.read_text())
        else:
            cands_of = rescore(frozen, [json.loads(l) for l in open(path)], a.seconds, a.workers)
            cache.write_text(json.dumps(cands_of))
        per[name] = cands_of
        sysf = systems(cands_of)
        report[name] = {"scored": len(cands_of),
                        **{g: {s: round(100 * sum(fn(i) for i in gi) / max(len(gi), 1), 1) for s, fn in sysf.items()}
                           for g, gi in groups.items()}}
        print(name, json.dumps(report[name]), flush=True)
    if len(per) >= 2:  # SFT + RL writer samples pooled, and paired comparisons
        n1, n2 = list(per)[:2]
        pooled = {i: per[n1][i] + [dict(c, rank=c["rank"] + 1000) for c in per[n2].get(i, []) if c["src"] == "w"]
                  for i in per[n1]}
        sp = systems(pooled)
        report["pooled"] = {g: {s: round(100 * sum(sp[s](i) for i in gi) / max(len(gi), 1), 1)
                                for s in ("writer_vote", "writer_vote_empty", "writer_top2", "writer_oracle")}
                            for g, gi in groups.items()}
        print("pooled", json.dumps(report["pooled"]), flush=True)
        s1, s2 = systems(per[n1]), systems(per[n2])
        on = groups["in_slots"]  # where both files have candidates
        comp = {
            f"{n2}_greedy - {n1}_greedy": (s2["writer_greedy"], s1["writer_greedy"]),
            f"{n2}_rerank - {n2}_greedy": (s2["writer_rerank"], s2["writer_greedy"]),
            f"{n2}_vote_empty - {n2}_vote": (s2["writer_vote_empty"], s2["writer_vote"]),
            f"{n2}_vote_empty_major - {n2}_vote": (s2["writer_vote_empty_major"], s2["writer_vote"]),
            f"pooled_vote - {n2}_vote": (sp["writer_vote"], s2["writer_vote"]),
        }
        report["paired"] = {}
        for k, (fa, fb) in comp.items():
            sub = [i for i in on if i in per[n1] and i in per[n2]]
            report["paired"][k] = paired({i: fa(i) for i in sub}, {i: fb(i) for i in sub}, db_of)
            print("PAIRED", k, json.dumps(report["paired"][k]), flush=True)
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
