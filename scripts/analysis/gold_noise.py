"""Gold-noise drop list (research problem 04-07, cheap first step): BIRD-train questions whose ORIGINAL gold SQL
errors, times out, returns no rows, or returns a single 0 / a single NULL. Under an execution reward such a question
rewards only plans that reproduce the pathology (or nothing), and BIRD-Verified marks most of them wrong.

  python scripts/analysis/gold_noise.py [--seconds 30] [--workers 6]   (writes data/sys1/gold_noise_drop.json; ~20 min)

Executes every train.json gold read-only on data/raw/bird_train/train_databases (all 9,428, so the P(wrong) rates of
reports/fanout_r1/labels/r1/run_all.log can be reproduced), then writes drop ids for the questions OUTSIDE the nine
held-out databases (data/sys1/dev_dbs_bird.json); held-out pathologies are tagged in data/sys1/heldout_v2/universe.json.
P(wrong | category) uses BIRD-Verified (data/raw/bird_verified/cmp.json: `changed` = the corrected SQL returns a
different result than the original), matched on db_id + normalised question text as in the label study.
No dev data is read. Output log: gold_noise.out next to this file.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout_v2 as hv  # noqa: E402

REPO = hv.REPO
CMP = REPO / "data/raw/bird_verified/cmp.json"
OUT = REPO / "data/sys1/gold_noise_drop.json"
DROP_CATS = ("error", "timeout", "empty", "single_zero", "single_null")
norm_q = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()  # noqa: E731  (same key as the label study)


def category(rec: dict) -> str | None:
    """Gold execution record (heldout_v2._gold_chunk) -> drop category or None (keep)."""
    if rec["status"] in ("error", "timeout"):
        return rec["status"]
    p = rec.get("pathology")
    return p if p in DROP_CATS else None


def verified_changed(T: list[dict]) -> dict:
    """train.json index -> BIRD-Verified `changed` flag, for the questions BIRD-Verified covers."""
    cmp = json.load(open(CMP))
    out = {}
    for i, t in enumerate(T):
        k = t["db_id"] + "|" + norm_q(t["question"])
        if k in cmp:
            out[i] = bool(cmp[k]["changed"])
    return out


def rate_table(T, gold, ver, idx, title):
    print(f"\n== {title}: n={len(idx)}, BIRD-Verified covers {sum(i in ver for i in idx)} ==")
    print(f"{'category':12s} {'count':>6s} {'rate':>6s} {'verified n':>10s} {'P(wrong|cat)':>13s}")
    for c in DROP_CATS + ("all_null", "any_drop", "kept"):
        if c == "any_drop":
            sel = [i for i in idx if category(gold[hv.qid(i)])]
        elif c == "kept":
            sel = [i for i in idx if not category(gold[hv.qid(i)])]
        elif c == "all_null":
            sel = [i for i in idx if gold[hv.qid(i)].get("pathology") == "all_null"]
        else:
            sel = [i for i in idx if category(gold[hv.qid(i)]) == c]
        v = [ver[i] for i in sel if i in ver]
        pw = f"{100 * sum(v) / len(v):12.1f}%" if v else f"{'n/a':>13s}"
        print(f"{c:12s} {len(sel):6d} {100 * len(sel) / len(idx):5.1f}% {len(v):10d} {pw}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    T = json.load(open(hv.TRAIN_JSON))
    held = set(json.load(open(hv.HELDOUT_DBS)))
    by = collections.defaultdict(list)
    for i, t in enumerate(T):
        by[t["db_id"]].append((i, t["SQL"]))
    t0 = time.time()
    gold = hv.exec_golds(by, a.seconds, a.workers)
    print(f"executed {len(gold)} ORIGINAL BIRD-train golds in {time.time() - t0:.0f}s "
          f"(limit {a.seconds}s per query, {a.workers} workers)")
    ver = verified_changed(T)
    every = list(range(len(T)))
    train = [i for i in every if T[i]["db_id"] not in held]
    rate_table(T, gold, ver, every, "all BIRD-train (reproduces run_all.log, which counted 8 s timeouts as non-errors)")
    rate_table(T, gold, ver, train, "BIRD-train outside the 9 held-out DBs (the drop list)")
    # the label study's definitions: error excluding timeouts; single 0 = one row, one column, value 0
    print("\nlabel-study definitions on all train (reference: empty 2.8% / 98.6% (n 73), sql error 3.7% / 100% (n 41), "
          "single 0 1.6% / 80.8% (n 26); reports/fanout_r1/labels/r1/run_all.log):")
    for name, fn in (("empty result", lambda r: r["status"] == "ok" and r["nrows"] == 0),
                     ("sql error", lambda r: r["status"] == "error"),
                     ("single 0 value", lambda r: r.get("pathology") == "single_zero")):
        sel = [i for i in every if fn(gold[hv.qid(i)])]
        v = [ver[i] for i in sel if i in ver]
        print(f"  {name:15s} {100 * len(sel) / len(T):4.1f}%  P(wrong) {100 * sum(v) / max(1, len(v)):5.1f}% (n {len(v)})")
    errs = collections.Counter(re.sub(r"[\"'`].*", "", gold[hv.qid(i)]["error"])[:40]
                               for i in train if gold[hv.qid(i)]["status"] == "error")
    print("\ntop error messages (train, outside held-out):", dict(errs.most_common(8)))
    per_db = collections.Counter(T[i]["db_id"] for i in train if category(gold[hv.qid(i)]))
    print("drops per DB (top 12):", dict(per_db.most_common(12)))
    drops = {hv.qid(i): category(gold[hv.qid(i)]) for i in train if category(gold[hv.qid(i)])}
    out = {"version": 1, "built": time.strftime("%Y-%m-%d"),
           "what": "BIRD-train questions outside the 9 held-out DBs whose ORIGINAL gold errors, times out, returns no "
                   "rows, or returns a single 0 / single NULL; drop them from training data and RL rewards",
           "gold_seconds": a.seconds, "id_format": "bird_train_train_<train.json index>",
           "sources": {"train_json": hv.file_sha256(hv.TRAIN_JSON), "heldout_dbs": hv.file_sha256(hv.HELDOUT_DBS)},
           "counts": dict(collections.Counter(drops.values())), "n_considered": len(train), "n": len(drops),
           "ids_sha256": hv.ids_sha256(drops), "ids": dict(sorted(drops.items(), key=lambda kv: hv._idnum(kv[0])))}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}: {len(drops)} ids {out['counts']}")


if __name__ == "__main__":
    main()
