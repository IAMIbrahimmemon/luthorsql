"""Mac side, CPU: stage data/ff1 for the ff1 box chain (scripts/chains/ff1_box.sh; plan:
docs/paper_notes/research/freeform_plan_2026-10-10.md, steps S1 + S2). DECISION + BIRD-train only: no RESERVED record
and no BIRD-dev file is written here.

  uv run python scripts/freeform/ff1_stage.py            # -> data/ff1/{decision_ids.json, decision_pool.jsonl,
                                                         #    bug_train_ids.json, dbs.txt[, extra_cands.jsonl]}

- decision_ids.json: copy of data/sys1/heldout_v2/decision.json (796 ids, 8 held-out DBs).
- decision_pool.jsonl: the R1 u R2 wide grammar pool (runs/phase1/p1w/p1w/heldout_union.jsonl, unscored) restricted to
  DECISION ids, any "rr" removed (S2 rescores it). RESERVED records are dropped, never read further.
- bug_train_ids.json: BIRD-train questions on the NON-held-out DBs whose ORIGINAL gold is a BIRD-Verified-fixed bug
  (probe.bug_questions, 5 s gold limit; ~5 min, cached). "ids" keeps only questions whose DB is <= --max-db-mb (the
  push budget); "ids_all" has all of them.
- dbs.txt: DB names the box needs (DECISION DBs + the kept bug-set DBs), pushed from data/raw/bird_train/train_databases.
- extra_cands.jsonl: OPTIONAL arithmetic mutants (--extra, default data/pool/decision_arith.jsonl) restricted to
  DECISION ids; removed when the source file is absent.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/freeform"))
import probe  # noqa: E402

HV2 = ROOT / "data/sys1/heldout_v2"
TRAIN = ROOT / "data/raw/bird_train/train.json"
DBROOT = ROOT / "data/raw/bird_train/train_databases"


def db_mb(db: str) -> float:
    return os.path.getsize(DBROOT / db / f"{db}.sqlite") / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=ROOT / "runs/phase1/p1w/p1w/heldout_union.jsonl")
    ap.add_argument("--extra", type=Path, default=ROOT / "data/pool/decision_arith.jsonl")
    ap.add_argument("--max-db-mb", type=float, default=1000.0, help="bug-set DBs larger than this are not pushed")
    ap.add_argument("--out", type=Path, default=ROOT / "data/ff1")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    dec = json.loads((HV2 / "decision.json").read_text())
    ids = list(dec["ids"])
    keep = set(ids)
    reserved = set(json.loads((HV2 / "reserved.json").read_text())["ids"])
    assert not keep & reserved, "DECISION and RESERVED overlap"
    (a.out / "decision_ids.json").write_text(json.dumps(dec))

    n_in = n_out = 0
    with open(a.out / "decision_pool.jsonl", "w") as f:
        for line in open(a.pool):
            n_in += 1
            r = json.loads(line)
            if r["id"] not in keep:
                continue
            r["cands"] = [{k: v for k, v in c.items() if k != "rr"} for c in r["cands"]]
            f.write(json.dumps(r) + "\n")
            n_out += 1
    print(f"decision_pool: {n_out} DECISION records of {n_in} (the rest RESERVED, dropped)")

    tr = json.loads(TRAIN.read_text())
    dec_dbs = sorted({tr[int(i.rsplit('_', 1)[1])]["db_id"] for i in ids})
    bf = a.out / "bug_train_ids.json"
    if bf.exists() and "ids_all" in json.loads(bf.read_text()):
        bug = json.loads(bf.read_text())
    else:
        u = json.loads((HV2 / "universe.json").read_text())
        held = {r["db_id"] for r in u["rows"]} | set(u["excluded_dbs"])
        cand = {f"bird_train_train_{i}" for i, r in enumerate(tr) if r["db_id"] not in held}
        b = probe.bug_questions(cand, seconds=5.0)
        all_ids = sorted(b, key=lambda k: int(k.rsplit("_", 1)[1]))
        bug = {"built": "2026-10-10", "what": "BIRD-train questions on non-held-out DBs whose ORIGINAL gold is a "
               "BIRD-Verified-fixed bug (probe.bug_questions, 5 s gold limit)", "held_dbs": sorted(held),
               "ids_all": all_ids, "sigs": b}
    dbs_of = {i: tr[int(i.rsplit("_", 1)[1])]["db_id"] for i in bug["ids_all"]}
    bug["max_db_mb"] = a.max_db_mb
    bug["ids"] = [i for i in bug["ids_all"] if db_mb(dbs_of[i]) <= a.max_db_mb]
    big = sorted({dbs_of[i] for i in bug["ids_all"]} - {dbs_of[i] for i in bug["ids"]})
    bug["dropped_large_dbs"] = big
    bug["n"], bug["n_all"] = len(bug["ids"]), len(bug["ids_all"])
    assert not set(bug["ids_all"]) & (keep | reserved), "bug set overlaps the held-out ids"
    bf.write_text(json.dumps(bug, indent=1))
    bug_dbs = sorted({dbs_of[i] for i in bug["ids"]})
    print(f"bug set: {bug['n']} of {bug['n_all']} questions on {len(bug_dbs)} DBs (dropped DBs > {a.max_db_mb:.0f} MB: "
          f"{', '.join(big)})")

    dbs = dec_dbs + [d for d in bug_dbs if d not in dec_dbs]
    (a.out / "dbs.txt").write_text("\n".join(dbs) + "\n")
    print(f"dbs.txt: {len(dbs)} DBs, {sum(db_mb(d) for d in dbs) / 1e3:.2f} GB")

    ex = a.out / "extra_cands.jsonl"
    if a.extra.exists():
        n = collections.Counter()
        with open(ex, "w") as f:
            for line in open(a.extra):
                r = json.loads(line)
                n["in"] += 1
                if r["id"] in keep:
                    f.write(json.dumps(r) + "\n")
                    n["kept"] += 1
        print(f"extra_cands: {n['kept']} DECISION records of {n['in']} from {a.extra}")
    elif ex.exists():
        ex.unlink()
        print(f"extra_cands: {a.extra} absent, stale copy removed")
    else:
        print(f"extra_cands: {a.extra} absent (optional)")


if __name__ == "__main__":
    main()
