"""VAL4 validator boxes, Mac side, CPU, stdlib only: stage the label-free held-out pools, check the lists, size the run.

  python3 scripts/val4/stage.py                    # stage data/val4/box/ + checks + ESTIMATE line
  python3 scripts/val4/stage.py --check-only       # checks + estimate, writes nothing

Staged into data/val4/box/ (pushed flat to /root/d/valhard/val4/ with data/val4/{train,val}_lists.jsonl):
  k32_pool.jsonl    R1's k=32 held-out pool (data/valhard/r1/heldout_pool.jsonl, VALHARD's pool) WITHOUT labels
  k64r_pool.jsonl   the k=64+repair merged pool (runs/phase1/repair/repair/k64_repair_pool.jsonl) WITHOUT labels
  MANIFEST.json, MANIFEST.sha256 (the box runs `sha256sum -c` over the 4 files it reads)
Label-free exactly as scripts/chains/val3_stage.py (its stage_pool: every ok / gold* / in_slots / covered key removed);
both pools hold DECISION + RESERVED rows, labels are joined back on the Mac (DECISION only).
Checks (exit 1, nothing staged): the PREREG data rule passed (data/val4/unseen_check.json status PASS), no list on a
held-out DB or with a DECISION / RESERVED id, val lists only on the 4 pinned DBs and disjoint from train, every list
DB has a card in data/sys1/bird_train. Train ids MAY repeat (the same BIRD-train question with VALHARD's in-sample list
and a fold generator's DB-unseen list: two different lists). Prints counts only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("val3_stage", ROOT / "scripts/chains/val3_stage.py")
V3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V3)

DATA = ROOT / "data/val4"
OUT = DATA / "box"
K32 = ROOT / "data/valhard/r1/heldout_pool.jsonl"
K64R = ROOT / "runs/phase1/repair/repair/k64_repair_pool.jsonl"
CARDS = ROOT / "data/sys1/bird_train/cards"
PINNED = {"citeseer", "disney", "professional_basketball", "trains"}

# time model: VALHARD r1 measured on one H100 (runs/phase1/valhard/PREREG.md VERDICT): 44.7k entries / epoch, 2 epochs
# in 1.92 h incl. 2 val passes on 179 lists (TOK 120000); k=32 pool scored in ~0.15 h, k64+repair ~2x its candidates.
REF_ENTRIES, REF_TRAIN_H = 44700, 1.92
H_SCORE = 0.15 + 0.30
H_SETUP = 0.35
PRICE = 2.73


def estimate(entries: int) -> dict:
    train_h = REF_TRAIN_H * entries / REF_ENTRIES
    max_hours = round(max(6.5, 1.3 * train_h), 1)        # VALHARD's 6.5 h unless the lists need more
    box_h = H_SETUP + train_h + H_SCORE
    worst_h = H_SETUP + max_hours + 1.5 * H_SCORE
    return {"entries": entries, "train_h": round(train_h, 2), "max_hours": max_hours, "box_h": round(box_h, 2),
            "box_usd": round(box_h * PRICE, 2), "cap_usd": min(math.ceil(worst_h * PRICE) + 2, 30)}


def scan(path: Path, bad_ids: set, held: set, cards: set, name: str) -> dict:
    ids, dbs, n, usable, entries, problems = set(), set(), 0, 0, 0, []
    for k, line in enumerate(open(V3.guard(path))):
        r = json.loads(line)
        n += 1
        i, db = r.get("id"), r.get("db_id")
        if not i or not db or not isinstance(r.get("cands"), list):
            problems.append(f"{name} line {k + 1}: missing id / db_id / cands")
            continue
        ids.add(i)
        dbs.add(db)
        if i in bad_ids:
            problems.append(f"{name}: held-out id {i} (DECISION / RESERVED)")
        if db in held:
            problems.append(f"{name}: held-out database {db}")
        if db not in cards:
            problems.append(f"{name}: no card for {db} in data/sys1/bird_train")
        it = V3.list_items(r)
        oks = [bool(c.get("ok")) for c in it]
        if any(oks) and not all(oks):
            usable += 1
            entries += len(it)
    return {"ids": ids, "dbs": dbs, "lists": n, "usable": usable, "entries": entries, "problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    train, val, uc = a.data / "train_lists.jsonl", a.data / "val_lists.jsonl", a.data / "unseen_check.json"
    for p in (train, val, uc, K32, K64R, CARDS, V3.HELDOUT_DBS, *V3.SPLITS):
        if not p.exists():
            print(f"VAL4_STAGE_MISSING {p}")
            return 1
    status = json.load(open(uc)).get("status")
    if status != "PASS":
        print(f"VAL4_STAGE_FAIL the PREREG data rule did not pass ({uc}: {status}): no validator is trained")
        return 1
    bad_ids = set()
    for s in V3.SPLITS:
        bad_ids |= set(json.load(open(s))["ids"])
    held = set(json.load(open(V3.HELDOUT_DBS)))
    cards = {p.stem for p in CARDS.glob("*.json")}
    T = scan(train, bad_ids, held, cards, "train")
    Va = scan(val, bad_ids, held, cards, "val")
    problems = T["problems"] + Va["problems"]
    problems += [f"train list on pinned val DB {d}" for d in sorted(T["dbs"] & PINNED)]
    problems += [f"val list on {d}, not a pinned val DB" for d in sorted(Va["dbs"] - PINNED)]
    both = T["ids"] & Va["ids"]
    if both:
        problems.append(f"{len(both)} ids in both train and val (e.g. {sorted(both)[:3]})")
    if problems:
        for p in problems[:30]:
            print("VAL4_STAGE_PROBLEM", p)
        print(f"VAL4_STAGE_FAIL {len(problems)} problems: nothing staged")
        return 1
    counts = {"train_lists": T["lists"], "train_usable": T["usable"], "train_entries": T["entries"],
              "val_lists": Va["lists"], "val_usable": Va["usable"], "train_dbs": len(T["dbs"]), "val_dbs": len(Va["dbs"])}
    est = estimate(T["entries"])
    print("VAL4_STAGE_COUNTS " + json.dumps(counts))
    if a.check_only:
        print("ESTIMATE " + json.dumps(est))
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    pools = {"k32": V3.stage_pool(K32, a.out / "k32_pool.jsonl"), "k64r": V3.stage_pool(K64R, a.out / "k64r_pool.jsonl")}
    files = {"train_lists.jsonl": V3.sha256(train), "val_lists.jsonl": V3.sha256(val),
             "k32_pool.jsonl": V3.sha256(a.out / "k32_pool.jsonl"), "k64r_pool.jsonl": V3.sha256(a.out / "k64r_pool.jsonl")}
    man = {"sources": {"train": V3.rel(train), "val": V3.rel(val), "k32": V3.rel(K32), "k64r": V3.rel(K64R)},
           "counts": counts, "pools": pools, "label_free": list(V3.LABEL_KEYS) + ["gold*"], "estimate": est,
           "files": files}
    V3.write_atomic(a.out / "MANIFEST.json", [json.dumps(man, indent=1)])
    V3.write_atomic(a.out / "MANIFEST.sha256", [f"{h}  {f}\n" for f, h in files.items()])
    print(f"VAL4_STAGED {V3.rel(a.out)}: pools {pools}")
    print("ESTIMATE " + json.dumps(est))
    return 0


if __name__ == "__main__":
    sys.exit(main())
