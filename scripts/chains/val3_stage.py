"""VAL3 validator boxes (runs/phase1/val3/PREREG.md), Mac side, CPU, stdlib only: stage what every VAL3 box reads into
data/valhard/val3/ (inside leakcheck's data/valhard allowlist), check it, and size the run.

  python3 scripts/chains/val3_stage.py [--train F] [--val F]           # stage + checks + ESTIMATE line per arm
  python3 scripts/chains/val3_stage.py --check-only                     # checks + estimate, writes nothing
  python3 scripts/chains/val3_stage.py --estimate-entries 44700 --estimate-val 179   # size a hypothetical run

Staged (data/valhard/val3/, each written atomically; a file whose source is unchanged is not rewritten):
  train_lists.jsonl, val_lists.jsonl   copies of the data-v3 lists (built by the lists agent: scripts/val3/,
                                       scripts/chains/val3_lists_box.sh; defaults data/val3/{train,val}_lists.jsonl)
  k32_pool.jsonl                       R1's k=32 held-out pool (data/valhard/r1/heldout_pool.jsonl) WITHOUT labels
  k64r_pool.jsonl                      the k=64+repair merged pool (runs/phase1/repair/repair/k64_repair_pool.jsonl)
                                       WITHOUT labels
  cards/cards/<db>.json                schema cards of every train/val database that data/sys1/bird_train lacks
  MANIFEST.json, MANIFEST.sha256       sources, counts, sha256 (the box runs `sha256sum -c MANIFEST.sha256`)
Label-free = every 'ok' / 'gold*' / 'in_slots' / 'covered' key removed (row and candidate level). Both pools hold
DECISION + RESERVED rows, so the box can never print or write a label-derived number; labels are joined back on the
Mac by scripts/analysis/val3_eval.py (DECISION only).
Checks (any failure: exit 1, nothing staged): list ids unique, train and val disjoint, no DECISION / RESERVED id, no
held-out database (data/sys1/dev_dbs_bird.json), no BIRD-dev / sealed id or path, every list database has a card, no
staged card name shadowed by data/sys1/bird_train (the box loads that dir first). A concurrent run waits on a lock.
Prints counts only (no label-derived number on the held-out pools; train-list mixedness is a training-side count).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/valhard/val3"
TRAIN = ROOT / "data/val3/train_lists.jsonl"
VAL = ROOT / "data/val3/val_lists.jsonl"
K32 = ROOT / "data/valhard/r1/heldout_pool.jsonl"
K64R = ROOT / "runs/phase1/repair/repair/k64_repair_pool.jsonl"
BIRD_CARDS = ROOT / "data/sys1/bird_train/cards"  # the box's first --cards dir (VALHARD's held-out cards)
CARD_DIRS = [ROOT / "data/val3/q/bird_rr/cards", ROOT / "data/val3/q/bird_big/cards", ROOT / "data/val3/q/synsql/cards",
             ROOT / "data/sys1/synsql_train/cards", ROOT / "data/sys1/spider_train/cards"]
SPLITS = [ROOT / "data/sys1/heldout_v2/decision.json", ROOT / "data/sys1/heldout_v2/reserved.json"]
HELDOUT_DBS = ROOT / "data/sys1/dev_dbs_bird.json"
FORBIDDEN = ("bird_dev", "sealed", "gate.json", "/score_", "dev_databases")
LABEL_KEYS = ("ok", "in_slots", "covered")

# ---- time model (docs/paper_notes/research/validator/val3_trainer.md, "H100 time per arm and seed") -------------------
# VALHARD r1 measured: 44.7k entries/epoch, 2 epochs in 1.92 h incl. 2 val passes (TOK 120000, 1,736 OOM retries).
# TOK 60000: 0.62-0.76 h per epoch per 44.7k entries (central 0.70). Joint (arm B) x1.4-1.8 (central 1.6).
# Val: 8 half-epoch passes + 1 average pass on ~400 lists = ~0.2 h (A) / ~0.5 h (B, 2 orders).
# Held-out scoring: k=32 pool 8.3 min measured (A ~0.15 h with load), k64+repair ~2x the candidates; B x2.7 (2 orders,
# one prefix per question).
REF_ENTRIES, EPOCHS = 44700, 4
H_EPOCH = {"A": 0.70, "B": 0.70 * 1.6}
H_EPOCH_HI = {"A": 0.76, "B": 0.76 * 1.8}
H_VAL_PASS_PER_LIST = {"A": 0.2 / 9 / 400, "B": 0.5 / 9 / 400}
H_SCORE = {"A": 0.15 + 0.30, "B": 0.40 + 0.80}
H_SETUP = 0.35  # push + setup6 (OmniSQL-7B download) + model load
PRICE = 2.73     # $/h, Shadeform H100 (the price every earlier box paid)


def estimate(entries: int, val_lists: int) -> dict:
    out = {}
    for arm in ("A", "B"):
        scale = entries / REF_ENTRIES
        val_h = (EPOCHS * 2 + 1) * val_lists * H_VAL_PASS_PER_LIST[arm]
        train_h = EPOCHS * scale * H_EPOCH[arm] + val_h
        train_hi = EPOCHS * scale * H_EPOCH_HI[arm] + val_h * 1.3
        max_hours = round(max(1.3 * train_h, train_hi * 1.05), 1)  # >= 1.3x the central estimate and above the high end
        box_h = H_SETUP + train_h + H_SCORE[arm]
        worst_h = H_SETUP + max_hours + 1.5 * H_SCORE[arm]
        out[arm] = {"train_h": round(train_h, 2), "train_h_high": round(train_hi, 2), "max_hours": max_hours,
                    "box_h": round(box_h, 2), "box_usd": round(box_h * PRICE, 2), "worst_h": round(worst_h, 2),
                    "cap_usd": math.ceil(worst_h * PRICE) + 2}
    return out


def guard(p: Path) -> Path:
    if any(f in str(p) for f in FORBIDDEN):
        raise SystemExit(f"refusing {p}: BIRD dev / sealed / gate paths are off limits")
    return p


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def list_items(r: dict) -> list:
    """rerank_sq2.list_items: unique SQL that runs (mutants count)."""
    seen, out = set(), []
    for c in r.get("cands", []):
        s = c.get("sql")
        if s and s not in seen:
            seen.add(s)
            if c.get("runs") or c.get("mutant"):
                out.append(c)
    return out


def scan_lists(path: Path, bad_ids: set, bad_dbs: set, name: str) -> dict:
    ids, dbs, n, usable, entries, problems = set(), set(), 0, 0, 0, []
    for k, line in enumerate(open(guard(path))):
        r = json.loads(line)
        n += 1
        i, db = r.get("id"), r.get("db_id")
        if not i or not db or not isinstance(r.get("cands"), list):
            problems.append(f"{name} line {k + 1}: missing id / db_id / cands")
            continue
        if i in ids:
            problems.append(f"{name}: duplicate id {i}")
        ids.add(i)
        dbs.add(db)
        if i in bad_ids:
            problems.append(f"{name}: held-out id {i} (DECISION / RESERVED)")
        if db in bad_dbs:
            problems.append(f"{name}: held-out database {db}")
        if any(f in str(i) for f in FORBIDDEN):
            problems.append(f"{name}: forbidden id {i}")
        it = list_items(r)
        oks = [bool(c.get("ok")) for c in it]
        if any(oks) and not all(oks):
            usable += 1
            entries += len(it)
    return {"ids": ids, "dbs": dbs, "lists": n, "usable": usable, "entries": entries, "problems": problems}


def strip_labels(obj: dict) -> dict:
    return {k: v for k, v in obj.items() if k not in LABEL_KEYS and not k.startswith("gold")}


def write_atomic(dst: Path, lines) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "w") as f:
        for x in lines:
            f.write(x)
    os.replace(tmp, dst)


def stage_pool(src: Path, dst: Path) -> dict:
    n = nc = 0

    def rows():
        nonlocal n, nc
        for line in open(guard(src)):
            r = json.loads(line)
            r = strip_labels(r)
            r["cands"] = [strip_labels(c) for c in r["cands"]]
            n += 1
            nc += len(r["cands"])
            yield json.dumps(r) + "\n"
    write_atomic(dst, rows())
    with open(dst) as f:  # nothing label-like may survive
        for line in f:
            r = json.loads(line)
            assert not any(k in r for k in LABEL_KEYS) and not any(k.startswith("gold") for k in r), dst
            assert not any(k in c for c in r["cands"] for k in LABEL_KEYS), dst
    return {"rows": n, "cands": nc}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path(os.environ.get("TRAIN", TRAIN)))
    ap.add_argument("--val", type=Path, default=Path(os.environ.get("VAL", VAL)))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check-only", action="store_true", help="checks + estimate; stage nothing")
    ap.add_argument("--estimate-entries", type=int, help="only print the estimate for this many train entries")
    ap.add_argument("--estimate-val", type=int, default=400, help="with --estimate-entries: val lists")
    a = ap.parse_args(argv)
    if a.estimate_entries:
        print("ESTIMATE " + json.dumps({"entries": a.estimate_entries, "val_lists": a.estimate_val,
                                        **estimate(a.estimate_entries, a.estimate_val)}))
        return 0
    for p in (a.train, a.val, K32, K64R, BIRD_CARDS, HELDOUT_DBS, *SPLITS):
        if not p.exists():
            print(f"VAL3_STAGE_MISSING {p}")
            return 1
    bad_ids = set()
    for s in SPLITS:
        bad_ids |= set(json.load(open(s))["ids"])
    bad_dbs = set(json.load(open(HELDOUT_DBS)))
    T = scan_lists(a.train, bad_ids, bad_dbs, "train")
    V = scan_lists(a.val, bad_ids, bad_dbs, "val")
    problems = T["problems"] + V["problems"]
    both = T["ids"] & V["ids"]
    if both:
        problems.append(f"{len(both)} ids in both train and val (e.g. {sorted(both)[:3]})")
    shared_dbs = sorted(T["dbs"] & V["dbs"])
    bird = {p.stem for p in BIRD_CARDS.glob("*.json")}
    need, cards = sorted((T["dbs"] | V["dbs"]) - bird), {}
    for db in need:
        src = next((d / f"{db}.json" for d in CARD_DIRS if (d / f"{db}.json").exists()), None)
        if src is None:
            problems.append(f"no schema card for list database {db}")
        else:
            cards[db] = src
    if problems:
        for p in problems[:30]:
            print("VAL3_STAGE_PROBLEM", p)
        print(f"VAL3_STAGE_FAIL {len(problems)} problems: nothing staged")
        return 1
    counts = {"train_lists": T["lists"], "train_usable": T["usable"], "train_entries": T["entries"],
              "val_lists": V["lists"], "val_usable": V["usable"], "train_dbs": len(T["dbs"]), "val_dbs": len(V["dbs"]),
              "dbs_in_train_and_val": len(shared_dbs), "cards_from_bird_train": len((T["dbs"] | V["dbs"]) & bird),
              "cards_staged": len(cards)}
    est = estimate(T["entries"], V["usable"])
    print("VAL3_STAGE_COUNTS " + json.dumps(counts))
    if shared_dbs:
        print(f"note: {len(shared_dbs)} databases have both train and val lists (e.g. {shared_dbs[:5]})")
    if a.check_only:
        print("ESTIMATE " + json.dumps({"entries": T["entries"], "val_lists": V["usable"], **est}))
        return 0

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out.parent / f".{a.out.name}.stage.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)  # parallel launches stage one after another
        man_p = a.out / "MANIFEST.json"
        old = json.loads(man_p.read_text()) if man_p.exists() else {}
        src_sha = {"train": sha256(a.train), "val": sha256(a.val), "k32": sha256(K32), "k64r": sha256(K64R)}
        files = {}
        for key, src, dst in (("train", a.train, "train_lists.jsonl"), ("val", a.val, "val_lists.jsonl")):
            d = a.out / dst
            if not (d.exists() and old.get("sources", {}).get(key, {}).get("sha256") == src_sha[key]):
                tmp = d.with_suffix(".jsonl.tmp")
                shutil.copyfile(src, tmp)
                os.replace(tmp, d)
        pools = {}
        for key, src, dst in (("k32", K32, "k32_pool.jsonl"), ("k64r", K64R, "k64r_pool.jsonl")):
            d = a.out / dst
            if d.exists() and old.get("sources", {}).get(key, {}).get("sha256") == src_sha[key]:
                pools[key] = old["pools"][key]
            else:
                pools[key] = stage_pool(src, d)
        cd = a.out / "cards" / "cards"
        cd.mkdir(parents=True, exist_ok=True)
        for p in cd.glob("*.json"):  # a card no list needs any more
            if p.stem not in cards:
                p.unlink()
        for db, src in cards.items():  # rewrite only what changed (a parallel launch may be pushing this dir)
            d = cd / f"{db}.json"
            if not (d.exists() and d.read_bytes() == src.read_bytes()):
                shutil.copyfile(src, d.with_suffix(".json.tmp"))
                os.replace(d.with_suffix(".json.tmp"), d)
        staged = ["train_lists.jsonl", "val_lists.jsonl", "k32_pool.jsonl", "k64r_pool.jsonl"]
        staged += [f"cards/cards/{db}.json" for db in sorted(cards)]
        for f in staged:
            files[f] = sha256(a.out / f)
        man = {"sources": {"train": {"path": rel(a.train.resolve()), "sha256": src_sha["train"]},
                           "val": {"path": rel(a.val.resolve()), "sha256": src_sha["val"]},
                           "k32": {"path": rel(K32), "sha256": src_sha["k32"]},
                           "k64r": {"path": rel(K64R), "sha256": src_sha["k64r"]}},
               "counts": counts, "pools": pools, "label_free": list(LABEL_KEYS) + ["gold*"],
               "card_sources": {db: rel(p) for db, p in cards.items()},
               "estimate": est, "files": files}
        write_atomic(man_p, [json.dumps(man, indent=1)])
        write_atomic(a.out / "MANIFEST.sha256", [f"{h}  {f}\n" for f, h in files.items()])
    print(f"VAL3_STAGED {rel(a.out)}: pools {pools}, "
          f"{len(cards)} extra cards")
    print("ESTIMATE " + json.dumps({"entries": T["entries"], "val_lists": V["usable"], **est}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
