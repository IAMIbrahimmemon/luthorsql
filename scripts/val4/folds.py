"""VAL4 step 1 (Mac, CPU, stdlib only): split the BIRD-train databases into 2 folds for the cross-fitted generators.

  python3 scripts/val4/folds.py [--seed 0] [--out data/val4]      # writes folds.json + the per-fold DB lists
  python3 scripts/val4/folds.py --check                            # recompute and compare with what is on disk

runs/phase1/val4/PREREG.md: the 60 BIRD-train DBs (data/raw/bird_train/train.json db_id) outside
data/sys1/dev_dbs_bird.json (the 9 held-out DBs: DECISION / RESERVED + retail_world; never trained on by a fold
generator, never put in a fold) split into 2 folds balanced by question count. The 4 pinned validator-val DBs
(citeseer, disney, professional_basketball, trains) stay together, in fold 1 by convention.

Balance: every BIRD-train question counts (the PREREG rule) and, with equal weight, the questions on the DBs whose
SQLite file the box gets (data/sys1/rrdbs: the only DBs a fold generator can be evaluated and listed on; 13 large
BIRD-train DBs are not in it). Score = |q1 - q2| / Q + |s1 - s2| / S; seeded random restarts + single-DB moves/swaps,
so the split is deterministic for a seed.

Written (plain JSON lists, the format both train_sq2.py --drop-dbs and --dev-dbs accept; rerank_sq2.py gen --dev-dbs
reads a plain list too):
  folds.json           the split, counts, sources, and which file is used where
  fold<F>_drop.json    fold F's DBs + the 9 held-out DBs   -> train_sq2.py --drop-dbs for fold generator F
                       (and rerank_sq2.py gen --split train --dev-dbs for the OTHER fold's generator: its complement)
  fold<F>_dev.json     fold F's DBs that have an SQLite on the box (data/sys1/rrdbs) -> train_sq2.py --dev-dbs
                       (checkpoint selection on the left-out fold; a dev DB without its SQLite would score 0 on every
                       question and only dilute the selection metric)
  fold<F>_dbs.json     every fold-F DB (record; the large DBs are dropped from generator F but never listed)
data/sys1/contaminated_dbs.json (formula_1, soccer_1: Spider train DBs) is NOT added: R1 (runs/r1r2/r1/args.json)
trained with them, so the fold generators keep R1's behaviour.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/raw/bird_train/train.json"
HELDOUT = ROOT / "data/sys1/dev_dbs_bird.json"
RRDBS = ROOT / "data/sys1/rrdbs"
OUT = ROOT / "data/val4"
PINNED = ("citeseer", "disney", "professional_basketball", "trains")
RESTARTS = 400


def db_counts(train_path: Path = TRAIN) -> collections.Counter:
    return collections.Counter(r["db_id"] for r in json.load(open(train_path)))


def sqlite_dbs(root: Path = RRDBS) -> set[str]:
    return {p.name for p in root.iterdir() if p.is_dir() and (p / f"{p.name}.sqlite").exists()} if root.is_dir() else set()


def split(counts: dict[str, int], held: set[str], with_sqlite: set[str], seed: int = 0,
          pinned: tuple[str, ...] = PINNED) -> tuple[list[str], list[str]]:
    """Two folds of the non-held-out DBs; the pinned DBs move as one unit and land in fold 1."""
    dbs = sorted(d for d in counts if d not in held)
    missing = [p for p in pinned if p not in dbs]
    if missing:
        raise SystemExit(f"pinned val DBs not among the BIRD-train DBs: {missing}")
    units = [tuple(pinned)] + [(d,) for d in dbs if d not in pinned]
    qa = [sum(counts[d] for d in u) for u in units]
    qs = [sum(counts[d] for d in u if d in with_sqlite) for u in units]
    QA, QS = max(sum(qa), 1), max(sum(qs), 1)

    def score(side):
        da = sum(q if s else -q for q, s in zip(qa, side))
        ds = sum(q if s else -q for q, s in zip(qs, side))
        return abs(da) / QA + abs(ds) / QS

    rnd = random.Random(seed)
    best, best_s = None, float("inf")
    n = len(units)
    for _ in range(RESTARTS):
        side = [rnd.random() < 0.5 for _ in range(n)]
        s = score(side)
        improved = True
        while improved:  # first-improvement local search: single moves, then pair swaps
            improved = False
            for i in range(n):
                side[i] = not side[i]
                t = score(side)
                if t < s - 1e-12:
                    s, improved = t, True
                else:
                    side[i] = not side[i]
            if improved:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    if side[i] == side[j]:
                        continue
                    side[i], side[j] = side[j], side[i]
                    t = score(side)
                    if t < s - 1e-12:
                        s, improved = t, True
                    else:
                        side[i], side[j] = side[j], side[i]
        if s < best_s - 1e-12:
            best, best_s = list(side), s
    f1 = sorted(d for u, s in zip(units, best) if s == best[0] for d in u)
    f2 = sorted(d for u, s in zip(units, best) if s != best[0] for d in u)
    return f1, f2


def build(seed: int = 0, train_path: Path = TRAIN, held_path: Path = HELDOUT, rr: Path = RRDBS) -> dict:
    counts = db_counts(train_path)
    held = set(json.load(open(held_path)))
    with_sqlite = sqlite_dbs(rr)
    f1, f2 = split(counts, held, with_sqlite, seed)
    folds = {}
    for k, f in (("1", f1), ("2", f2)):
        folds[k] = {"dbs": f, "n_dbs": len(f), "questions": sum(counts[d] for d in f),
                    "dev_dbs": [d for d in f if d in with_sqlite],
                    "n_dev_dbs": sum(d in with_sqlite for d in f),
                    "questions_with_sqlite": sum(counts[d] for d in f if d in with_sqlite),
                    "no_sqlite_on_box": [d for d in f if d not in with_sqlite]}
    return {"seed": seed, "pinned_val_dbs": list(PINNED), "pinned_fold": "1",
            "sources": {"questions": str(train_path.relative_to(ROOT)) if train_path.is_relative_to(ROOT) else str(train_path),
                        "heldout_dbs": "data/sys1/dev_dbs_bird.json (excluded: in every drop list, never in a fold)",
                        "sqlite_on_box": "data/sys1/rrdbs"},
            "n_bird_train_dbs": len(counts), "n_heldout_dbs": len(held & set(counts)),
            "n_fold_dbs": len(f1) + len(f2), "questions": sum(counts[d] for d in f1 + f2),
            "folds": folds,
            "files": {"drop": "fold<F>_drop.json = fold F + held-out DBs (train_sq2 --drop-dbs; gen --dev-dbs of the other fold)",
                      "dev": "fold<F>_dev.json = fold F DBs with an SQLite on the box (train_sq2 --dev-dbs)",
                      "dbs": "fold<F>_dbs.json = every fold F DB"},
            "contaminated_dbs": "not dropped (R1 trained with them; runs/r1r2/r1/args.json has no drop_dbs)"}


def files_for(info: dict, held: set[str]) -> dict[str, list[str]]:
    out = {}
    for k, f in info["folds"].items():
        out[f"fold{k}_drop.json"] = sorted(set(f["dbs"]) | held)
        out[f"fold{k}_dev.json"] = list(f["dev_dbs"])
        out[f"fold{k}_dbs.json"] = list(f["dbs"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="recompute; exit 1 if the files on disk differ")
    a = ap.parse_args(argv)
    info = build(a.seed)
    held = set(json.load(open(HELDOUT)))
    files = files_for(info, held)
    files["folds.json"] = info
    if a.check:
        bad = [n for n, v in files.items() if not (a.out / n).exists() or json.loads((a.out / n).read_text()) != v]
        print("VAL4_FOLDS_" + ("MISMATCH " + " ".join(bad) if bad else "OK"))
        return 1 if bad else 0
    a.out.mkdir(parents=True, exist_ok=True)
    for n, v in files.items():
        tmp = a.out / (n + ".tmp")
        tmp.write_text(json.dumps(v, indent=1) + "\n")
        os.replace(tmp, a.out / n)
    for k, f in info["folds"].items():
        print(f"fold {k}: {f['n_dbs']} DBs, {f['questions']} questions; {f['n_dev_dbs']} DBs / "
              f"{f['questions_with_sqlite']} questions with an SQLite on the box"
              + (" (pinned val DBs)" if k == info["pinned_fold"] else ""))
    print(f"VAL4_FOLDS {info['n_fold_dbs']} DBs, {info['questions']} questions -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
