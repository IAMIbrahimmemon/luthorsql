#!/bin/bash
# VAL4 Mac-side list build (CPU only, ~15 min, < 1 GB RAM), after both fold boxes are pulled (runs/phase1/val4/PREREG.md):
#   bash scripts/val4/build_lists.sh
# Inputs: the fold generators' labelled k=32 lists (F1 / F2, default runs/phase1/val4/fold<F>/val4/fold<F>/cands_fold.jsonl
# as watch_box.sh pulls them) and VALHARD's train lists (data/valhard/r1/train_lists.jsonl).
#   1 pool   scripts/val4/lists.py pool: fold checks (fold DBs only, no held-out DB), the union of both folds, raw greedy
#            right per fold; the PREREG data rule (greedy right <= 68%) -> data/val4/unseen_check.json. STOP = exit 3
#            (report and stop: no validator; BUILD_ANYWAY=1 still builds the lists, for reading only).
#   2 prep   VALHARD's steps exactly as scripts/valhard/prep.sh runs them, defaults unchanged (BIRD-Verified relabel,
#            one live representative per result class, cap 12, x3 weight on greedy-wrong lists, near-miss mutants;
#            mined + synthetic hard negatives, LIMIT mutants dropped), into data/val4/work/.
#   3 merge  scripts/val4/lists.py merge: data/val4/train_lists.jsonl = fold lists on non-val DBs + VALHARD's train
#            lists; data/val4/val_lists.jsonl = fold lists on the 4 pinned val DBs (not in train); lists_stats.json.
# Then leakcheck of the outputs. Prints counts and BIRD-train rates only (no held-out data is read).
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${WORKERS:-6}; PY=${PY:-.venv/bin/python}; OUT=${OUT:-data/val4}; WORK=$OUT/work  # OUT: a scratch dir for smoke runs
F1=${F1:-runs/phase1/val4/fold1/val4/fold1/cands_fold.jsonl}
F2=${F2:-runs/phase1/val4/fold2/val4/fold2/cands_fold.jsonl}
OLD=${OLD:-data/valhard/r1/train_lists.jsonl}
for f in "$F1" "$F2" "$OLD" data/val4/fold1_dev.json data/val4/fold2_dev.json; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
python3 scripts/val4/folds.py --check
mkdir -p "$WORK/base"
POOL=$WORK/cands_folds.jsonl
rc=0
python3 scripts/val4/lists.py pool --fold "1=$F1" --fold "2=$F2" --out "$POOL" | tee "$WORK/pool.log" || rc=$?
python3 - "$WORK/pool.log" "$OUT/unseen_check.json" <<'PY'
import json, re, sys
txt = open(sys.argv[1]).read()
m = re.search(r"UNSEEN_CHECK (\w+): greedy right ([0-9.]+)% over (\d+)", txt)
st = {"status": m.group(1), "greedy_right": float(m.group(2)), "questions": int(m.group(3)), "rule": "<= 68.0",
      "folds": re.findall(r"^fold \d: .*$", txt, re.M)} if m else {"status": "FAIL", "log": txt[-2000:]}
json.dump(st, open(sys.argv[2], "w"), indent=1)
PY
case "$rc" in
  0) ;;
  3) echo "UNSEEN_CHECK STOP: the fold lists are not DB-unseen-like (PREREG: report and stop, no validator)"
     [ "${BUILD_ANYWAY:-0}" = 1 ] || exit 3
     echo "BUILD_ANYWAY=1: building the lists for reading only (run_val4_box.sh refuses them)";;
  *) echo "pool check failed (rc $rc)"; exit 1;;
esac
# VALHARD's prep steps (scripts/valhard/prep.sh), the pool named r1 as there (the R1 recipe wrote it)
$PY scripts/val2/build_lists.py --pool "r1=$POOL" --out "$WORK/base" --workers "$W" > "$WORK/base/build.log" 2>&1
$PY scripts/valhard/mine.py --base "$WORK/base" --pool "r1=$POOL" --out "$WORK" --workers "$W" > "$WORK/mine.log" 2>&1
$PY scripts/valhard/mix.py --base "$WORK/base" --hard "$WORK/hard.jsonl" --out "$WORK" > "$WORK/mix.log" 2>&1
python3 scripts/val4/lists.py merge --work "$WORK" --old "$OLD" --out "$OUT"
python3 - "$OUT/lists_stats.json" "$OUT/unseen_check.json" <<'PY'
import json, sys
s, u = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
print(f"raw greedy right (fold k=32 lists, gen labels): {u['greedy_right']}% over {u['questions']} questions [{u['status']}]")
for k in ("train_fold", "train_valhard", "train", "val"):
    v = s[k]
    print(f"{k:14s} {v['lists']:5d} lists on {v['dbs']:2d} DBs, {v['mean_entries']} entries/list, "
          f"list greedy right {v['list_greedy_right']}% (weight-3 share {v['weight3_share']})")
PY
[ "$OUT" != data/val4 ] || bash scripts/vast/leakcheck.sh "$OUT/train_lists.jsonl" "$OUT/val_lists.jsonl" > /dev/null
echo "VAL4_LISTS_DONE: $(wc -l < "$OUT/train_lists.jsonl" | tr -d ' ') train lists, $(wc -l < "$OUT/val_lists.jsonl" | tr -d ' ') val lists"
