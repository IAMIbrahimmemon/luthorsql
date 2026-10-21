#!/bin/bash
# VALHARD Mac-side data prep (CPU only, ~10 min, < 0.5 GB RAM): base lists -> hard pairs -> mixed lists -> held-out pool.
#   bash scripts/valhard/prep.sh r1        # R1 generator: runs/phase1/p1w/p1w/r1 (k=32 distinct BIRD-train lists)
#   bash scripts/valhard/prep.sh v4desc    # phase-3 generator after its box is pulled: runs/phase1/v4desc/r1/rr
# Writes data/valhard/<tag>/{base/,hard.jsonl,mine_stats.json,pairs_sample.jsonl,train_lists.jsonl,val_lists.jsonl,
# mix_stats.json,heldout_pool.jsonl[,base_rr/]}. BIRD-train DBs only for training; the pool is held-out DBs (blind).
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=${1:?tag: r1 | v4desc}; W=${WORKERS:-6}; PY=.venv/bin/python; OUT=data/valhard/$TAG
case "$TAG" in
  r1) POOL=runs/phase1/p1w/p1w/r1/cands_train.jsonl; HELD=runs/phase1/p1w/p1w/heldout_union.jsonl; KEEP="--keep-src r1 both"; BASE_RR="";;
  v4desc) G=runs/phase1/v4desc/r1/rr; POOL=$G/cands_train.jsonl; HELD=$G/wide/cands_dev_k32.jsonl; KEEP=""; BASE_RR=$G;;
  *) echo "tag must be r1 or v4desc"; exit 2;;
esac
for f in "$POOL" "$HELD"; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
mkdir -p "$OUT/base"
$PY scripts/val2/build_lists.py --pool "r1=$POOL" --out "$OUT/base" --workers "$W" > "$OUT/base/build.log" 2>&1
$PY scripts/valhard/mine.py --base "$OUT/base" --pool "r1=$POOL" --out "$OUT" --workers "$W" > "$OUT/mine.log" 2>&1
$PY scripts/valhard/mix.py --base "$OUT/base" --hard "$OUT/hard.jsonl" --out "$OUT" > "$OUT/mix.log" 2>&1
# shellcheck disable=SC2086
$PY scripts/valhard/pool.py --src "$HELD" $KEEP --out "$OUT/heldout_pool.jsonl"
if [ -n "$BASE_RR" ]; then   # the generator's own validator, scored on the same pool on the box as the baseline
  [ -s "$BASE_RR/rerank_head.pt" ] || { echo "missing $BASE_RR/rerank_head.pt"; exit 1; }
  rm -rf "$OUT/base_rr"; mkdir -p "$OUT/base_rr"
  cp -R "$BASE_RR/rerank_lora" "$BASE_RR/rerank_head.pt" "$OUT/base_rr/"
  [ -f "$BASE_RR/settings.json" ] && cp "$BASE_RR/settings.json" "$OUT/base_rr/"
fi
bash scripts/vast/leakcheck.sh "$OUT/train_lists.jsonl" "$OUT/val_lists.jsonl" "$OUT/heldout_pool.jsonl" ${BASE_RR:+"$OUT/base_rr"}
echo "PREP_DONE $TAG: $(wc -l < "$OUT/train_lists.jsonl") train lists, $(wc -l < "$OUT/val_lists.jsonl") val lists"
