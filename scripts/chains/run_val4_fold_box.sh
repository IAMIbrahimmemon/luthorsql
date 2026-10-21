#!/bin/bash
# Mac side of a VAL4 fold generator (runs/phase1/val4/PREREG.md): ONE box per fold. Check the fold files
# (scripts/val4/folds.py --check), leakcheck, wait for the box, push (push6.sh: code + R1's plan sets + rrdbs + rare1 +
# the fold lists; nothing held-out, BIRD dev or sealed), start val4_fold_box.sh detached, then watch (watch_box.sh
# pulls /workspace/val4 + logs into runs/phase1/val4/fold<F>/ and deletes ONLY this instance id, verified).
#   DRY=1 bash scripts/chains/run_val4_fold_box.sh - 1           # checks + push plan + start command; no ssh, no spend
#   bash scripts/chains/run_val4_fold_box.sh <instance_id> 1      # fold 1 (holds the 4 pinned val DBs)
#   bash scripts/chains/run_val4_fold_box.sh <instance_id> 2      # fold 2 (both boxes can run in parallel)
# Box (one H100 80 GB per fold, created first by hand; this script never creates one):
#   python3 scripts/shade/sf.py offers H100 --n 5
#   python3 scripts/shade/sf.py create <cloud> <region> <type> --max-spend 27 --name luthor-val4-fold<F>
# Env: CAP (watch spend cap, default 25 = PREREG), STALL_MIN (90: the gen stage logs nothing for ~30 min).
# Projected (from logs): ~6.7 h per fold on an H100 PCIe (setup 0.35 + train 5.74 at 3.08 s/step + gen ~0.6), ~$18.
# Refuses to start without the frozen PREREG, or when runs/phase1/val4/fold<F>/ already holds a result.
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=${1:-}; FOLD=${2:-}; DRY=${DRY:-0}
case "$FOLD" in 1|2) ;; *) echo "usage: bash scripts/chains/run_val4_fold_box.sh <instance_id|-> <1|2>"; exit 2;; esac
[ -n "$ID" ] && { [ "$ID" != - ] || [ "$DRY" = 1 ]; } || { echo "instance id required (or DRY=1 with -)"; exit 2; }
OTHER=$((3 - FOLD)); NAME=val4/fold$FOLD; OUT=runs/phase1/$NAME; CAP=${CAP:-25}
PREREG=runs/phase1/val4/PREREG.md; MARK=PREREG-VAL4-FROZEN-2026-10-11
mkdir -p "$OUT"
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a "$OUT/watch.log"; }
[ "$DRY" = 1 ] && log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME DRY] $*"; }
grep -q "$MARK" "$PREREG" 2>/dev/null || { log "no frozen PREREG ($PREREG, $MARK): not starting"; exit 1; }
if [ "$DRY" != 1 ] && { [ -e "$OUT/VERDICT" ] || [ -d "$OUT/val4" ]; }; then
  log "$OUT already holds a run (VERDICT / val4/): move it away first; not starting"; exit 1
fi
python3 scripts/val4/folds.py --check | while read -r l; do log "$l"; done
python3 scripts/val4/folds.py --check > /dev/null || { log "data/val4 fold files missing or stale (run python3 scripts/val4/folds.py): not starting"; exit 1; }
VD=data/val4
FOLDFILES=("$VD/folds.json" "$VD/fold${FOLD}_drop.json" "$VD/fold${FOLD}_dev.json" "$VD/fold${OTHER}_drop.json")
SYS1=(data/sys1/bird_train data/sys1/spider_train data/sys1/synsql_train data/sys1/rrdbs data/sys1/templates.json
      data/sys1/templates_nums.json)
[ "$(shasum -a 256 data/rare1/plans.jsonl | cut -c1-16)" = c9ac3783cfb92e0a ] || { log "data/rare1/plans.jsonl is not R1's rare rows: not starting"; exit 1; }
pushes=("/root/d/sys1|${SYS1[*]}" "/root/d|data/rare1" "/root/d/val4|${FOLDFILES[*]}")
all=(src scripts pyproject.toml "${SYS1[@]}" data/rare1 "${FOLDFILES[@]}")
bash scripts/vast/leakcheck.sh "${all[@]}" > /dev/null || { bash scripts/vast/leakcheck.sh "${all[@]}" | tail -n 20; log "leakcheck failed: nothing sent"; exit 1; }
log "leakcheck ok: ${#all[@]} paths, $(du -shc "${all[@]}" 2>/dev/null | tail -1 | cut -f1) on disk"
log "projection: ~6.7 h on one H100 PCIe (setup 0.35 + train 6714 x 3.08 s = 5.74 + gen ~0.6), ~\$18 at \$2.73/h; CAP \$$CAP"
START="mkdir -p /workspace/logs && (FOLD=$FOLD setsid nohup bash /workspace/luthor/scripts/chains/val4_fold_box.sh > /workspace/logs/val4fold.log 2>&1 < /dev/null &)"
WATCH=(scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/val4fold.log VAL4FOLD_DONE /workspace/val4)
if [ "$DRY" = 1 ]; then
  for p in "${pushes[@]}"; do log "would push -> ${p%%|*}: ${p#*|}"; done
  log "would start: $START"
  log "would watch: STALL_MIN=${STALL_MIN:-90} FAIL_REGEX=VAL4FOLD_FAIL CAP=$CAP ${WATCH[*]}"
  exit 0
fi

for i in $(seq 1 240); do
  REFRESH=1 bash scripts/vast/ssh.sh "$ID" true >/dev/null 2>&1 && break
  [ "$i" = 240 ] && { log "ssh never came up: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
  sleep 15
done
log "ssh up"
ok=0
for t in 1 2 3; do
  good=1
  for p in "${pushes[@]}"; do
    # shellcheck disable=SC2086
    REMOTE_D=${p%%|*} bash scripts/vast/push6.sh "$ID" ${p#*|} >> "$OUT/push.log" 2>&1 || { good=0; break; }
  done
  [ $good = 1 ] && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
bash scripts/vast/ssh.sh "$ID" "$START" 2>/dev/null
log "started"
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') id=$ID fold=$FOLD cap=$CAP folds=$(shasum -a 256 $VD/folds.json | cut -c1-16)" >> runs/phase1/val4/launches.log
CAP=$CAP STALL_MIN=${STALL_MIN:-90} FAIL_REGEX='VAL4FOLD_FAIL' exec bash "${WATCH[@]}"
