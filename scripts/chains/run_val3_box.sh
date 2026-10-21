#!/bin/bash
# Mac side of VAL3 (runs/phase1/val3/PREREG.md): ONE box per arm + seed. Stage data/valhard/val3 (val3_stage.py: data-v3
# lists, label-free held-out pools, extra schema cards, manifest), size the run (MAX_HOURS ~1.3x the estimate, CAP),
# leakcheck, wait for the box, push, start val3_box.sh detached, then watch (watch_box.sh pulls /workspace/val3 + logs
# into runs/phase1/val3/<ARM>_s<SEED>/ and deletes ONLY this instance id, verified).
#   DRY=1 bash scripts/chains/run_val3_box.sh - A 1                # stage + checks + estimate + push plan; no ssh, no spend
#   bash scripts/chains/run_val3_box.sh <instance_id> A 1           # arm A seed 1 (CAP / MAX_HOURS from the estimate)
#   bash scripts/chains/run_val3_box.sh <instance_id> B 1           # arm B (joint) seed 1
# Env: TRAIN / VAL (data-v3 list files; default data/val3/{train,val}_lists.jsonl), MAX_HOURS, CAP (override the
#      estimate), TOK (60000), STALL_MIN (90).
# Box (one H100 80 GB per run, created first by hand; this script never creates one):
#   python3 scripts/shade/sf.py offers H100 --n 5
#   python3 scripts/shade/sf.py create <cloud> <region> <type> --max-spend <CAP+5> --name luthor-val3-<arm>s<seed>
# Refuses to start without the frozen PREREG, or when runs/phase1/val3/<ARM>_s<SEED>/ already holds a result.
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=${1:-}; ARM=${2:-}; SEED=${3:-}; DRY=${DRY:-0}
case "$ARM" in A|B) ;; *) echo "usage: bash scripts/chains/run_val3_box.sh <instance_id|-> <A|B> <seed>"; exit 2;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer"; exit 2; }
[ -n "$ID" ] && { [ "$ID" != - ] || [ "$DRY" = 1 ]; } || { echo "instance id required (or DRY=1 with -)"; exit 2; }
NAME=val3/${ARM}_s${SEED}; OUT=runs/phase1/$NAME; TOK=${TOK:-60000}
PREREG=runs/phase1/val3/PREREG.md; MARK=PREREG-VAL3-FROZEN-2026-10-11
mkdir -p "$OUT"
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a "$OUT/watch.log"; }
[ "$DRY" = 1 ] && log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME DRY] $*"; }
grep -q "$MARK" "$PREREG" 2>/dev/null || { log "no frozen PREREG ($PREREG, $MARK): not starting"; exit 1; }
if [ "$DRY" != 1 ] && { [ -e "$OUT/VERDICT" ] || [ -d "$OUT/val3" ]; }; then
  log "$OUT already holds a run (VERDICT / val3/): move it away first; not starting"; exit 1
fi
TRAIN=${TRAIN:-data/val3/train_lists.jsonl}; VAL=${VAL:-data/val3/val_lists.jsonl}
for f in "$TRAIN" "$VAL"; do
  [ -s "$f" ] || { log "missing $f (data v3 not built yet: scripts/chains/val3_lists_box.sh); not starting"; exit 1; }
done

# ---- stage + check + estimate (a parallel launch waits on the stage lock) ---------------------------------------------
st=$(python3 scripts/chains/val3_stage.py --train "$TRAIN" --val "$VAL" 2>&1) || { echo "$st" | tail -n 30; log "staging failed: nothing sent"; exit 1; }
echo "$st" | grep -v '^ESTIMATE' | while read -r l; do log "$l"; done
est=$(echo "$st" | sed -n 's/^ESTIMATE //p')
read -r EST_MAXH EST_CAP EST_H EST_USD < <(echo "$est" | python3 -c "import json,sys; e=json.load(sys.stdin)['$ARM']; print(e['max_hours'], e['cap_usd'], e['box_h'], e['box_usd'])")
MAX_HOURS=${MAX_HOURS:-$EST_MAXH}; CAP=${CAP:-$EST_CAP}
log "arm $ARM seed $SEED: estimate ${EST_H} h (\$${EST_USD}) on one H100; trainer --max-hours $MAX_HOURS; watch CAP \$$CAP"

pushes=("/root/d/sys1|data/sys1/bird_train" "/root/d/valhard|data/valhard/val3")
all=(src scripts pyproject.toml data/sys1/bird_train data/valhard/val3)
bash scripts/vast/leakcheck.sh "${all[@]}" > /dev/null || { bash scripts/vast/leakcheck.sh "${all[@]}" | tail -n 20; log "leakcheck failed: nothing sent"; exit 1; }
log "leakcheck ok: ${#all[@]} paths, $(du -shc "${all[@]}" 2>/dev/null | tail -1 | cut -f1) on disk"
START="mkdir -p /workspace/logs && (ARM=$ARM SEED=$SEED MAX_HOURS=$MAX_HOURS TOK=$TOK setsid nohup bash /workspace/luthor/scripts/chains/val3_box.sh > /workspace/logs/val3.log 2>&1 < /dev/null &)"
if [ "$DRY" = 1 ]; then
  for p in "${pushes[@]}"; do log "would push -> ${p%%|*}: ${p#*|} ($(du -sh "${p#*|}" | cut -f1))"; done
  log "would start: $START"
  log "would watch: STALL_MIN=${STALL_MIN:-90} FAIL_REGEX=VAL3_FAIL CAP=$CAP watch_box.sh <id> $NAME /workspace/logs/val3.log VAL3_DONE /workspace/val3"
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
    REMOTE_D=${p%%|*} bash scripts/vast/push6.sh "$ID" ${p#*|} >> "$OUT/push.log" 2>&1 || { good=0; break; }
  done
  [ $good = 1 ] && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
bash scripts/vast/ssh.sh "$ID" "$START" 2>/dev/null
log "started"
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') id=$ID arm=$ARM seed=$SEED max_hours=$MAX_HOURS cap=$CAP tok=$TOK train=$(shasum -a 256 data/valhard/val3/train_lists.jsonl | cut -c1-16) val=$(shasum -a 256 data/valhard/val3/val_lists.jsonl | cut -c1-16) eval_sha=$(shasum -a 256 scripts/analysis/val3_eval.py | cut -c1-16)" >> runs/phase1/val3/launches.log
CAP=$CAP STALL_MIN=${STALL_MIN:-90} FAIL_REGEX='VAL3_FAIL' exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/val3.log VAL3_DONE /workspace/val3
