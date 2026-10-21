#!/bin/bash
# Mac side of a VAL4 validator box (runs/phase1/val4/PREREG.md): ONE box per seed (PREREG: seeds 1 and 2). Stage
# data/val4/box (scripts/val4/stage.py: label-free k=32 and k64+repair pools, list checks, the PREREG data rule, manifest,
# size estimate), leakcheck, wait for the box, push lists + pools to /root/d/valhard/val4 and the BIRD-train cards, start
# val4_box.sh detached, then watch (watch_box.sh pulls /workspace/val4 + logs into runs/phase1/val4/s<SEED>/ and deletes
# ONLY this instance id, verified).
#   bash scripts/val4/build_lists.sh                                 # first, on the Mac (after both fold boxes)
#   DRY=1 bash scripts/chains/run_val4_box.sh - 1                    # stage + checks + estimate + push plan; no ssh
#   bash scripts/chains/run_val4_box.sh <instance_id> 1               # seed 1 (CAP / MAX_HOURS from the estimate)
# Box (one H100 80 GB, created first by hand; this script never creates one):
#   python3 scripts/shade/sf.py create <cloud> <region> <type> --max-spend <CAP+2> --name luthor-val4-s<seed>
# Env: MAX_HOURS, CAP (override the estimate; PREREG cap $12 per validator box unless the lists need more), STALL_MIN (75).
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=${1:-}; SEED=${2:-}; DRY=${DRY:-0}
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "usage: bash scripts/chains/run_val4_box.sh <instance_id|-> <seed>"; exit 2; }
[ -n "$ID" ] && { [ "$ID" != - ] || [ "$DRY" = 1 ]; } || { echo "instance id required (or DRY=1 with -)"; exit 2; }
NAME=val4/s$SEED; OUT=runs/phase1/$NAME
PREREG=runs/phase1/val4/PREREG.md; MARK=PREREG-VAL4-FROZEN-2026-10-11
mkdir -p "$OUT"
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a "$OUT/watch.log"; }
[ "$DRY" = 1 ] && log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME DRY] $*"; }
grep -q "$MARK" "$PREREG" 2>/dev/null || { log "no frozen PREREG ($PREREG, $MARK): not starting"; exit 1; }
if [ "$DRY" != 1 ] && { [ -e "$OUT/VERDICT" ] || [ -d "$OUT/val4" ]; }; then
  log "$OUT already holds a run (VERDICT / val4/): move it away first; not starting"; exit 1
fi
st=$(python3 scripts/val4/stage.py 2>&1) || { echo "$st" | tail -n 30; log "staging failed: nothing sent"; exit 1; }
echo "$st" | grep -v '^ESTIMATE' | while read -r l; do log "$l"; done
est=$(echo "$st" | sed -n 's/^ESTIMATE //p')
read -r EST_MAXH EST_CAP EST_H EST_USD < <(echo "$est" | python3 -c "import json,sys; e=json.load(sys.stdin); print(e['max_hours'], e['cap_usd'], e['box_h'], e['box_usd'])")
MAX_HOURS=${MAX_HOURS:-$EST_MAXH}; CAP=${CAP:-$EST_CAP}
log "seed $SEED: estimate ${EST_H} h (\$${EST_USD}) on one H100; trainer --max-hours $MAX_HOURS; watch CAP \$$CAP"

VD=data/val4
FILES=("$VD/train_lists.jsonl" "$VD/val_lists.jsonl" "$VD/box/k32_pool.jsonl" "$VD/box/k64r_pool.jsonl"
       "$VD/box/MANIFEST.sha256" "$VD/box/MANIFEST.json")
pushes=("/root/d/valhard/val4|${FILES[*]}" "/root/d/sys1|data/sys1/bird_train")
all=(src scripts pyproject.toml "${FILES[@]}" data/sys1/bird_train)
bash scripts/vast/leakcheck.sh "${all[@]}" > /dev/null || { bash scripts/vast/leakcheck.sh "${all[@]}" | tail -n 20; log "leakcheck failed: nothing sent"; exit 1; }
log "leakcheck ok: ${#all[@]} paths, $(du -shc "${all[@]}" 2>/dev/null | tail -1 | cut -f1) on disk"
START="mkdir -p /workspace/logs && (SEED=$SEED VH_TAG=val4 MAX_HOURS=$MAX_HOURS LOSS=group TOK=120000 setsid nohup bash /workspace/luthor/scripts/chains/val4_box.sh > /workspace/logs/val4.log 2>&1 < /dev/null &)"
WATCH=(scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/val4.log VAL4_DONE /workspace/val4)
if [ "$DRY" = 1 ]; then
  for p in "${pushes[@]}"; do log "would push -> ${p%%|*}: ${p#*|}"; done
  log "would start: $START"
  log "would watch: STALL_MIN=${STALL_MIN:-75} FAIL_REGEX=VAL4_FAIL CAP=$CAP ${WATCH[*]}"
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
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') id=$ID seed=$SEED max_hours=$MAX_HOURS cap=$CAP train=$(shasum -a 256 $VD/train_lists.jsonl | cut -c1-16) val=$(shasum -a 256 $VD/val_lists.jsonl | cut -c1-16)" >> runs/phase1/val4/launches.log
CAP=$CAP STALL_MIN=${STALL_MIN:-75} FAIL_REGEX='VAL4_FAIL' exec bash "${WATCH[@]}"
