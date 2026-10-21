#!/bin/bash
# Mac side: watch one Shadeform box running a detached stage script, pull its results, then DELETE it (verified).
#   CAP=8 bash scripts/chains/watch_box.sh <instance_id> <name> <remote_log> <done_regex> <remote_dir>...
# Every POLL seconds (default 120):
#   - instance status and cost (sf.py info): cost >= CAP dollars  -> pull, delete (spend guard on top of --max-spend)
#   - remote log: a line matching <done_regex>                   -> pull, delete (finished)
#                 a line matching FAIL_REGEX (default _FAIL|FATAL|Traceback) -> pull, delete (failed)
#                 log unchanged for STALL_MIN minutes (default 45) -> pull, delete (stalled)
#   - box gone / unreachable for 20 polls                         -> stop watching (pulled results stay)
# Results land in runs/phase1/<name>/ (remote dirs + the logs dir). Status lines: runs/phase1/<name>/watch.log,
# final verdict in runs/phase1/<name>/VERDICT (finished | failed | stalled | cap | lost).
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=$1; NAME=$2; RLOG=$3; DONE=$4; shift 4; DIRS=("$@")
CAP=${CAP:?CAP dollars required}; POLL=${POLL:-120}; STALL_MIN=${STALL_MIN:-45}
FAIL_REGEX=${FAIL_REGEX:-'_FAIL|FATAL|Traceback'}
OUT=runs/phase1/$NAME; mkdir -p "$OUT"
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a "$OUT/watch.log"; }
SF=(python3 scripts/shade/sf.py)
pull() {
  ( source scripts/vast/ssh.sh "$ID" >/dev/null 2>&1 || exit 1
    for d in "${DIRS[@]}" "$(dirname "$RLOG")"; do
      mkdir -p "$OUT/$(basename "$d")"
      "${RS[@]}" "$H:$d/" "$OUT/$(basename "$d")/" >/dev/null 2>&1 || echo "pull $d failed"
    done )
}
finish() {
  local verdict=$1
  log "verdict $verdict: pulling"
  for t in 1 2 3; do pull && break; log "pull attempt $t failed"; sleep 30; done
  log "deleting"
  for t in 1 2 3; do
    if CONFIRM=yes bash scripts/vast/destroy.sh "$ID" 2>&1 | tee -a "$OUT/watch.log" | grep -q VERIFIED_DELETED; then
      echo "$verdict" > "$OUT/VERDICT"; log "VERIFIED_DELETED"; exit 0
    fi
    log "delete attempt $t not verified; retrying"; sleep 60
  done
  echo "$verdict DELETE_UNVERIFIED" > "$OUT/VERDICT"; log "DELETE_UNVERIFIED: check sf.py list"; exit 1
}
last_size=-1; last_change=$(date +%s); miss=0
log "watching $ID (cap \$$CAP, done /$DONE/, stall ${STALL_MIN} min)"
while true; do
  info=$("${SF[@]}" info "$ID" 2>/dev/null) || info=""
  st=$("${SF[@]}" status "$ID" 2>/dev/null) || st=unreachable
  case "$st" in gone|deleted|deleting) log "box is $st"; echo lost > "$OUT/VERDICT"; exit 0;; esac
  cost=$(echo "$info" | python3 -c 'import json,sys
try: d=json.load(sys.stdin); print(d.get("cost_estimate") or 0)
except Exception: print(0)' 2>/dev/null)
  if python3 -c "import sys; sys.exit(0 if float('${cost:-0}') >= float('$CAP') else 1)"; then finish cap; fi
  # first line: total bytes under the log dir + GPU utilisation (activity = either changed), then the log tail
  tailtxt=$(bash scripts/vast/ssh.sh "$ID" "echo \$(find $(dirname "$RLOG") -type f -printf '%s\\n' 2>/dev/null | awk '{s+=\$1} END {print s+0}')_\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9); tail -n 400 $RLOG 2>/dev/null" 2>/dev/null | grep -av Welcome)
  if [ -z "$tailtxt" ]; then
    miss=$((miss + 1)); log "no log yet / unreachable ($st, \$$cost, miss $miss)"
    [ "$miss" -ge 20 ] && finish stalled
  else
    miss=0
    first=$(echo "$tailtxt" | head -1); size=${first%%_*}; gpu=${first##*_}
    body=$(echo "$tailtxt" | tail -n +2)
    if echo "$body" | grep -qaE "$DONE"; then finish finished; fi
    if echo "$body" | grep -qaE "$FAIL_REGEX"; then log "failure line: $(echo "$body" | grep -aE "$FAIL_REGEX" | tail -1 | cut -c1-200)"; finish failed; fi
    if [ "$size" != "$last_size" ] || [ "${gpu:-0}" -gt 5 ] 2>/dev/null; then last_size=$size; last_change=$(date +%s); fi
    idle=$(( ($(date +%s) - last_change) / 60 ))
    [ "$idle" -ge "$STALL_MIN" ] && finish stalled
    log "$st \$$cost logs=${size}B gpu=${gpu}% idle=${idle}m | $(echo "$body" | grep -av '^\s*$' | tail -1 | cut -c1-120)"
  fi
  sleep "$POLL"
done
