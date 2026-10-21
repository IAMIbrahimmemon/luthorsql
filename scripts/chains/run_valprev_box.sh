#!/bin/bash
# Mac side: wait for a box, push (launch_phase1.sh ROLES=r2 + box W's R2 wide lists and held-out union), start
# valprev_box.sh, watch (watch_box.sh pulls /workspace/valprev and deletes the box).
#   CAP=7.5 bash scripts/chains/run_valprev_box.sh <id>
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=$1; NAME=valprev; P=runs/phase1/p1w/p1w
mkdir -p runs/phase1/$NAME
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a runs/phase1/$NAME/watch.log; }
for i in $(seq 1 240); do
  REFRESH=1 bash scripts/vast/ssh.sh "$ID" true >/dev/null 2>&1 && break
  [ "$i" = 240 ] && { log "ssh never came up: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
  sleep 15
done
log "ssh up"
ok=0
for t in 1 2 3; do
  ROLES=r2 bash scripts/chains/launch_phase1.sh "$ID" >> runs/phase1/$NAME/push.log 2>&1 && \
  REMOTE_D=/workspace/p1w/r2 bash scripts/vast/push6.sh "$ID" $P/r2/cands_train.jsonl $P/r2/cands_dev.jsonl >> runs/phase1/$NAME/push.log 2>&1 && \
  REMOTE_D=/workspace/p1w bash scripts/vast/push6.sh "$ID" $P/heldout_union.jsonl >> runs/phase1/$NAME/push.log 2>&1 && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (setsid nohup bash /workspace/luthor/scripts/chains/valprev_box.sh > /workspace/logs/vp.log 2>&1 < /dev/null &)" 2>/dev/null
log "started"
FAIL_REGEX='VALPREV_FAIL' exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/vp.log VALPREV_DONE /workspace/valprev
