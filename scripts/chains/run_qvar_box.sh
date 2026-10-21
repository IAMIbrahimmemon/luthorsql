#!/bin/bash
# Mac side: wait for a box, push (launch_phase1.sh RL=1 ROLES=r2 + data/sys1/qvar), start qvar_box.sh, watch.
# watch_box.sh pulls /workspace/qvar and deletes the box.
#   CAP=5.5 bash scripts/chains/run_qvar_box.sh <id>
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=$1; NAME=qvar
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
  RL=1 ROLES=r2 bash scripts/chains/launch_phase1.sh "$ID" >> runs/phase1/$NAME/push.log 2>&1 && \
  REMOTE_D=/root/d/sys1 bash scripts/vast/push6.sh "$ID" data/sys1/qvar >> runs/phase1/$NAME/push.log 2>&1 && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (setsid nohup bash /workspace/luthor/scripts/chains/qvar_box.sh > /workspace/logs/qv.log 2>&1 < /dev/null &)" 2>/dev/null
log "started"
FAIL_REGEX='QVAR_FAIL' exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/qv.log QVAR_DONE /workspace/qvar
