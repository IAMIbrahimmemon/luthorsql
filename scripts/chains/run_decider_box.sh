#!/bin/bash
# Mac side: wait for a box, push code + data/decider (push6 + leakcheck), start decider_box.sh, watch (watch_box.sh
# pulls /workspace/dout and deletes the box).   CAP=19.5 bash scripts/chains/run_decider_box.sh <id>
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=$1; NAME=decider
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
  REMOTE_D=/workspace bash scripts/vast/push6.sh "$ID" data/decider >> runs/phase1/$NAME/push.log 2>&1 && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (FT_ARGS='${FT_ARGS:-}' setsid nohup bash /workspace/luthor/scripts/chains/decider_box.sh > /workspace/logs/dec.log 2>&1 < /dev/null &)" 2>/dev/null
log "started"
STALL_MIN=${STALL_MIN:-60} FAIL_REGEX='DECIDER_FAIL' exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/dec.log DECIDER_FT_DONE /workspace/dout
