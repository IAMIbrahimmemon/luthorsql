#!/bin/bash
# Mac side: wait for a box, push (launch_phase1.sh, RL=1), start a phase-1 box script detached, then watch_box.sh.
#   CAP=8 bash scripts/chains/run_phase1_box.sh <id> <name> <box_script> <remote_log> <done_regex> <remote_dir>...
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=$1; NAME=$2; SCRIPT=$3; RLOG=$4; DONE=$5; shift 5
mkdir -p runs/phase1/$NAME
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a runs/phase1/$NAME/watch.log; }
for i in $(seq 1 120); do
  REFRESH=1 bash scripts/vast/ssh.sh "$ID" true >/dev/null 2>&1 && break
  [ "$i" = 120 ] && { log "ssh never came up: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
  sleep 15
done
log "ssh up"
ok=0
for t in 1 2 3; do RL=1 bash scripts/chains/launch_phase1.sh "$ID" >> runs/phase1/$NAME/push.log 2>&1 && { ok=1; break; }; log "push attempt $t failed"; sleep 30; done
[ $ok = 1 ] || { log "push failed 3 times: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
log "pushed; starting $SCRIPT"
bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (setsid nohup bash /workspace/luthor/$SCRIPT > $RLOG 2>&1 < /dev/null &)" 2>/dev/null
exec bash scripts/chains/watch_box.sh "$ID" "$NAME" "$RLOG" "$DONE" "$@"
