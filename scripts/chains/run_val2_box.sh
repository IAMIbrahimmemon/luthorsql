#!/bin/bash
# Mac side: wait for a box, push code + data/val2 and the BIRD-train cards (push6 + leakcheck), start val2_box.sh,
# watch (watch_box.sh pulls /workspace/val2 and deletes the box).   CAP=<dollars> bash scripts/chains/run_val2_box.sh <id>
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=$1; NAME=val2
mkdir -p runs/phase1/$NAME
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a runs/phase1/$NAME/watch.log; }
for f in data/val2/train_lists.jsonl data/val2/val_lists.jsonl data/val2/heldout_union.jsonl data/sys1/bird_train/cards; do
  [ -e "$f" ] || { log "missing $f (the box would fail): not starting"; exit 1; }
done
for i in $(seq 1 240); do
  REFRESH=1 bash scripts/vast/ssh.sh "$ID" true >/dev/null 2>&1 && break
  [ "$i" = 240 ] && { log "ssh never came up: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
  sleep 15
done
log "ssh up"
ok=0
for t in 1 2 3; do  # data/val2 -> /root/d/val2, data/sys1/bird_train -> /root/d/sys1/bird_train
  REMOTE_D=/root/d bash scripts/vast/push6.sh "$ID" data/val2 >> runs/phase1/$NAME/push.log 2>&1 \
    && REMOTE_D=/root/d/sys1 bash scripts/vast/push6.sh "$ID" data/sys1/bird_train >> runs/phase1/$NAME/push.log 2>&1 \
    && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (TOK='${TOK:-}' setsid nohup bash /workspace/luthor/scripts/chains/val2_box.sh > /workspace/logs/val2.log 2>&1 < /dev/null &)" 2>/dev/null
log "started"
STALL_MIN=${STALL_MIN:-60} FAIL_REGEX='VAL2_FAIL' exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/val2.log VAL2_DONE /workspace/val2
