#!/bin/bash
# Mac side of VALHARD: wait for a box, push code + data/valhard/<tag> + the generator's schema cards (push6 runs
# leakcheck), start valhard_box.sh, watch (watch_box.sh pulls /workspace/valhard + logs, then deletes the box).
#   bash scripts/valhard/prep.sh r1                                   # first, on the Mac (CPU, ~10 min)
#   CAP=16 bash scripts/chains/run_valhard_box.sh <id> [r1|v4desc]   # results -> runs/phase1/valhard[_v4desc]/
# r1:     R1 generator lists (runs/phase1/p1w), cards data/sys1/bird_train, grammar v3, no descriptions.
# v4desc: phase-3 generator (runs/phase1/v4desc): cards data/plans_v4/train/bird_train, grammar v4,
#         --bird-desc /root/d/bird_train_desc (the generator's own cards/flags), and its own validator (base_rr) scored
#         on the same k=32 pool as the baseline.
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=$1; TAG=${2:-r1}
case "$TAG" in r1) NAME=valhard;; v4desc) NAME=valhard_v4desc;; *) echo "tag must be r1 or v4desc"; exit 2;; esac
mkdir -p runs/phase1/$NAME
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a runs/phase1/$NAME/watch.log; }
VD=data/valhard/$TAG
NEED=("$VD/train_lists.jsonl" "$VD/val_lists.jsonl" "$VD/heldout_pool.jsonl")
if [ "$TAG" = r1 ]; then
  CARDS_LOCAL=data/sys1/bird_train; CARDS_REMOTE_D=/root/d/sys1; CARDS=/root/d/sys1/bird_train; GRAMMAR=v3; DESC_ARGS=""
else
  CARDS_LOCAL=data/plans_v4/train/bird_train; CARDS_REMOTE_D=/root/d/plans_v4; CARDS=/root/d/plans_v4/bird_train; GRAMMAR=v4
  DESC_ARGS=${DESC_ARGS:---bird-desc /root/d/bird_train_desc}
  NEED+=("$VD/base_rr/rerank_head.pt" data/raw/bird_train_desc)
fi
for f in "${NEED[@]}" "$CARDS_LOCAL/cards"; do
  [ -e "$f" ] || { log "missing $f (run scripts/valhard/prep.sh $TAG first): not starting"; exit 1; }
done
for i in $(seq 1 240); do
  REFRESH=1 bash scripts/vast/ssh.sh "$ID" true >/dev/null 2>&1 && break
  [ "$i" = 240 ] && { log "ssh never came up: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
  sleep 15
done
log "ssh up"
PUSH=("$VD/train_lists.jsonl" "$VD/val_lists.jsonl" "$VD/heldout_pool.jsonl")
[ -d "$VD/base_rr" ] && PUSH+=("$VD/base_rr")
ok=0
for t in 1 2 3; do
  REMOTE_D=/root/d/valhard/$TAG bash scripts/vast/push6.sh "$ID" "${PUSH[@]}" >> runs/phase1/$NAME/push.log 2>&1 \
    && REMOTE_D=$CARDS_REMOTE_D bash scripts/vast/push6.sh "$ID" "$CARDS_LOCAL" >> runs/phase1/$NAME/push.log 2>&1 \
    && { [ "$TAG" = r1 ] || REMOTE_D=/root/d bash scripts/vast/push6.sh "$ID" data/raw/bird_train_desc >> runs/phase1/$NAME/push.log 2>&1; } \
    && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (VH_TAG=$TAG CARDS=$CARDS GRAMMAR=$GRAMMAR DESC_ARGS='$DESC_ARGS' LOSS='${LOSS:-group}' TOK='${TOK:-120000}' setsid nohup bash /workspace/luthor/scripts/chains/valhard_box.sh > /workspace/logs/valhard.log 2>&1 < /dev/null &)" 2>/dev/null
log "started ($TAG)"
STALL_MIN=${STALL_MIN:-75} FAIL_REGEX='VALHARD_FAIL' exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/valhard.log VALHARD_DONE /workspace/valhard
