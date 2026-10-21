#!/bin/bash
# Mac side of the single-LoRA System 1 run (runs/phase1/single_lora/PREREG.md): wait for a box, push code + the data
# single_lora_box.sh reads (push6 runs leakcheck), start it detached, watch (watch_box.sh pulls /workspace/single +
# logs into runs/phase1/single_lora/, then deletes the box).
#   CAP=25 bash scripts/chains/run_single_lora_box.sh <instance_id>
# Pushed: R1's checkpoint (runs/r1r2/r1: best_lora, best_heads.pt, args.json, vocab.json -> /root/d/r1), R1's
# training plan dirs (data/sys1/{bird_train,spider_train,synsql_train} + templates, data/rare1), the held-out DBs
# (data/sys1/dbs) and dev_dbs_bird.json, VALHARD's lists + R1's held-out k=32 pool (data/valhard/r1). Public data only;
# no BIRD dev, no sealed sets. Forwarded env: EPOCHS TOK MAX_HOURS GEN_WEIGHT LR EXTRA.
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=${1:?usage: CAP=<dollars> run_single_lora_box.sh <instance_id>}; NAME=${NAME:-single_lora}   # NAME: one results dir per arm
: "${CAP:?CAP dollars required (spend guard)}"
mkdir -p runs/phase1/$NAME
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a runs/phase1/$NAME/watch.log; }
R1=(runs/r1r2/r1/best_lora runs/r1r2/r1/best_heads.pt runs/r1r2/r1/args.json runs/r1r2/r1/vocab.json)
SYS1=(data/sys1/bird_train data/sys1/spider_train data/sys1/synsql_train data/sys1/dbs data/sys1/templates.json
      data/sys1/templates_nums.json data/sys1/dev_dbs_bird.json)
VH=(data/valhard/r1/train_lists.jsonl data/valhard/r1/val_lists.jsonl data/valhard/r1/heldout_pool.jsonl)
for f in "${R1[@]}" "${SYS1[@]}" "${VH[@]}" data/rare1/plans.jsonl data/rare1/cards; do
  [ -e "$f" ] || { log "missing $f: not starting"; exit 1; }
done
for i in $(seq 1 240); do
  REFRESH=1 bash scripts/vast/ssh.sh "$ID" true >/dev/null 2>&1 && break
  [ "$i" = 240 ] && { log "ssh never came up: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
  sleep 15
done
log "ssh up"
ok=0
for t in 1 2 3; do
  REMOTE_D=/root/d/r1 bash scripts/vast/push6.sh "$ID" "${R1[@]}" >> runs/phase1/$NAME/push.log 2>&1 \
    && REMOTE_D=/root/d/sys1 bash scripts/vast/push6.sh "$ID" "${SYS1[@]}" >> runs/phase1/$NAME/push.log 2>&1 \
    && REMOTE_D=/root/d bash scripts/vast/push6.sh "$ID" data/rare1 >> runs/phase1/$NAME/push.log 2>&1 \
    && REMOTE_D=/root/d/valhard/r1 bash scripts/vast/push6.sh "$ID" "${VH[@]}" >> runs/phase1/$NAME/push.log 2>&1 \
    && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
if [ -n "${PRESET_RUN:-}" ]; then   # a ready single-LoRA run dir (e.g. data/single_soup/run): the box skips training
  [ "$(basename "$PRESET_RUN")" = run ] || { log "PRESET_RUN must be a dir named run"; exit 1; }
  ok=0
  for t in 1 2 3; do
    REMOTE_D=/workspace/single bash scripts/vast/push6.sh "$ID" "$PRESET_RUN" >> runs/phase1/$NAME/push.log 2>&1 && { ok=1; break; }
    log "preset push attempt $t failed"; sleep 30
  done
  [ $ok = 1 ] || { log "preset push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
  log "pushed preset run $PRESET_RUN (no training)"
fi
envs=""
for v in EPOCHS TOK MAX_HOURS GEN_WEIGHT LR EXTRA; do
  if [ -n "${!v+x}" ]; then envs+="$v=$(printf %q "${!v}") "; fi
done
bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (${envs} setsid nohup bash /workspace/luthor/scripts/chains/single_lora_box.sh > /workspace/logs/single.log 2>&1 < /dev/null &)" 2>/dev/null
log "started ($envs)"
STALL_MIN=${STALL_MIN:-75} FAIL_REGEX='SINGLE_FAIL' exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/single.log SINGLE_DONE /workspace/single
