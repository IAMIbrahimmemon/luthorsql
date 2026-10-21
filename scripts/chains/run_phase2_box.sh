#!/bin/bash
# Mac side, Phase 2 (docs/BUDGET500_PLAN.md): wait for a box, push, start, watch (watch_box.sh pulls + deletes).
#   CAP=16 bash scripts/chains/run_phase2_box.sh <id> seed      R3a: R1 recipe, SEED=1, contaminated DBs dropped
#   CAP=8 ARM=grpo|passk8 bash scripts/chains/run_phase2_box.sh <id> rl
#   CAP=24 bash scripts/chains/run_phase2_box.sh <id> teacher   2a: R3a recipe + DeepSeek teacher rows (b1_rare1 + b2_mut)
#   CAP=24 bash scripts/chains/run_phase2_box.sh <id> verified  2a': R3a recipe with BIRD-Verified BIRD-train plans
#   CAP=16 bash scripts/chains/run_phase2_box.sh <id> seedfull  R3b: R1 recipe, SEED=1, contaminated DBs kept (pure seed control)
#   CAP=24 bash scripts/chains/run_phase2_box.sh <id> v4desc   phase 3 (runs/phase1/v4desc/PREREG.md): R3a recipe with
#     grammar v4 plans (data/plans_v4/...), --vocab templates_union.json, BIRD-train column descriptions (DESC=1 default:
#     --bird-desc /root/d/bird_train_desc, pushed from data/raw/bird_train_desc; DESC=0 to run without) and the
#     arithmetic-with-constants rows (data/plans_v4/div_extra, scripts/build_div_extra.py) as --extra-train ...:DIVX_W
#     (default 4); WIDE_K=32 lists too. V4_PUSH_EXTRA: more local paths pushed to /root/d/plans_v4. V4_DROP overrides
#     the dropped DBs (default contaminated_dbs.json, as R3a).
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=$1; KIND=$2
if [ "$KIND" = v4desc ] && [ "${DESC:-1}" = 1 ]; then   # checked before any box time is spent
  n=$(find data/raw/bird_train_desc -name '*.csv' -print0 2>/dev/null | xargs -0 -n1 dirname 2>/dev/null | sort -u | wc -l | tr -d ' ')
  [ "$n" -ge 69 ] || { echo "v4desc: data/raw/bird_train_desc covers $n of 69 BIRD-train DBs (fetch_bird_train_desc.py not done?)"; exit 2; }
  [ -f data/plans_v4/div_extra/plans.jsonl ] || { echo "v4desc: run scripts/build_div_extra.py first"; exit 2; }
fi
NAME=$KIND; [ "$KIND" = rl ] && NAME=rl_${ARM:?ARM}
mkdir -p runs/phase1/$NAME
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a runs/phase1/$NAME/watch.log; }
for i in $(seq 1 120); do
  REFRESH=1 bash scripts/vast/ssh.sh "$ID" true >/dev/null 2>&1 && break
  [ "$i" = 120 ] && { log "ssh never came up: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
  sleep 15
done
log "ssh up"
if [ "$KIND" = seed ] || [ "$KIND" = teacher ] || [ "$KIND" = verified ] || [ "$KIND" = seedfull ] || [ "$KIND" = v4desc ]; then
  EXTRA="--max-oom-skip 0.05"; BP=""; DROP=/root/d/sys1/contaminated_dbs.json
  [ "$KIND" = seedfull ] && DROP=""   # R1 trained with formula_1 / soccer_1: keep them so only the seed differs
  if [ "$KIND" = teacher ]; then   # teacher rows: their own cards beside the shard file; held-out / dev DBs dropped by train_sq2
    REMOTE_D=/root/d/teacher bash scripts/vast/push6.sh "$ID" data/teacher/export/b1_rare1 data/teacher/export/b2_mut \
      >> runs/phase1/$NAME/push.log 2>&1 || { log "teacher push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
    EXTRA="$EXTRA --extra-train /root/d/teacher/b1_rare1/plans-00000.jsonl --extra-train /root/d/teacher/b2_mut/plans-00000.jsonl"
  fi
  if [ "$KIND" = verified ]; then  # same dir name (bird_train) so --repeat bird_train=3 still applies
    REMOTE_D=/root/d/sys1v bash scripts/vast/push6.sh "$ID" data/sys1/verified/bird_train \
      >> runs/phase1/$NAME/push.log 2>&1 || { log "verified push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
    BP=/root/d/sys1v/bird_train
  fi
  RARE_DECIDED='use 5483 rows sha256:c9ac3783cfb92e0a'; V4ENV=()
  if [ "$KIND" = v4desc ]; then   # the same rows as R3a, converted to grammar v4 (data/plans_v4/...)
    P4=/root/d/plans_v4
    # shellcheck disable=SC2086
    if [ "${DESC:-1}" = 1 ]; then REMOTE_D=/root/d bash scripts/vast/push6.sh "$ID" data/raw/bird_train_desc \
      >> runs/phase1/$NAME/push.log 2>&1 || { log "desc push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }; fi
    REMOTE_D=$P4 bash scripts/vast/push6.sh "$ID" data/plans_v4/train/bird_train data/plans_v4/train/spider_train \
      data/plans_v4/train_big/synsql_train data/plans_v4/rare1 data/plans_v4/div_extra data/plans_v4/templates_union.json \
      data/plans_v4/templates_union_nums.json ${V4_PUSH_EXTRA:-} >> runs/phase1/$NAME/push.log 2>&1 \
      || { log "v4 push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
    BP=$P4/bird_train; DROP=${V4_DROP-/root/d/sys1/contaminated_dbs.json}
    EXTRA="$EXTRA --extra-train $P4/div_extra:${DIVX_W:-4}"
    RARE_DECIDED="use 5483 rows sha256:$(shasum -a 256 data/plans_v4/rare1/plans.jsonl | cut -c1-16) (v4)"
    V4ENV=(GRAMMAR=v4 VOCAB=$P4/templates_union.json SPIDER_PLANS=$P4/spider_train SYNSQL_PLANS=$P4/synsql_train
           RARE_PLANS=$P4/rare1 DESC=${DESC:-1} "DESC_ARGS=${DESC_ARGS:---bird-desc /root/d/bird_train_desc}" WIDE_K=${WIDE_K:-32})
    [ -n "${BIRD_DEV_PLANS:-}" ] && V4ENV+=(BIRD_DEV_PLANS=$BIRD_DEV_PLANS)
  fi
  # rare data = the same 5,483 rare1 rows R1 used (sha256 c9ac3783cfb92e0a; v4desc: its v4 conversion): skip the qwrite stage
  bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && echo '$RARE_DECIDED' > /workspace/logs/rare.decided" 2>/dev/null
  ok=0
  for t in 1 2 3; do
    # --max-oom-skip 0.05: R1 itself skipped 237 OOM'd works_cycles batches and went on (the 0.2% guard is newer)
    env ${V4ENV[@]+"${V4ENV[@]}"} N=6714 SEED=1 TRAIN_EXTRA="$EXTRA" BIRD_PLANS="$BP" DROP_DBS="$DROP" QW_MODELS="" \
      bash scripts/chains/launch_r1r2.sh "$ID" R1 \
      >> runs/phase1/$NAME/push.log 2>&1 && { ok=1; break; }
    log "launch attempt $t failed"; sleep 30
  done
  [ $ok = 1 ] || { log "launch failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
  log "launched"
  FAIL_REGEX='R1_FAIL|FATAL' exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/chain.log DONE_R1R2 \
    /workspace/luthor/runs/r1r2/r1
else
  ok=0
  for t in 1 2 3; do RL=1 bash scripts/chains/launch_phase1.sh "$ID" >> runs/phase1/$NAME/push.log 2>&1 && \
    REMOTE_D=/root/d/sys1 bash scripts/vast/push6.sh "$ID" data/sys1/gold_noise_drop.json >> runs/phase1/$NAME/push.log 2>&1 && { ok=1; break; }
    log "push attempt $t failed"; sleep 30; done
  [ $ok = 1 ] || { log "push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
  EX=""
  for X in ${EVAL_PUSH:-}; do   # local trained arm dirs (runs/phase1/rl_<arm>/out/<arm>) -> box runs/rl1/<arm>, scored first
    n=$(basename "$X")
    REMOTE_D=/workspace/luthor/runs/rl1/$n bash scripts/vast/push6.sh "$ID" "$X/best_lora" "$X/best_heads.pt" "$X/args.json" \
      "$X/vocab.json" >> runs/phase1/$NAME/push.log 2>&1 && EX="$EX runs/rl1/$n" || log "push of $X failed (not evaluated)"
  done
  bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (EVAL_ONLY=${EVAL_ONLY:-0} EVAL_EXTRA='$EX' ARM=$ARM G=${G:-64} setsid nohup bash /workspace/luthor/scripts/chains/phase2_rl_box.sh > /workspace/logs/rl.log 2>&1 < /dev/null &)" 2>/dev/null
  log "started RL arm $ARM"
  exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/rl.log PHASE2_RL_DONE /workspace/out
fi
