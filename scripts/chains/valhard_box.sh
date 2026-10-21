#!/bin/bash
# VALHARD (docs/paper_notes/valhard_2026-10-10.md, runs/phase1/valhard/PREREG.md), ON the box:
#   VH_TAG=r1 setsid nohup bash /workspace/luthor/scripts/chains/valhard_box.sh > /workspace/logs/valhard.log 2>&1 < /dev/null &
# One System-1 validator (rerank LoRA r=16 + head on frozen OmniSQL-7B) trained with `rerank_sq2.py train` v2 on the
# hard-negative lists /root/d/valhard/$VH_TAG/{train_lists,val_lists}.jsonl (val2 recipe: 2 epochs, warmup + cosine,
# label smoothing 0.05, token-budget batching with the gradient cache, best epoch by val pick accuracy; loss LOSS,
# default group = multipos over result groups). Then it scores ONE generator's own k=32 held-out pool
# (/root/d/valhard/$VH_TAG/heldout_pool.jsonl) into /workspace/valhard/heldout_by_new.jsonl; with
# /root/d/valhard/$VH_TAG/base_rr present (v4desc: the generator's own validator) the same pool is also scored by it
# into heldout_by_base.jsonl. Ends VALHARD_DONE (VALHARD_FAIL on errors). No BIRD dev, no sealed sets; the pool holds
# DECISION + RESERVED rows and no score is printed here (scored on the Mac, DECISION only).
# Env (set by run_valhard_box.sh from the tag):
#   VH_TAG   r1 (default) | v4desc
#   CARDS    schema-card plan dir (default /root/d/sys1/bird_train; v4desc /root/d/plans_v4/bird_train)
#   DESC_ARGS validator card flags, the SAME as the generator's (v4desc: --bird-desc /root/d/bird_train_desc)
#   GRAMMAR  v3 (default) | v4, exported as LUTHOR_GRAMMAR
#   LOSS (group) EPOCHS (2) TOK (120000) MAX_HOURS (6.5)
set -uo pipefail
R=/workspace; D=/root/d; VH_TAG=${VH_TAG:-r1}; VH=$D/valhard/$VH_TAG; L=$R/logs; O=$R/valhard; V=$R/venv6/bin/python
CARDS=${CARDS:-$D/sys1/bird_train}; DESC_ARGS=${DESC_ARGS:-}; GRAMMAR=${GRAMMAR:-v3}
LOSS=${LOSS:-group}; EPOCHS=${EPOCHS:-2}; TOK=${TOK:-120000}; MAX_HOURS=${MAX_HOURS:-6.5}
mkdir -p "$L" "$O"; cd $R/luthor
export LUTHOR_GRAMMAR=$GRAMMAR HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH=$R/venv6/bin:$PATH
log() { echo "$(date -u '+%m-%d %H:%M:%S') [valhard] $*"; }
die() { log "VALHARD_FAIL $*"; exit 1; }
x() { local n=$1; shift; log "run $n"; "$@" > "$L/$n.log" 2>&1; local rc=$?; [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 20 "$L/$n.log" | grep -av GATE | sed 's/Traceback/TB/'; }; return $rc; }
log "tag $VH_TAG cards $CARDS grammar $GRAMMAR desc '${DESC_ARGS}' loss $LOSS epochs $EPOCHS tok $TOK"
g=$(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits) || die "nvidia-smi failed"
log "GPUs: $g"
if echo "$g" | awk -F', *' '$3 + 0 != 0 { bad = 1 } END { exit !bad }'; then die "GPU memory already in use at boot"; fi
for f in $VH/train_lists.jsonl $VH/val_lists.jsonl $VH/heldout_pool.jsonl; do [ -s $f ] || die "missing $f"; done
[ -d $CARDS/cards ] || die "missing $CARDS/cards"
for f in $DESC_ARGS; do case "$f" in /*) [ -e "$f" ] || die "missing $f";; esac; done
if [ ! -e "$L/setup.ok" ]; then
  EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/'; die setup; }
  grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
  touch "$L/setup.ok"
fi
log DONE_SETUP
# preflight: the trainer must know the flags this chain passes (fails in seconds, not after setup + hours)
$V scripts/rerank_sq2.py train --help 2>&1 | grep -q -- "group" || die "rerank_sq2.py has no --loss group (code not pushed?)"
for f in $DESC_ARGS; do case "$f" in --*) $V scripts/rerank_sq2.py train --help 2>&1 | grep -q -- "$f" || die "rerank_sq2.py has no $f";; esac; done
wc -l $VH/train_lists.jsonl $VH/val_lists.jsonl $VH/heldout_pool.jsonl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# shellcheck disable=SC2086
[ -s $O/rr/rerank_head.pt ] || x valhard_train $V scripts/rerank_sq2.py train --cards $CARDS \
  --train-file $VH/train_lists.jsonl --val-file $VH/val_lists.jsonl --out $O/rr --base seeklhy/OmniSQL-7B --subfolder "" \
  --lora-r 16 --epochs $EPOCHS --sched cosine --loss $LOSS --label-smooth 0.05 --tok-budget $TOK --max-hours $MAX_HOURS \
  $DESC_ARGS || die valhard_train
grep -aE '^(train lists|[0-9]+ steps per epoch|VAL |SKIPPED_FOR_MEMORY|trained )' "$L/valhard_train.log" | tail -12
cp -f "$L/valhard_train.log" $O/ 2>/dev/null
# shellcheck disable=SC2086
[ -s $O/heldout_by_new.jsonl ] || x valhard_score_new $V scripts/rerank_sq2.py score --rr $O/rr --cards $CARDS \
  --dev-cands $VH/heldout_pool.jsonl --out $O/heldout_by_new.jsonl --base seeklhy/OmniSQL-7B --subfolder "" --lora-r 16 \
  $DESC_ARGS || die valhard_score_new
if [ -s $VH/base_rr/rerank_head.pt ]; then
  # shellcheck disable=SC2086
  [ -s $O/heldout_by_base.jsonl ] || x valhard_score_base $V scripts/rerank_sq2.py score --rr $VH/base_rr --cards $CARDS \
    --dev-cands $VH/heldout_pool.jsonl --out $O/heldout_by_base.jsonl --base seeklhy/OmniSQL-7B --subfolder "" --lora-r 16 \
    $DESC_ARGS || die valhard_score_base
fi
# no SCORE print: the held-out pool includes RESERVED, which stays blind (scored on the Mac, DECISION only)
log VALHARD_DONE
