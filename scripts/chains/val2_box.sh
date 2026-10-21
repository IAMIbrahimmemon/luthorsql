#!/bin/bash
# Validator v2 (selection memo 2026-10-10, step 1 "Recipe A+"), ON the box:
#   setsid nohup bash /workspace/luthor/scripts/chains/val2_box.sh > /workspace/logs/val2.log 2>&1 < /dev/null &
# One validator (rerank LoRA r=16 on frozen OmniSQL-7B) trained with `rerank_sq2.py train` v2 on the prebuilt lists
# /root/d/val2/train_lists.jsonl (val pick accuracy per epoch on val_lists.jsonl, best epoch kept): 2 epochs, warmup +
# cosine, multipos loss with label smoothing 0.05, token-budget batching with the gradient cache (no question skipped
# for memory). Then it scores the R1 ∪ R2 wide held-out pool (/root/d/val2/heldout_union.jsonl, pushed by the Mac)
# into /workspace/val2/heldout_by_new.jsonl. Ends with VAL2_DONE (VAL2_FAIL on errors). No BIRD dev, no sealed sets.
set -uo pipefail
R=/workspace; D=/root/d; S=$D/sys1; V2=$D/val2; L=$R/logs; O=$R/val2; V=$R/venv6/bin/python
mkdir -p "$L" "$O"; cd $R/luthor
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH=$R/venv6/bin:$PATH
TOK=${TOK:-120000}   # memory units per pass (rerank_sq2.chunk_cost); ~0.4 MB each on OmniSQL-7B -> ~48 GB of an 80 GB card
log() { echo "$(date -u '+%m-%d %H:%M:%S') [val2] $*"; }
die() { log "VAL2_FAIL $*"; exit 1; }
x() { local n=$1; shift; log "run $n"; "$@" > "$L/$n.log" 2>&1; local rc=$?; [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 20 "$L/$n.log" | sed 's/Traceback/TB/'; }; return $rc; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $V2/train_lists.jsonl $V2/val_lists.jsonl $V2/heldout_union.jsonl; do [ -s $f ] || die "missing $f"; done
[ -d $S/bird_train/cards ] || die "missing $S/bird_train/cards"
if [ ! -e "$L/setup.ok" ]; then
  EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/'; die setup; }
  grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
  touch "$L/setup.ok"
fi
log DONE_SETUP
wc -l $V2/train_lists.jsonl $V2/val_lists.jsonl $V2/heldout_union.jsonl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -s $O/rr/rerank_head.pt ] || x val2_train $V scripts/rerank_sq2.py train --cards $S/bird_train \
  --train-file $V2/train_lists.jsonl --val-file $V2/val_lists.jsonl --out $O/rr --base seeklhy/OmniSQL-7B --subfolder "" \
  --lora-r 16 --epochs 2 --sched cosine --loss multipos --label-smooth 0.05 --tok-budget $TOK --max-hours 6.5 \
  || die val2_train
grep -aE '^(train lists|[0-9]+ steps per epoch|VAL |SKIPPED_FOR_MEMORY|trained )' "$L/val2_train.log" | tail -12
cp -f "$L/val2_train.log" $O/ 2>/dev/null
[ -s $O/heldout_by_new.jsonl ] || x val2_score_heldout $V scripts/rerank_sq2.py score --rr $O/rr --cards $S/bird_train \
  --dev-cands $V2/heldout_union.jsonl --out $O/heldout_by_new.jsonl --base seeklhy/OmniSQL-7B --subfolder "" --lora-r 16 \
  || die val2_score_heldout
# no SCORE print: the held-out file includes RESERVED, which stays blind (scored on the Mac, DECISION only)
log VAL2_DONE
