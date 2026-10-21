#!/bin/bash
# Single-LoRA System 1 (docs/paper_notes/single_lora_2026-10-16.md, runs/phase1/single_lora/PREREG.md), ON the box:
#   setsid nohup bash /workspace/luthor/scripts/chains/single_lora_box.sh > /workspace/logs/single.log 2>&1 < /dev/null &
# 1 train: scripts/single_lora.py from R1 (/root/d/r1) on VALHARD's lists (/root/d/valhard/r1) + R1's generator replay
#   (R1's args.json plan dirs under /root/d/sys1 and /root/d/rare1) -> /workspace/single/run
# 2 gen: the single LoRA's own k=32 --distinct candidates on the held-out DBs (phase1_w_box.sh's flags)
# 3 score: those candidates with the same adapter (heldout_by_single.jsonl, PRIMARY), and R1's own k=32 pool
#   (r1pool_by_single.jsonl: selection alone on identical candidates, secondary)
# Ends SINGLE_DONE (SINGLE_FAIL on errors). No BIRD dev, no sealed sets. The held-out files hold DECISION + RESERVED
# rows: their GEN / SCORE summary lines are removed from the logs (scored on the Mac, DECISION only).
# Env: EPOCHS (2) TOK (120000) MAX_HOURS (5.0) GEN_WEIGHT (1.0) LR (1e-4) EXTRA (more single_lora.py flags)
set -uo pipefail
R=/workspace; D=/root/d; S=$D/sys1; VH=$D/valhard/r1; L=$R/logs; O=$R/single; V=$R/venv6/bin/python
EPOCHS=${EPOCHS:-2}; TOK=${TOK:-120000}; MAX_HOURS=${MAX_HOURS:-5.0}; GEN_WEIGHT=${GEN_WEIGHT:-1.0}; LR=${LR:-1e-4}
mkdir -p "$L" "$O"; cd $R/luthor
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH=$R/venv6/bin:$PATH
NP=$(nproc 2>/dev/null || echo 8); W=$(( NP > 4 ? NP - 2 : 2 ))
log() { echo "$(date -u '+%m-%d %H:%M:%S') [single] $*"; }
die() { log "SINGLE_FAIL $*"; exit 1; }
x() { local n=$1; shift; log "run $n"; "$@" > "$L/$n.log" 2>&1; local rc=$?; [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 20 "$L/$n.log" | grep -avE '^(GEN|SCORE)' | sed 's/Traceback/TB/'; }; return $rc; }
blind() { sed -i -E '/^(GEN|SCORE|greedy|oracle|first)/d' "$L/$1.log" 2>/dev/null || true; }  # RESERVED stays blind
log "epochs $EPOCHS tok $TOK max_hours $MAX_HOURS gen_weight $GEN_WEIGHT lr $LR extra '${EXTRA:-}'"
g=$(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits) || die "nvidia-smi failed"
log "GPUs: $g"
if echo "$g" | awk -F', *' '$3 + 0 != 0 { bad = 1 } END { exit !bad }'; then die "GPU memory already in use at boot"; fi
for f in $VH/train_lists.jsonl $VH/val_lists.jsonl $VH/heldout_pool.jsonl $D/r1/args.json $D/r1/best_heads.pt \
         $D/r1/best_lora/adapter_model.safetensors $S/templates.json $S/dev_dbs_bird.json $D/rare1/plans.jsonl; do
  [ -s $f ] || die "missing $f"
done
for d in $S/bird_train $S/spider_train $S/synsql_train $S/dbs $D/rare1/cards; do [ -d $d ] || die "missing $d"; done
if [ ! -e "$L/setup.ok" ]; then
  EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/'; die setup; }
  grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
  touch "$L/setup.ok"
fi
log DONE_SETUP
$V scripts/single_lora.py --help >/dev/null 2>&1 || die "single_lora.py does not run (code not pushed?)"
$V -c "import sys; sys.path.insert(0, 'scripts'); import rerank_sq2; assert hasattr(rerank_sq2, 'shared_adapter')" \
  || die "rerank_sq2.py has no shared_adapter (code not pushed?)"
wc -l $VH/train_lists.jsonl $VH/val_lists.jsonl $VH/heldout_pool.jsonl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 1 train
# shellcheck disable=SC2086
[ -s $O/run/rerank_head.pt ] || x single_train $V scripts/single_lora.py --init $D/r1 \
  --train-file $VH/train_lists.jsonl --val-file $VH/val_lists.jsonl --cards $S/bird_train --db-roots $S/dbs \
  --out $O/run --epochs $EPOCHS --tok-budget $TOK --hours $MAX_HOURS --gen-weight $GEN_WEIGHT --lr $LR ${EXTRA:-} \
  || die single_train
grep -aE '^(generator replay|validator lists|one adapter|[0-9]+ validator steps|EVAL|trained|SKIPPED_FOR_MEMORY|SINGLE_LORA_SAVED)' \
  "$L/single_train.log" | tail -12
cp -f "$L/single_train.log" $O/ 2>/dev/null
log DONE_TRAIN

# ---- 2 gen: own k=32 candidates on the held-out DBs (as phase1_w_box.sh did for R1)
[ -s $O/cands_dev_k32.jsonl ] || x single_gen $V scripts/rerank_sq2.py gen --run $O/run --k 32 --distinct --workers $W \
  --plans $S/bird_train --db-roots $S/dbs --dev-dbs $S/dev_dbs_bird.json --split dev --out $O/cands_dev_k32.jsonl \
  || { blind single_gen; die single_gen; }
blind single_gen
log DONE_GEN

# ---- 3 score with the same adapter
[ -s $O/heldout_by_single.jsonl ] || x single_score $V scripts/rerank_sq2.py score --rr $O/run --cards $S/bird_train \
  --dev-cands $O/cands_dev_k32.jsonl --out $O/heldout_by_single.jsonl --base seeklhy/OmniSQL-7B --subfolder "" \
  || { blind single_score; die single_score; }
blind single_score
[ -s $O/r1pool_by_single.jsonl ] || x single_score_r1pool $V scripts/rerank_sq2.py score --rr $O/run --cards $S/bird_train \
  --dev-cands $VH/heldout_pool.jsonl --out $O/r1pool_by_single.jsonl --base seeklhy/OmniSQL-7B --subfolder "" \
  || { blind single_score_r1pool; die single_score_r1pool; }
blind single_score_r1pool
log SINGLE_DONE
