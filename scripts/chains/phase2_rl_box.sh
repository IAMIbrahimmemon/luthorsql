#!/bin/bash
# Phase 2b RL pilot arm (docs/BUDGET500_PLAN.md; rung 1b), ON the box, detached; pushed by RL=1 launch_phase1.sh:
#   ARM=grpo|passk8 setsid nohup bash /workspace/luthor/scripts/chains/phase2_rl_box.sh > /workspace/logs/rl.log 2>&1 &
# From R2: GRPO (A) or pass@8 advantage (B), both with SFT + list anchors, G=$G, beta 0.01, seed 0, 1 epoch over
# BIRD-train (held-out training DBs and works_cycles excluded; contaminated DBs and gold-noise ids dropped), then the
# held-out eval (k=8 lists, frozen R2 validator) and k=8 lists on BIRD dev (blind). Ends with PHASE2_RL_DONE.
set -uo pipefail
R=/workspace; D=/root/d; S=$D/sys1; L=$R/logs; V=$R/venv6/bin/python
mkdir -p "$L"; cd $R/luthor
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH=$R/venv6/bin:$PATH
W=$(( $(nproc) - 2 )); [ "$W" -lt 2 ] && W=2
export S V W   # scripts/rl/heldout_eval.sh (run by --heldout-eval) reads them; its defaults are Mac paths
ARM=${ARM:?ARM=grpo|passk8}; G=${G:-32}; HOURS=${HOURS:-0}
log() { echo "$(date -u '+%m-%d %H:%M:%S') [rl $ARM] $*"; }
die() { log "PHASE2_RL_FAIL $*"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -e "$L/setup.ok" ]; then
  EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/'; die setup; }
  grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
  touch "$L/setup.ok"
fi
log DONE_SETUP
# EVAL_EXTRA="runs/rl1/grpo ...": earlier arms' weights pushed to the box; held-out eval them first (≈ 20 min each)
for X in ${EVAL_EXTRA:-}; do
  log "held-out eval of $X"
  bash scripts/rl/heldout_eval.sh "$X" > "$L/heldout_$(basename $X).log" 2>&1 || log "held-out eval of $X failed"
  [ -s "$X/rr/heldout_compare.json" ] && cat "$X/rr/heldout_compare.json" || log "HELDOUT_EVAL_MISSING $X"
  mkdir -p $R/out/extra && cp -r "$X/rr" "$R/out/extra/$(basename $X)_rr"
done
# EVAL_ONLY=1: only the EVAL_EXTRA held-out evals above (no training, no dev lists)
if [ "${EVAL_ONLY:-0}" = 1 ]; then mkdir -p $R/out; log PHASE2_RL_DONE; exit 0; fi
EXTRA=(); [ "$ARM" = passk8 ] && EXTRA=(--adv passk --passk 8)
OUT=runs/rl1/$ARM
DROP=(); [ -s $S/gold_noise_drop.json ] && DROP=(--drop-ids $S/gold_noise_drop.json)
log "train $OUT"
$V -I scripts/rl/rl_train.py --run runs/r1r2/r2 --plans $S/bird_train --db-roots $S/dbs $S/rrdbs \
  --exclude-dbs $S/dev_dbs_bird.json --skip-dbs works_cycles --drop-dbs $S/contaminated_dbs.json "${DROP[@]}" \
  --epochs 1 --G $G --q-per-step 32 --sft-weight 0.5 --list-weight 0.5 --beta 0.01 --seed 0 --workers $W \
  "${EXTRA[@]}" --out $OUT > "$L/rl_train.log" 2>&1 || { tail -n 30 "$L/rl_train.log" | sed 's/Traceback/TB/'; die train; }
grep -aE '^(STEP|EPOCH|HELDOUT|SCORE|greedy|oracle)' "$L/rl_train.log" | tail -n 8
# held-out eval as its own process: run inside rl_train.py (--heldout-eval) it OOMs, the trainer still holds ~73 GB
bash scripts/rl/heldout_eval.sh $OUT > "$L/heldout_$ARM.log" 2>&1 || log "held-out eval of $OUT failed"
[ -s $OUT/rr/heldout_compare.json ] && cat $OUT/rr/heldout_compare.json || log "HELDOUT_EVAL_MISSING (see $L/rl_train.log)"
log "BIRD dev lists (blind)"
$V scripts/rerank_sq2.py gen --run $OUT --k 8 --workers $W --plans $S/bird_dev --db-roots $D/dev_databases \
  --dev-dbs $S/bird_dev_dbs.json --split dev --all-questions --questions $D/dev.json --out $OUT/rr/cands_bird_dev.jsonl \
  > /dev/null 2>&1 || log "bird dev gen failed (non-fatal)"
mkdir -p $R/out && cp -r $OUT $R/out/
log PHASE2_RL_DONE
