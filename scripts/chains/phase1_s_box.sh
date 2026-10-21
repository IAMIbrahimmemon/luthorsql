#!/bin/bash
# Phase 1, box S (docs/BUDGET500_PLAN.md), ON the box, detached; pushed by: RL=1 START=0 launch_phase1.sh <id>
#   setsid nohup bash /workspace/luthor/scripts/chains/phase1_s_box.sh > /workspace/logs/s.log 2>&1 < /dev/null &
# 1e RL timing smoke (G=32 then G=64) -> 1c head dumps of R1, R2 (dump_box.sh) -> 1b R1+R2 LoRA soup: k=8 lists on
# held-out and BIRD dev, scored by R2's frozen validator. BIRD dev outputs are written blind (no accuracy printed).
# Ends with PHASE1_S_DONE; any fatal step prints PHASE1_S_FAIL <step>.
set -uo pipefail
R=/workspace; D=/root/d; S=$D/sys1; L=$R/logs; O=$R/p1s; V=$R/venv6/bin/python
mkdir -p "$L" "$O"; cd $R/luthor
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH=$R/venv6/bin:$PATH
W=$(( $(nproc) - 2 )); [ "$W" -lt 2 ] && W=2
log() { echo "$(date -u '+%m-%d %H:%M:%S') [p1s] $*"; }
die() { log "PHASE1_S_FAIL $*"; exit 1; }
x() { local n=$1; shift; log "run $n"; "$@" > "$L/$n.log" 2>&1; local rc=$?; [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 20 "$L/$n.log" | sed 's/Traceback/TB/'; }; return $rc; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -e "$L/setup.ok" ]; then
  EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/'; die setup; }
  grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
  touch "$L/setup.ok"
fi
log DONE_SETUP

# ---- 1e: RL timing ----------------------------------------------------------------------------------------------------
for G in 32 64; do
  [ -s runs/rl_timing_g$G/timing.json ] && continue
  x rl_timing_g$G $V -I scripts/rl/rl_train.py --run runs/r1r2/r2 --plans $S/bird_train --db-roots $S/dbs $S/rrdbs \
    --exclude-dbs $S/dev_dbs_bird.json --skip-dbs works_cycles --G $G --q-per-step 32 --sft-weight 0.5 \
    --list-weight 0.5 --beta 0.01 --workers $W --timing-smoke 20 --out runs/rl_timing_g$G \
    || { [ $G = 32 ] && die "rl timing G=32"; log "G=64 failed (kept G=32)"; }
  grep -aE '^(TIMING|PROJECTION)' "$L/rl_timing_g$G.log" | tail -n 4
done
mkdir -p $O/rl_timing; cp -r runs/rl_timing_g* $O/rl_timing/ 2>/dev/null
log DONE_RL_TIMING

# ---- 1c: head dumps ---------------------------------------------------------------------------------------------------
ROLES="r1 r2" SKIP_SETUP=1 bash scripts/chains/dump_box.sh > "$L/dump_master.log" 2>&1 || { tail -n 30 "$L/dump_master.log" | sed 's/Traceback/TB/'; die dump; }
grep -aE 'CHECK_CANDS|VERIFY' "$L/dump_master.log" | cut -c1-200
log DONE_DUMP

# ---- 1b: LoRA soup ----------------------------------------------------------------------------------------------------
[ -s runs/r1r2/soup/best_heads.pt ] || x soup_build $V scripts/analysis/soup_lora.py runs/r1r2/r1 runs/r1r2/r2 runs/r1r2/soup || die soup_build
mkdir -p $O/soup
G8=($V scripts/rerank_sq2.py gen --run runs/r1r2/soup --k 8 --workers $W)
[ -s $O/soup/cands_dev.jsonl ] || x soup_gen_heldout "${G8[@]}" --plans $S/bird_train --db-roots $S/dbs \
  --dev-dbs $S/dev_dbs_bird.json --split dev --out $O/soup/cands_dev.jsonl || die soup_gen_heldout
[ -s $O/soup/cands_bird_dev.jsonl ] || x soup_gen_bird_dev "${G8[@]}" --plans $S/bird_dev --db-roots $D/dev_databases \
  --dev-dbs $S/bird_dev_dbs.json --split dev --all-questions --questions $D/dev.json --out $O/soup/cands_bird_dev.jsonl || die soup_gen_bird_dev
[ -s $O/soup/heldout_scored.jsonl ] || x soup_score_heldout $V scripts/rerank_sq2.py score --rr runs/r1r2/r2/rr \
  --cards $S/bird_train --dev-cands $O/soup/cands_dev.jsonl --out $O/soup/heldout_scored.jsonl --base seeklhy/OmniSQL-7B --subfolder "" || die soup_score_heldout
grep -aE '^SCORE' "$L/soup_score_heldout.log" | tail -2
[ -s $O/soup/bird_dev_scored.jsonl ] || x soup_score_bird_dev $V scripts/rerank_sq2.py score --rr runs/r1r2/r2/rr \
  --cards $S/bird_dev --dev-cands $O/soup/cands_bird_dev.jsonl --out $O/soup/bird_dev_scored.jsonl --base seeklhy/OmniSQL-7B --subfolder "" || die soup_score_bird_dev
rm -f "$L/soup_score_bird_dev.log" "$L/soup_gen_bird_dev.log"   # dev stays blind: no accuracy lines leave the box
log DONE_SOUP
log PHASE1_S_DONE
