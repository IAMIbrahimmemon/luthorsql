#!/bin/bash
#SBATCH --job-name=luthor-rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/hpc/%x-%j.out
# GRPO with execution reward on a trained System-1 run: scripts/rl/rl_train.py (invocation only; see that
# file for the method). BIRD-train minus the held-out training DBs; dev and sealed sets are never read. One GPU.
# rl_train.py has no resume: size the run (--steps / --epochs) to one job's wall time (HPC_TIME).
# Env: SFT (required: the SFT run dir, e.g. runs/hpc/sft_r1), RUN (required: output runs/hpc/$RUN),
#   RL_ARGS (replaces the default "--epochs 2 --G 16 --q-per-step 32"), RL_EXTRA (more rl_train.py flags).
source scripts/hpc/common.sh
SFT=${SFT:?SFT=<sft run dir>}; RUN=${RUN:?RUN=<name>}; O=$REPO/runs/hpc/$RUN
need scripts/rl/rl_train.py "$SFT/best_lora" "$S/bird_train" "$S/dbs" "$S/rrdbs" "$S/dev_dbs_bird.json"
mkdir -p "$O"
# shellcheck disable=SC2086
cmd=(python -I scripts/rl/rl_train.py --run "$SFT" --plans "$S/bird_train" --db-roots "$S/dbs" "$S/rrdbs"
     --exclude-dbs "$S/dev_dbs_bird.json" ${RL_ARGS:---epochs 2 --G 16 --q-per-step 32} --workers "$W"
     --out "$O" --device cuda ${RL_EXTRA:-})
log "run: ${cmd[*]}"
"${cmd[@]}"
log "DONE_RL: $O"
