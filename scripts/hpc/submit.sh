#!/bin/bash
# Submit a Luthor job with the site's partition / account / GPU type (scripts/hpc/site.env). Run from anywhere on the
# cluster's login node; the job runs from the repo root and inherits this shell's environment (RUN, GPUS, ...).
#   RUN=sft_r1 GPUS=1 bash scripts/hpc/submit.sh sft.sbatch
#   RUN=sft_r1_x4 GPUS=4 TRAIN_EXTRA="--seed 1" bash scripts/hpc/submit.sh sft.sbatch
#   RUN=sft_r1 bash scripts/hpc/submit.sh rerank.sbatch --dependency=afterok:<sft job id>
# Extra arguments go to sbatch. Prints the job id.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -r scripts/hpc/site.env ] || { echo "copy scripts/hpc/site.env.example to scripts/hpc/site.env and fill it in"; exit 2; }
grep -qE '^[A-Z_]+=CHANGE_ME' scripts/hpc/site.env && { echo "scripts/hpc/site.env still has CHANGE_ME values"; exit 2; }
source scripts/hpc/site.env
JOB=${1:?usage: submit.sh <sft|rerank|rl>.sbatch [sbatch args...]}; shift
[ -f "scripts/hpc/$JOB" ] || { echo "no scripts/hpc/$JOB"; exit 2; }
GPUS=${GPUS:-1}
export GPUS HPC_JOB=$JOB
mkdir -p logs/hpc
args=(--parsable -p "$HPC_PARTITION" --gres="gpu:${HPC_GPU_TYPE:+$HPC_GPU_TYPE:}$GPUS" --time="$HPC_TIME"
      --cpus-per-task=$(( GPUS * HPC_CPUS_PER_GPU )) --mem=$(( GPUS * HPC_MEM_PER_GPU_GB ))G)
[ -n "${HPC_ACCOUNT:-}" ] && args+=(-A "$HPC_ACCOUNT")
sbatch "${args[@]}" "$@" "scripts/hpc/$JOB"
