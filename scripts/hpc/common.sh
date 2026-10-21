# Sourced by the scripts/hpc job scripts (on a compute node, from the repo root): site settings, modules, venv,
# caches, data paths. Defines D (data root), S (D/sys1), W (worker count), need (fail on missing inputs).
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"
[ -r scripts/hpc/site.env ] || { echo "no scripts/hpc/site.env (copy site.env.example)"; exit 2; }
source scripts/hpc/site.env
for m in ${HPC_MODULES:-}; do module load "$m"; done
source "$HPC_VENV/bin/activate"
# compute nodes may have no internet: setup.sh prefetches the checkpoints on the login node
export HF_HOME PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$LUTHOR_DATA; S=$D/sys1
NP=${SLURM_CPUS_PER_TASK:-8}; W=$(( NP > 4 ? NP - 2 : 2 ))
mkdir -p logs/hpc runs/hpc
log() { echo "$(date -u '+%m-%d %H:%M:%S') [${SLURM_JOB_NAME:-job} ${SLURM_JOB_ID:-local}] $*"; }
need() { for p; do [ -e "$p" ] || { log "missing input: $p"; exit 1; }; done; }
log "host $(hostname), repo $REPO, data $D"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || { log "no GPU visible"; exit 1; }
