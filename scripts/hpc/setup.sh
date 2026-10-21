#!/bin/bash
# One-time setup on the cluster's LOGIN node (has internet; compute nodes may not): the venv and the checkpoints.
#   bash scripts/hpc/setup.sh            (from the repo checkout on the cluster, after filling scripts/hpc/site.env)
# Env: MODELS (checkpoints to prefetch into $HF_HOME, default seeklhy/OmniSQL-7B).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/hpc/site.env
for m in ${HPC_MODULES:-}; do module load "$m"; done
export HF_HOME PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x "$HPC_VENV/bin/python" ] || uv venv -q --python 3.12 "$HPC_VENV"
export VIRTUAL_ENV=$HPC_VENV
# torch for the GPU nodes' driver (the login node has no GPU, so uv cannot detect it: HPC_TORCH_BACKEND says it)
uv pip install -q --torch-backend="${HPC_TORCH_BACKEND:-cu126}" torch transformers peft accelerate safetensors sqlglot \
  numpy huggingface_hub hf_transfer pytest
uv pip install -q -e . --no-deps
mkdir -p "$HF_HOME"
for m in ${MODELS:-seeklhy/OmniSQL-7B}; do
  HF_HUB_ENABLE_HF_TRANSFER=1 "$HPC_VENV/bin/python" -c "from huggingface_hub import snapshot_download as s; s('$m')"
done
"$HPC_VENV/bin/python" -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'tf', transformers.__version__, 'peft', peft.__version__)"
# CPU-only checks of the training code (no GPU here): seed / extra-train / data parallel (gloo) / resume
"$HPC_VENV/bin/python" -m pytest -q tests/test_train_seed.py tests/test_ddp.py tests/test_train_oom.py || \
  echo "WARNING: CPU tests failed (tests/test_ddp.py needs a local OmniSQL tokenizer: the prefetch above)"
echo DONE_HPC_SETUP
