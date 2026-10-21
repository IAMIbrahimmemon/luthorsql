#!/bin/bash
# Runs ON the GPU box: score the R1 ∪ R2 ∪ R3a held-out pool (data/ens3/heldout_union3.jsonl) with each of the three
# validators (W's wide r1, r2 and the seed-control run R3a's k=8 validator); sum_z over 3 vs 2 is computed on the Mac
# (DECISION only). Inputs: /root/d/ens3/{heldout_union3.jsonl,v_r1,v_r2,v_r3a,sys1/bird_train/cards}.
# Outputs: /workspace/ens3/heldout_by_<v>.jsonl, DONE_ENS3. Scoring never reads the ok flags.
set -euo pipefail
R=/workspace; D=/root/d; O=$R/ens3; mkdir -p "$O" "$R/logs"
export HF_HOME=$R/hf UV_CACHE_DIR=$R/uvcache PATH=$HOME/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
if [ ! -x $R/venv/bin/python ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv -q --python 3.12 $R/venv
  VIRTUAL_ENV=$R/venv uv pip install -q --torch-backend=auto torch "transformers>=5.0" peft sqlglot numpy safetensors \
    accelerate huggingface_hub hf_transfer
  VIRTUAL_ENV=$R/venv uv pip install -q -e $R/luthor --no-deps
fi
V=$R/venv/bin/python
$V -c "import torch, transformers, peft; print('torch', torch.__version__, torch.version.cuda, 'tf', transformers.__version__, 'peft', peft.__version__); assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo ENS3_FAIL setup; exit 1; }
$V -c "from huggingface_hub import snapshot_download as s; s('seeklhy/OmniSQL-7B')"
echo DONE_SETUP
cd $R/luthor
for v in r3a r1 r2; do
  out=$O/heldout_by_${v}.jsonl
  [ -s "$out" ] && continue
  echo "== heldout by $v $(date -u +%T)"
  $V scripts/rerank_sq2.py score --rr $D/ens3/v_$v --cards $D/ens3/sys1/bird_train --dev-cands $D/ens3/heldout_union3.jsonl \
    --out "$out.tmp" --base seeklhy/OmniSQL-7B --subfolder "" 2>&1 | grep -E "Error|error" || true
  [ -s "$out.tmp" ] && mv "$out.tmp" "$out" || { echo "ENS3_FAIL $v"; exit 1; }
done
echo DONE_ENS3
