#!/bin/bash
# Runs ON the GPU box (research problem 08-03, attack 1): score the R1 ∪ R2 candidate pools with each run's validator.
# Inputs (pushed by the Mac): /root/d/ens1/{heldout,dev}_union.jsonl, /root/d/ens1/v_{r1,r2}/ (validator LoRA + head),
# /root/d/sys1/{bird_train,bird_dev} (schema cards). Outputs: /workspace/ens1/<set>_by_<validator>.jsonl, DONE_ENS1.
# No training and no gold use here: the candidate files already carry the ok flags, which scoring never reads.
set -euo pipefail
R=/workspace; D=/root/d; O=$R/ens1; mkdir -p "$O" "$R/logs"
export HF_HOME=$R/hf UV_CACHE_DIR=$R/uvcache PATH=$HOME/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if [ ! -x $R/venv/bin/python ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv -q --python 3.12 $R/venv
  VIRTUAL_ENV=$R/venv uv pip install -q --torch-backend=auto torch "transformers>=5.0" peft sqlglot numpy safetensors \
    accelerate huggingface_hub hf_transfer
  VIRTUAL_ENV=$R/venv uv pip install -q -e $R/luthor --no-deps
fi
V=$R/venv/bin/python
$V -c "import torch, transformers, peft; print('torch', torch.__version__, torch.version.cuda, 'tf', transformers.__version__, 'peft', peft.__version__); assert torch.cuda.is_available(); torch.zeros(1).cuda()"
$V -c "from huggingface_hub import snapshot_download as s; s('seeklhy/OmniSQL-7B')"
echo DONE_SETUP
cd $R/luthor
for set in heldout dev; do
  cards="$D/sys1/bird_train"; [ $set = dev ] && cards="$D/sys1/bird_dev"
  for v in r2 r1; do
    out=$O/${set}_by_${v}.jsonl
    [ -s "$out" ] && continue
    echo "== $set by $v $(date -u +%T)"
    $V scripts/rerank_sq2.py score --rr $D/ens1/v_$v --cards $cards --dev-cands $D/ens1/${set}_union.jsonl \
      --out "$out.tmp" --base seeklhy/OmniSQL-7B --subfolder "" 2>&1 | if [ $set = dev ]; then grep -E "Error|error" || true; else grep -E "^SCORE|Error|error" || true; fi  # dev stays blind until the rule is frozen
    [ -s "$out.tmp" ] && mv "$out.tmp" "$out" || { echo "ENS1_FAIL $set $v"; exit 1; }
  done
done
echo DONE_ENS1
