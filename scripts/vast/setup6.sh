#!/bin/bash
# Runs ON the Vast instance (round 6): venv6 with torch cu130 + vLLM + TRL/peft/bitsandbytes (4-bit QLoRA of a
# 32B model), HF cache on the big disk. Fails loudly if the driver's CUDA is < 13.0 or the GPU is unusable.
#   nohup bash /workspace/luthor/scripts/vast/setup6.sh > /workspace/logs/setup6.log 2>&1 &     -> DONE_SETUP6
# Env: REMOTE_ROOT (default /workspace), VLLM_SPEC (default vllm, e.g. vllm==X.Y.Z to pin),
#      MODELS="org/name ..." to prefetch checkpoints into $HF_HOME, EXPECT_GPUS (default 1).
set -euo pipefail
R=${REMOTE_ROOT:-/workspace}
trap 'echo SETUP6_FAIL' ERR
export HF_HOME=$R/hf UV_CACHE_DIR=$R/uvcache PATH=$HOME/.local/bin:$PATH
mkdir -p "$R/logs" "$HF_HOME"
# Driver CUDA must be >= 13.0 (round 5's first pod had 12.8 and torch cu130 could not see the GPU).
DRV=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# Driver CUDA 12.8-12.9 (e.g. Shadeform images with no cuda13 option): torch from the cu128 index instead (TORCH_BACKEND)
if awk -v v="${DRV:-0}" 'BEGIN { split(v, a, "."); exit !(a[1] + 0 >= 13) }'; then TB=${TORCH_BACKEND:-cu130}
elif awk -v v="${DRV:-0}" 'BEGIN { split(v, a, "."); exit !(a[1] + 0 == 12 && a[2] + 0 >= 8) }'; then TB=${TORCH_BACKEND:-cu128}
else echo "driver CUDA ${DRV:-unknown} < 12.8: destroy this instance"; echo SETUP6_FAIL; exit 1; fi
# current vLLM wheels need libcudart 13: on a cu128 box vLLM is left out (VLLM_SPEC=none); the RL arms and the
# R1/R2 chain with QW_MODELS="" never import it (only qwrite.py, gen6.py, sel_eval.py, sqlw_*.py do)
[ "$TB" = cu130 ] && VS=${VLLM_SPEC:-vllm} || VS=${VLLM_SPEC:-none}
[ "$VS" = none ] && VS=torch
echo "torch backend $TB (driver CUDA $DRV), first package: $VS"; export TB
[ "$(nvidia-smi -L | wc -l)" -eq "${EXPECT_GPUS:-1}" ] || echo "WARNING: not exactly ${EXPECT_GPUS:-1} GPU(s)"
df -h "$R"
set -x
command -v uv >/dev/null || pip install -q --break-system-packages uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -q --python 3.12 "$R/venv6"
export VIRTUAL_ENV=$R/venv6
# --torch-backend makes uv take torch (and vLLM's torch pin) from the $TB wheel index.
uv pip install -q --torch-backend="$TB" "$VS" trl peft bitsandbytes accelerate datasets sqlglot \
  ninja huggingface_hub hf_transfer
uv pip install -q -e "$R/luthor" --no-deps
"$R/venv6/bin/python" - <<'EOF'
import importlib.util, torch, trl, transformers, peft, bitsandbytes as bnb
vllm = __import__("vllm") if importlib.util.find_spec("vllm") else None
print("V6", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm and vllm.__version__, "trl", trl.__version__,
      "tf", transformers.__version__, "peft", peft.__version__, "bnb", bnb.__version__)
assert torch.cuda.is_available(), "CUDA driver too old for this torch build"
import os
want = os.environ["TB"][2:4]
assert (torch.version.cuda or "0").split(".")[0] == want, f"torch built for CUDA {torch.version.cuda}, want {os.environ['TB']}"
torch.zeros(1).cuda()
l = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()   # 4-bit kernel smoke test
print("bnb4", l(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)).shape, torch.cuda.get_device_name(0))
EOF
for m in ${MODELS:-}; do
  HF_HUB_ENABLE_HF_TRANSFER=1 "$R/venv6/bin/python" -c "from huggingface_hub import snapshot_download as s; s('$m')"
done
echo DONE_SETUP6
