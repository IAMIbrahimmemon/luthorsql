#!/bin/bash
# Runs ON the box: the BIRD-test environment check (BIRD runs submissions on one A100 80G with CUDA 12.2 / 12.3).
# venv6 with ONLY what System 1 inference needs (torch, transformers, peft, accelerate, sqlglot, numpy), pinned to the
# versions of the H100 latency run. Torch builds are tried in order until one sees the GPU:
#   cu128 torch 2.11.0 (the H100 run's build; runs on an older 12.x driver only through CUDA minor-version compatibility)
#   cu124 torch 2.6.0  (transformers 5.x needs torch >= 2.6; cu124 is the oldest index with 2.6)
# Writes $R/logs/bird_env.txt (driver, chosen build, every attempt) and $R/logs/requirement.txt (pip freeze of venv6).
#   EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup_bird.sh > /workspace/logs/setup.log 2>&1  -> DONE_SETUP6
# Env: REMOTE_ROOT (default /workspace), BUILDS (default "cu128:2.11.0 cu124:2.6.0"), MODELS, EXPECT_GPUS (default 1).
set -uo pipefail
R=${REMOTE_ROOT:-/workspace}
export HF_HOME=$R/hf UV_CACHE_DIR=$R/uvcache PATH=$HOME/.local/bin:$PATH
mkdir -p "$R/logs" "$HF_HOME"
E=$R/logs/bird_env.txt; : > "$E"
DRV=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1)
echo "driver $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) CUDA $DRV" | tee -a "$E"
[ "$(nvidia-smi -L | wc -l)" -eq "${EXPECT_GPUS:-1}" ] || echo "WARNING: not exactly ${EXPECT_GPUS:-1} GPU(s)"
command -v uv >/dev/null || pip install -q --break-system-packages uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
OK=
for b in ${BUILDS:-cu128:2.11.0 cu124:2.6.0}; do
  TB=${b%%:*}; TV=${b#*:}
  rm -rf "$R/venv6"; uv venv -q --python 3.12 "$R/venv6"; export VIRTUAL_ENV=$R/venv6
  if ! uv pip install -q --torch-backend="$TB" "torch==$TV" transformers==5.19.0 peft==0.21.2 accelerate sqlglot numpy \
       huggingface_hub hf_transfer; then
    echo "attempt $TB torch $TV: install failed" | tee -a "$E"; continue
  fi
  uv pip install -q -e "$R/luthor" --no-deps
  if r=$("$R/venv6/bin/python" - 2>&1 <<'EOF'
import torch, transformers, peft
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().sum().item()
print("torch", torch.__version__, "cuda", torch.version.cuda, "tf", transformers.__version__, "peft", peft.__version__,
      "gpu", torch.cuda.get_device_name(0), "bf16 matmul ok")
EOF
  ); then
    echo "attempt $TB torch $TV: OK  $r" | tee -a "$E"; OK=$TB; break
  else
    echo "attempt $TB torch $TV: GPU check failed: $(echo "$r" | tail -1)" | tee -a "$E"
  fi
done
[ -n "$OK" ] || { echo "no torch build works on driver CUDA $DRV"; echo SETUP6_FAIL; exit 1; }
echo "chosen $OK" | tee -a "$E"
VIRTUAL_ENV=$R/venv6 uv pip freeze > "$R/logs/requirement.txt"
for m in ${MODELS:-}; do
  HF_HUB_ENABLE_HF_TRANSFER=1 "$R/venv6/bin/python" -c "from huggingface_hub import snapshot_download as s; s('$m')" \
    || { echo SETUP6_FAIL; exit 1; }
done
echo DONE_SETUP6
