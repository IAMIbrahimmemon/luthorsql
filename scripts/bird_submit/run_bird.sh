#!/bin/bash
# LuthorSQL on BIRD: one command, questions + databases in, BIRD prediction file out.
#   bash run_bird.sh <test.json> <test_tables.json> <test_databases/> <predict_test.json> [weights_dir]
# First run: creates ./venv (Python 3.12 via uv) and installs requirement.txt (torch 2.11.0 cu128 wheels; they run on a
# CUDA 12.2 / 12.3 driver), then runs luthor/scripts/bird_submit/predict.py on one GPU. Later runs reuse ./venv.
# weights_dir (default ./weights) holds single/ (one LoRA: generator + validator) or r1/ + valhard_rr/; the base model seeklhy/OmniSQL-7B
# is downloaded from Hugging Face on first use (HF_HOME decides where). Progress is saved per database: if the run stops,
# rerun the same command to resume (see README).
set -euo pipefail
[ $# -ge 4 ] || { sed -n 2,8p "$0"; exit 2; }
H=$(cd "$(dirname "$0")" && pwd)
W=$(cd "${5:-$H/weights}" && pwd)
if [ -d "$W/single" ]; then   # single-LoRA weights: one adapter for generation and selection
  RUN=$W/single; RR=$W/single
  NEED=("$W/single/best_lora/adapter_model.safetensors" "$W/single/best_heads.pt" "$W/single/args.json"
        "$W/single/vocab.json" "$W/single/rerank_head.pt" "$W/single/settings.json")
else
  RUN=$W/r1; RR=$W/valhard_rr
  NEED=("$W/r1/best_lora/adapter_model.safetensors" "$W/r1/best_heads.pt" "$W/r1/args.json" "$W/r1/vocab.json"
        "$W/valhard_rr/rerank_head.pt" "$W/valhard_rr/rerank_lora/rerank/adapter_model.safetensors")
fi
for f in "${NEED[@]}"; do
  [ -s "$f" ] || { echo "missing $f (see README: weights)"; exit 1; }
done
if [ ! -x "$H/venv/bin/python" ]; then
  command -v uv >/dev/null || python3 -m pip install -q --user uv || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=$HOME/.local/bin:$PATH
  uv venv -q --python 3.12 "$H/venv"
  VIRTUAL_ENV=$H/venv uv pip install -q --torch-backend=cu128 -r "$H/requirement.txt"
fi
"$H/venv/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
exec "$H/venv/bin/python" "$H/luthor/scripts/bird_submit/predict.py" --questions "$1" --tables "$2" --db-root "$3" \
  --out "$4" --run "$RUN" --rr "$RR"
