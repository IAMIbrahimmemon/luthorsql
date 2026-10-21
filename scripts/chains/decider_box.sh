#!/bin/bash
# Decider v1.1 27B LoRA fine-tune as the candidate selector (scripts/decider/ft.py), ON the box:
#   setsid nohup bash /workspace/luthor/scripts/chains/decider_box.sh > /workspace/logs/dec.log 2>&1 < /dev/null &
# Own venv (Perplexity's pins: torch 2.14, transformers 5.17, flash-linear-attention) so venv6 is untouched; the
# checkpoint (Apache-2.0, ~54 GB) comes from Hugging Face. Data: /workspace/decider (pushed data/decider). Ends DECIDER_FT_DONE.
set -uo pipefail
R=/workspace; L=$R/logs; VD=$R/venv_dec; CK=$R/dckpt; DD=$R/decider; OUT=$R/dout
mkdir -p "$L" "$OUT"; cd $R/luthor
export HF_HOME=$R/hf UV_CACHE_DIR=$R/uvcache PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH=$HOME/.local/bin:$PATH \
  HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date -u '+%m-%d %H:%M:%S') [dec] $*"; }
die() { log "DECIDER_FAIL $*"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if [ ! -e "$L/dec_setup.ok" ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv -q --python 3.12 $VD || die venv
  VIRTUAL_ENV=$VD uv pip install -q --torch-backend=cu130 torch==2.14.0 torchvision==0.29.0 transformers==5.17.0 \
    accelerate==1.15.0 safetensors==0.8.0 pillow peft huggingface_hub hf_transfer flash-linear-attention==0.5.2 numpy \
    > $L/dec_pip.log 2>&1 || { tail -20 $L/dec_pip.log; die pip; }
  $VD/bin/python -c "import torch, transformers, peft, fla; print('DEC_ENV', torch.__version__, torch.version.cuda, transformers.__version__, peft.__version__); assert torch.cuda.is_available()" || die "env check"
  touch "$L/dec_setup.ok"
fi
log DONE_SETUP
if [ ! -e "$CK/readout.safetensors" ]; then
  $VD/bin/python -c "from huggingface_hub import snapshot_download as s; p=s('perplexity-ai/pplx-decider-v1.1-27b', local_dir='$CK'); print('DOWNLOADED', p)" \
    > $L/dec_download.log 2>&1 || { tail -5 $L/dec_download.log; die download; }
fi
du -sh $CK | cut -f1 | sed 's/^/checkpoint size /'
log "fine-tune"
$VD/bin/python scripts/decider/ft.py --ckpt $CK --data $DD --out $OUT ${FT_ARGS:-} > $L/dec_ft.log 2>&1 \
  || { tail -30 $L/dec_ft.log | sed 's/Traceback/TB/'; die ft; }
grep -a "temperature\|TRAINED" $L/dec_ft.log | tail -3
cp $L/dec_ft.log $OUT/
log DECIDER_FT_DONE
