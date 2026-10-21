# Runs ON the pod: trainer venv + the public OpenJev 4B checkpoint. Prints READY.
# torch comes from the cu128 index (the pod's toolkit is CUDA 12.8; the default wheel is cu130
# and causal-conv1d won't compile against it).
set -ex
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uvcache
which uv || pip install -q --break-system-packages uv
uv venv -q --python 3.12 /workspace/tenv
VIRTUAL_ENV=/workspace/tenv uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV=/workspace/tenv uv pip install -q "transformers==5.17.0" peft accelerate safetensors numpy sqlglot huggingface_hub \
  flash-linear-attention ninja packaging setuptools wheel
VIRTUAL_ENV=/workspace/tenv uv pip install -q causal-conv1d --no-build-isolation || echo "causal-conv1d failed: reference conv path"
VIRTUAL_ENV=/workspace/tenv uv pip install -q -e /workspace/luthor --no-deps
/workspace/tenv/bin/python -c "import torch, fla; print('TRAIN', torch.__version__, torch.cuda.is_available(), 'fla', fla.__version__)"
/workspace/tenv/bin/python -c "from huggingface_hub import snapshot_download as s; s('AlexWortega/openjev', allow_patterns=['qwen3.5-4b-nli-v5/*'])"
echo READY
