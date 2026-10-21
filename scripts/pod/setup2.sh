# Runs ON the pod: second venv for the SQL writer (vLLM + TRL + peft) and the public Qwen2.5-Coder-7B-Instruct
# checkpoint. Separate from /workspace/tenv because vLLM pins its own torch. Prints READY2.
set -ex
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uvcache
which uv || pip install -q --break-system-packages uv
uv venv -q --python 3.12 /workspace/venv2
VIRTUAL_ENV=/workspace/venv2 uv pip install -q vllm trl peft accelerate datasets sqlglot
VIRTUAL_ENV=/workspace/venv2 uv pip install -q -e /workspace/luthor --no-deps
/workspace/venv2/bin/python -c "import torch, vllm, trl, transformers, peft; print('V2', torch.__version__, vllm.__version__, trl.__version__, transformers.__version__, peft.__version__); assert torch.cuda.is_available(), 'CUDA driver too old for this torch build'; torch.zeros(1).cuda()"
/workspace/venv2/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-Coder-7B-Instruct')"
echo READY2
