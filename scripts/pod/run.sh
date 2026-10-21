# Runs ON the pod, one stage at a time:  STAGE=smoke|speed|2b|4b|merge bash /workspace/luthor/scripts/pod/run.sh
set -x
cd /workspace/luthor && export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=/workspace/tenv/bin/python
# the mix on the pod's local disk (the /workspace network volume is slow for 28k small files)
[ -f /root/mix/plans.jsonl ] || cp -r /workspace/luthor/data/mix /root/mix
COMMON="--plans /root/mix --db-roots /root/mix/dev_sqlite --dev-dbs /root/mix/dev_dbs.json --unsup-keep 1.0"  # the mix already subsampled unsupported rows
case "${STAGE:?}" in
  smoke) $T scripts/train_sq.py $COMMON --out runs/smoke --max-steps 50 --eval-every 50 && \
         $T scripts/merge_sq.py --run runs/smoke && echo SMOKE_OK || echo SMOKE_FAIL ;;
  speed) $T scripts/train_sq.py $COMMON --out runs/speed --speed-test 20 ;;
  2b)    $T scripts/train_sq.py $COMMON --subfolder qwen3.5-2b-nli-v5 --limit-train 30000 --out runs/sq2b_check --eval-every 500 && echo DONE_2B ;;
  4b)    $T scripts/train_sq.py $COMMON --out runs/sq4b_v1 --eval-every 1000 && echo DONE_4B ;;
  merge) $T scripts/merge_sq.py --run runs/sq4b_v1 && echo MERGED ;;
esac
