# Runs ON the pod: quick end-to-end checks of the round-4 stages on a few dev questions. Prints SMOKE4_OK.
set -x
cd /workspace/luthor && export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/workspace/venv2/bin:$PATH
T=/workspace/tenv/bin/python; V=/workspace/venv2/bin/python
BT=/workspace/bt/train/train_databases
DEVPL="/root/d/plans/spider_train /root/d/plans/bird_train"
COMMON="--db-roots /root/d/db $BT --dev-dbs /root/d/mix2_dev_dbs.json"
P=runs/sq3_pilot; S=runs/smoke4; mkdir -p $S
[ -s $S/cands_dev32.jsonl ] || $T scripts/rerank_sq2.py gen --run $P --plans $DEVPL $COMMON --split dev --max-q 40 --k 32 --wide-singles 12 \
  --wide-pairs 5 --runner-up2 6 --samples 24 --out $S/cands_dev32.jsonl || { echo SMOKE4_FAIL gen; exit 1; }
$V scripts/sqlw_gen.py --lora runs/smoke_sft/lora --slot-cands $S/cands_dev32.jsonl --cards $DEVPL --db-roots /root/d/db $BT \
  --out $S/union.jsonl --n 4 || { echo SMOKE4_FAIL wgen; exit 1; }
$T scripts/rerank_sq2.py score --cards /root/d/mix3 $DEVPL --rr $P/rr --dev-cands $S/union.jsonl --out $S/union_scored.jsonl \
  || { echo SMOKE4_FAIL score; exit 1; }
echo SMOKE4_OK
