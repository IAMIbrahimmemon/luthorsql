# Runs ON the pod (round 4):  STAGE=diag|... bash /workspace/luthor/scripts/pod/run4.sh
# Decisions use held-out TRAINING databases only (/root/d/mix2_dev_dbs.json). No sealed set is scored here.
set -x
cd /workspace/luthor && export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/workspace/venv2/bin:$PATH
T=/workspace/tenv/bin/python
V=/workspace/venv2/bin/python
QB=Qwen/Qwen2.5-Coder-7B-Instruct
W=runs/sqlw_sft
RL=runs/sqlw_rl
BT=/workspace/bt/train/train_databases
DEVPL="/root/d/plans/spider_train /root/d/plans/bird_train"
COMMON="--db-roots /root/d/db $BT --dev-dbs /root/d/mix2_dev_dbs.json"
CARDS="--cards /root/d/mix3 $DEVPL"
P=runs/sq3_pilot
WIDE="--k 32 --wide-singles 12 --wide-pairs 5 --runner-up2 6 --samples 24 --temp ${TEMP:-1.0}"
case "${STAGE:?}" in
  diag)  # wider candidate pool from the round-3 pilot model + reranker without memory skips
    [ -s $P/cands_dev32.jsonl ] || $T scripts/rerank_sq2.py gen --run $P --plans $DEVPL $COMMON --split dev $WIDE --out $P/cands_dev32.jsonl \
      || { echo DIAG_FAIL; exit 1; }
    [ -s $P/cands_train16.jsonl ] || $T scripts/rerank_sq2.py gen --run $P --plans /root/d/mix3 $COMMON --split train --max-q ${RRQ:-12000} \
      --k 16 --wide-singles 12 --wide-pairs 5 --runner-up2 6 --samples 8 --out $P/cands_train16.jsonl \
      || { echo DIAG_FAIL; exit 1; }
    $T scripts/rerank_sq2.py train $CARDS --train-cands $P/cands_train16.jsonl --dev-cands $P/cands_dev32.jsonl \
      --out $P/rr4 --epochs 1 --hours ${RRH:-1.0} && echo DONE_DIAG || echo DIAG_FAIL ;;
  sft)  # SQL writer: supervised LoRA on Qwen2.5-Coder-7B-Instruct
    $V scripts/sqlw_sft.py --data /root/d/sqlw/sft.jsonl --out $W --hours ${SFTH:-4} \
      && echo DONE_SFT || echo SFT_FAIL ;;
  wgen)  # writer candidates on held-out dev, merged with the slot pool; then the reranker over the union
    $V scripts/sqlw_gen.py --lora $W/lora --slot-cands $P/cands_dev32.jsonl --cards $DEVPL --db-roots /root/d/db $BT \
      --out $W/union_dev.jsonl || { echo WGEN_FAIL; exit 1; }
    $T scripts/rerank_sq2.py score $CARDS --rr $P/rr4 --dev-cands $W/union_dev.jsonl --out $W/union_dev_scored.jsonl \
      && echo DONE_WGEN || echo WGEN_FAIL ;;
  rl)  # merge the SFT LoRA, then execution-reward GRPO on Spider/BIRD train questions
    [ -f /workspace/m/sqlw_sft/config.json ] || $V scripts/sqlw_merge.py --base $QB --lora $W/lora --out /workspace/m/sqlw_sft \
      || { echo RL_FAIL; exit 1; }
    [ -s $W/rl_hard.jsonl ] || $V scripts/sqlw_rlfilter.py --model /workspace/m/sqlw_sft --data /root/d/sqlw/rl.jsonl \
      --db-roots /root/d/db $BT --out $W/rl_hard.jsonl || { echo RL_FAIL; exit 1; }
    $V scripts/sqlw_grpo.py --model /workspace/m/sqlw_sft --data $W/rl_hard.jsonl --db-roots /root/d/db $BT \
      --out $RL --hours ${RLH:-3.5} ${RLARGS} && echo DONE_RL || echo RL_FAIL ;;
  rlgen)  # the RL writer on the same dev questions, merged with the slot pool, then the reranker
    $V scripts/sqlw_gen.py --model /workspace/m/sqlw_sft --lora $RL/lora --slot-cands $P/cands_dev32.jsonl --cards $DEVPL \
      --db-roots /root/d/db $BT --out $RL/union_dev.jsonl || { echo RLGEN_FAIL; exit 1; }
    $T scripts/rerank_sq2.py score $CARDS --rr $P/rr4 --dev-cands $RL/union_dev.jsonl --out $RL/union_dev_scored.jsonl \
      && echo DONE_RLGEN || echo RLGEN_FAIL ;;
esac
