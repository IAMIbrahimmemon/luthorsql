# Runs ON the pod (round 5, session A):  STAGE=... bash /workspace/luthor/scripts/pod/run5.sh
# Dev decisions use the frozen held-out dev list (/root/d/eval/dev_frozen.jsonl). No sealed set is scored here.
set -x
cd /workspace/luthor && export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/workspace/venv2/bin:$PATH
V=/workspace/venv2/bin/python
QB=Qwen/Qwen2.5-Coder-7B-Instruct
M=/workspace/m/sqlw_sft                       # merged SFT writer (base for the RL LoRA)
DEVPL="/root/d/plans/spider_train /root/d/plans/bird_train"
DBS="/root/d/db /workspace/bt/train/train_databases"
CARDS="/root/d/plans/spider_train/cards /root/d/plans/bird_train/cards"
O=runs/r5
mkdir -p $O
case "${STAGE:?}" in
  merge)
    [ -f $M/config.json ] || $V scripts/sqlw_merge.py --base $QB --lora runs/sqlw_sft/lora --out $M || { echo MERGE_FAIL; exit 1; }
    echo DONE_MERGE ;;
  devgen)  # both round-4 writers on the full frozen dev list (complete denominator)
    $V scripts/sqlw_gen.py --lora runs/sqlw_sft/lora --questions /root/d/eval/dev_frozen.jsonl --cards $DEVPL \
      --db-roots $DBS --out $O/dev_sft.jsonl || { echo DEVGEN_FAIL; exit 1; }
    $V scripts/sqlw_gen.py --model $M --lora runs/sqlw_rl/lora --questions /root/d/eval/dev_frozen.jsonl --cards $DEVPL \
      --db-roots $DBS --out $O/dev_rl.jsonl && echo DONE_DEVGEN || echo DEVGEN_FAIL ;;
  traingen)  # 8 samples + greedy per training question from the SFT writer (8,000 questions: execution is the bottleneck)
    $V scripts/sqlw_gen.py --lora runs/sqlw_sft/lora --questions /root/d/sel/train_questions.jsonl --cards $DEVPL \
      --db-roots $DBS --n 8 --temp 1.0 --seconds 10 --limit ${TRQ:-8000} --out $O/train_sft.jsonl \
      && echo DONE_TRAINGEN || echo TRAINGEN_FAIL ;;
  seldata)
    $V scripts/sel_data.py --cands $O/train_sft.jsonl --cards $CARDS --db-roots $DBS \
      --exclude-dbs /root/d/mix2_dev_dbs.json --writer-only --workers 16 --out $O/sel_items.jsonl \
      && echo DONE_SELDATA || echo SELDATA_FAIL ;;
  seltrain)  # selector + reviser as one LoRA on the base coder model
    $V scripts/sqlw_sft.py --data $O/sel_items.jsonl --out $O/sel --hours ${SELH:-1.5} \
      && echo DONE_SELTRAIN || echo SELTRAIN_FAIL ;;
  seleval)
    $V scripts/sel_eval.py --cands $O/dev_sft.jsonl $O/dev_rl.jsonl --lora $O/sel/lora --cards $CARDS --db-roots $DBS \
      --frozen /root/d/eval/dev_frozen.jsonl --out $O/sel_dev.json && echo DONE_SELEVAL || echo SELEVAL_FAIL ;;
  sft5)  # writer v2: clean BIRD, verified direct-SQL curriculum, value hints (session B)
    $V scripts/sqlw_sft.py --data /root/d/sqlw5/sft.jsonl --out runs/sqlw5 --hours ${SFTH:-4} \
      && echo DONE_SFT5 || echo SFT5_FAIL ;;
  devgen5)
    $V scripts/sqlw_gen.py --lora runs/sqlw5/lora --questions /root/d/eval/dev_frozen.jsonl --cards $DEVPL \
      --db-roots $DBS --hints --out $O/dev_v2.jsonl && echo DONE_DEVGEN5 || echo DEVGEN5_FAIL ;;
  seleval5)  # vote and vote + repair for v2 alone and v2 pooled with the round-4 writers (selector lost to the vote in A)
    $V scripts/sel_eval.py --cands $O/dev_v2.jsonl --writer-lora runs/sqlw5/lora --cards $CARDS \
      --db-roots $DBS --frozen /root/d/eval/dev_frozen.jsonl --out $O/sel_dev_v2.json || { echo SELEVAL5_FAIL; exit 1; }
    $V scripts/sel_eval.py --cands $O/dev_v2.jsonl $O/dev_sft.jsonl $O/dev_rl.jsonl \
      --writer-lora runs/sqlw5/lora --cards $CARDS --db-roots $DBS --frozen /root/d/eval/dev_frozen.jsonl \
      --out $O/sel_dev_pool.json && echo DONE_SELEVAL5 || echo SELEVAL5_FAIL ;;
  sealed)  # ONE-TIME final scoring of the chosen system (env: WRITERS="v2 sft rl", SEL=1, REP=1)
    for st in spider_dev spider_test spider_syn_dev bird_dev; do
      case $st in spider_dev|spider_syn_dev) C=/root/d/sealed/spider_dev; R=/root/d/db;; spider_test) C=/root/d/sealed/spider_test; R=/root/d/db_test;;
        bird_dev) C=/root/d/sealed/bird_dev; R=/root/d/bird_dev;; esac
      F=""
      for w in ${WRITERS:-v2}; do
        case $w in v2) ARGS="--lora runs/sqlw5/lora --hints";; sft) ARGS="--lora runs/sqlw_sft/lora";; rl) ARGS="--model $M --lora runs/sqlw_rl/lora";; esac
        $V scripts/sqlw_gen.py $ARGS --questions /root/d/eval/sealed_$st.jsonl --cards $C --db-roots $R \
          --out $O/sealed_${st}_$w.jsonl || { echo SEALED_FAIL; exit 1; }
        F="$F $O/sealed_${st}_$w.jsonl"
      done
      $V scripts/sel_eval.py --cands $F ${SEL:+--lora $O/sel/lora} ${REP:+--writer-lora runs/sqlw5/lora} --cards $C/cards \
        --db-roots $R --frozen /root/d/eval/sealed_$st.jsonl --out $O/sealed_$st.json || { echo SEALED_FAIL; exit 1; }
    done
    echo DONE_SEALED ;;
esac
