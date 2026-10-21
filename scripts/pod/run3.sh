# Runs ON the pod (round-3 pilot):  STAGE=birddb|smoke|train|rerank bash /workspace/luthor/scripts/pod/run3.sh
# Decisions use held-out TRAINING databases only (/root/d/mix2_dev_dbs.json). No sealed set is scored here.
set -x
cd /workspace/luthor && export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=/workspace/tenv/bin/python
BT=/workspace/bt/train/train_databases
DEVP="--dev-plans /root/d/plans/spider_train /root/d/plans/bird_train"
COMMON="--db-roots /root/d/db $BT --dev-dbs /root/d/mix2_dev_dbs.json"
R=runs/${RUN:-sq3_pilot}
case "${STAGE:?}" in
  birddb)  # public BIRD train archive -> SQLite files (execution checks on BIRD training databases)
    mkdir -p /workspace/bt && cd /workspace/bt
    [ -f train.zip ] || curl -sSL -o train.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
    unzip -oq train.zip && cd train && unzip -oq train_databases.zip && rm -f train_databases.zip
    ls $BT | wc -l && echo BIRDDB_OK ;;
  smoke)
    $T scripts/train_sq2.py --plans /root/d/mix3 $DEVP $COMMON --out runs/smoke3 --limit 3000 --max-steps 30 \
        --eval-every 30 --schema-aug 0.35 || { echo SMOKE_FAIL; exit 1; }
    $T scripts/rerank_sq2.py gen --run runs/smoke3 --plans /root/d/mix3 $COMMON --split train --max-q 300 \
        --out runs/smoke3/cands_train.jsonl || { echo SMOKE_FAIL; exit 1; }
    $T scripts/rerank_sq2.py gen --run runs/smoke3 --plans /root/d/plans/spider_train /root/d/plans/bird_train $COMMON \
        --split dev --max-q 100 --out runs/smoke3/cands_dev.jsonl || { echo SMOKE_FAIL; exit 1; }
    $T scripts/rerank_sq2.py train --cards /root/d/mix3 /root/d/plans/spider_train /root/d/plans/bird_train \
        --train-cands runs/smoke3/cands_train.jsonl --dev-cands runs/smoke3/cands_dev.jsonl --out runs/smoke3/rr \
        --max-steps 10 && echo SMOKE_OK || echo SMOKE_FAIL ;;
  train)
    $T scripts/train_sq2.py --plans /root/d/mix3 $DEVP $COMMON --out $R --max-steps ${STEPS:?} --eval-every 500 \
        --schema-aug 0.35 --hours ${HOURS:-4.5} && echo DONE_TRAIN || echo TRAIN_FAIL ;;
  rerank)
    $T scripts/rerank_sq2.py gen --run $R --plans /root/d/mix3 $COMMON --split train --max-q ${RRQ:-12000} \
        --out $R/cands_train.jsonl || { echo RERANK_FAIL; exit 1; }
    $T scripts/rerank_sq2.py gen --run $R --plans /root/d/plans/spider_train /root/d/plans/bird_train $COMMON \
        --split dev --out $R/cands_dev.jsonl || { echo RERANK_FAIL; exit 1; }
    $T scripts/rerank_sq2.py train --cards /root/d/mix3 /root/d/plans/spider_train /root/d/plans/bird_train \
        --train-cands $R/cands_train.jsonl --dev-cands $R/cands_dev.jsonl --out $R/rr --epochs 1 --hours ${RRH:-1.2} \
        && echo DONE_RERANK || echo RERANK_FAIL ;;
esac
