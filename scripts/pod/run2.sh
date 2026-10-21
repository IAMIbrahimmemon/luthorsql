# Runs ON the pod (round 2):  STAGE=smoke|pilot bash /workspace/luthor/scripts/pod/run2.sh
set -x
cd /workspace/luthor && export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=/workspace/tenv/bin/python
P="--plans /root/d/plans/spider_train /root/d/plans/spider_syn_train --db-roots /root/d/db --dev-dbs /root/d/dev_dbs.json"
case "${STAGE:?}" in
  smoke) $T scripts/train_sq2.py $P --out runs/smoke2 --limit 600 --max-steps 30 --eval-every 30 && echo SMOKE_OK || echo SMOKE_FAIL ;;
  pilot) $T scripts/train_sq2.py $P --out runs/sq2_pilot --epochs ${EPOCHS:-3} --eval-every 300 --gate /root/d/gate/spider_dev \
           && echo DONE_PILOT || echo PILOT_FAIL ;;
  full)  $T scripts/train_sq2.py --plans /root/d/mix2 --dev-plans /root/d/plans/spider_train /root/d/plans/bird_train \
           --db-roots /root/d/db /root/bt/data/raw/bird_train/train_databases --dev-dbs /root/d/mix2_dev_dbs.json \
           --resume runs/sq2_pilot --out runs/sq2_r2 --epochs 1 --eval-every 1000 --hours ${HOURS:-4.6} \
           --gate /root/d/gate/spider_dev && echo DONE_FULL || echo FULL_FAIL ;;
esac
