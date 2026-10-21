#!/bin/bash
# Phase 1, box W (docs/BUDGET500_PLAN.md, test 1a = research problem 01-01 attack 2 on the R1 ∪ R2 pool), ON the box:
#   setsid nohup bash /workspace/luthor/scripts/chains/phase1_w_box.sh > /workspace/logs/w.log 2>&1 < /dev/null &
# Per role (r1, r2): k=32 distinct (lazy k-best) lists on BIRD-train (rrdbs, 9,000 q max), held-out and BIRD dev; a
# validator retrained on those wide lists (--train-k 32). Then the R1 ∪ R2 wide union on held-out and BIRD dev, scored
# by each wide validator (sum_z is computed on the Mac). BIRD dev is written blind. Ends with PHASE1_W_DONE.
set -uo pipefail
R=/workspace; D=/root/d; S=$D/sys1; L=$R/logs; O=$R/p1w; V=$R/venv6/bin/python
mkdir -p "$L" "$O"; cd $R/luthor
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH=$R/venv6/bin:$PATH
W=$(( $(nproc) - 2 )); [ "$W" -lt 2 ] && W=2
K=${K:-32}
log() { echo "$(date -u '+%m-%d %H:%M:%S') [p1w] $*"; }
die() { log "PHASE1_W_FAIL $*"; exit 1; }
x() { local n=$1; shift; log "run $n"; "$@" > "$L/$n.log" 2>&1; local rc=$?; [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 20 "$L/$n.log" | sed 's/Traceback/TB/'; }; return $rc; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -e "$L/setup.ok" ]; then
  EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/'; die setup; }
  grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
  touch "$L/setup.ok"
fi
log DONE_SETUP
for role in r1 r2; do
  o=$O/$role; mkdir -p $o
  G=($V scripts/rerank_sq2.py gen --run runs/r1r2/$role --k $K --distinct --workers $W)
  [ -s $o/cands_dev.jsonl ] || x gen_heldout_$role "${G[@]}" --plans $S/bird_train --db-roots $S/dbs \
    --dev-dbs $S/dev_dbs_bird.json --split dev --out $o/cands_dev.jsonl || die gen_heldout_$role
  grep -aE '^(greedy|oracle|GEN|first)' "$L/gen_heldout_$role.log" | tail -3
  [ -s $o/cands_train.jsonl ] || x gen_train_$role "${G[@]}" --plans $S/bird_train --db-roots $S/rrdbs \
    --dev-dbs $S/dev_dbs_bird.json --split train --max-q 9000 --out $o/cands_train.jsonl || die gen_train_$role
  [ -s $o/cands_bird_dev.jsonl ] || x gen_bird_dev_$role "${G[@]}" --plans $S/bird_dev --db-roots $D/dev_databases \
    --dev-dbs $S/bird_dev_dbs.json --split dev --all-questions --questions $D/dev.json --out $o/cands_bird_dev.jsonl || die gen_bird_dev_$role
  rm -f "$L/gen_bird_dev_$role.log"   # dev blind
  [ -s $o/rr/rerank_head.pt ] || x rrtrain_$role $V scripts/rerank_sq2.py train --cards $S/bird_train $S/bird_dev \
    --train-cands $o/cands_train.jsonl --dev-cands $o/cands_dev.jsonl --out $o/rr --base seeklhy/OmniSQL-7B \
    --subfolder "" --hours 1.5 --epochs 1 --train-k $K || die rrtrain_$role
  log "DONE_ROLE $role"
done
for set in heldout bird_dev; do
  f=cands_dev.jsonl; cards=$S/bird_train; [ $set = bird_dev ] && { f=cands_bird_dev.jsonl; cards=$S/bird_dev; }
  [ -s $O/${set}_union.jsonl ] || $V scripts/analysis/union_pools.py $O/r1/$f $O/r2/$f $O/${set}_union.jsonl || die union_$set
  for v in r1 r2; do
    out=$O/${set}_union_by_$v.jsonl
    [ -s $out ] || x score_${set}_$v $V scripts/rerank_sq2.py score --rr $O/$v/rr --cards $cards \
      --dev-cands $O/${set}_union.jsonl --out $out --base seeklhy/OmniSQL-7B --subfolder "" || die score_${set}_$v
    [ $set = bird_dev ] && rm -f "$L/score_${set}_$v.log"
  done
done
log PHASE1_W_DONE
