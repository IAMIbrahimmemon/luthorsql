#!/bin/bash
# Question-variant ensembling test, ON the box:
#   setsid nohup bash /workspace/luthor/scripts/chains/qvar_box.sh > /workspace/logs/qv.log 2>&1 < /dev/null &
# For each variant in /root/d/sys1/qvar/<v> (scripts/analysis/qvariants.py: orig p1 p2 p3 f s; only the question text
# differs): R2 k=8 lists on the held-out training DBs, scored by R2's frozen validator under that variant's wording.
# Combination happens on the Mac (scripts/analysis/qvar_eval.py). No BIRD dev. Ends with QVAR_DONE.
set -uo pipefail
R=/workspace; D=/root/d; S=$D/sys1; L=$R/logs; OUT=$R/qvar; V=$R/venv6/bin/python
mkdir -p "$L" "$OUT"; cd $R/luthor
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH=$R/venv6/bin:$PATH
W=$(( $(nproc) - 2 )); [ "$W" -lt 2 ] && W=2
log() { echo "$(date -u '+%m-%d %H:%M:%S') [qv] $*"; }
die() { log "QVAR_FAIL $*"; exit 1; }
x() { local n=$1; shift; log "run $n"; "$@" > "$L/$n.log" 2>&1; local rc=$?; [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 20 "$L/$n.log" | sed 's/Traceback/TB/'; }; return $rc; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -e "$L/setup.ok" ]; then
  EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/'; die setup; }
  grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
  touch "$L/setup.ok"
fi
log DONE_SETUP
for v in ${VARIANTS:-orig p1 p2 p3 f s}; do
  [ -s $S/qvar/$v/plans.jsonl ] || die "missing $S/qvar/$v"
  [ -s $OUT/cands_$v.jsonl ] || x gen_$v $V scripts/rerank_sq2.py gen --run runs/r1r2/r2 --k 8 --workers $W \
    --plans $S/qvar/$v --db-roots $S/dbs --dev-dbs $S/dev_dbs_bird.json --split dev --out $OUT/cands_$v.jsonl || die gen_$v
  grep -aE '^(GEN|greedy|oracle)' "$L/gen_$v.log" | tail -2
  [ -s $OUT/scored_$v.jsonl ] || x score_$v $V scripts/rerank_sq2.py score --rr runs/r1r2/r2/rr --cards $S/qvar/$v \
    --dev-cands $OUT/cands_$v.jsonl --out $OUT/scored_$v.jsonl --base seeklhy/OmniSQL-7B --subfolder "" || die score_$v
done
cp $L/gen_*.log $OUT/ 2>/dev/null
log QVAR_DONE
