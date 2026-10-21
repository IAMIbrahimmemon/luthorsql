#!/bin/bash
# Validator with result previews (research problem 07, "preview never tried on the System-1 validator at scale"), ON
# the box:  setsid nohup bash /workspace/luthor/scripts/chains/valprev_box.sh > /workspace/logs/vp.log 2>&1 < /dev/null &
# Same recipe as box W's wide R2 validator (scripts/chains/phase1_w_box.sh rrtrain_r2: R2's k=32 BIRD-train lists,
# --train-k 32, 1 epoch) plus --result-preview, stopped at the 147 steps the box-W validator reached (its 1.5 h cap),
# so the two differ only in the preview. Then it scores the held-out R1 ∪ R2 union pool. No BIRD dev here.
# Needs (pushed by run_valprev_box.sh): /workspace/p1w/r2/{cands_train,cands_dev}.jsonl, /workspace/p1w/heldout_union.jsonl
set -uo pipefail
R=/workspace; D=/root/d; S=$D/sys1; L=$R/logs; O=$R/p1w; OUT=$R/valprev; V=$R/venv6/bin/python
mkdir -p "$L" "$OUT"; cd $R/luthor
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH=$R/venv6/bin:$PATH
STEPS=${STEPS:-147}
log() { echo "$(date -u '+%m-%d %H:%M:%S') [vp] $*"; }
die() { log "VALPREV_FAIL $*"; exit 1; }
x() { local n=$1; shift; log "run $n"; "$@" > "$L/$n.log" 2>&1; local rc=$?; [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 20 "$L/$n.log" | sed 's/Traceback/TB/'; }; return $rc; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -e "$L/setup.ok" ]; then
  EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/'; die setup; }
  grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
  touch "$L/setup.ok"
fi
log DONE_SETUP
for f in $O/r2/cands_train.jsonl $O/r2/cands_dev.jsonl $O/heldout_union.jsonl; do [ -s $f ] || die "missing $f"; done
[ -s $OUT/rr/rerank_head.pt ] || x rrtrain_prev $V scripts/rerank_sq2.py train --cards $S/bird_train $S/bird_dev \
  --train-cands $O/r2/cands_train.jsonl --dev-cands $O/r2/cands_dev.jsonl --out $OUT/rr --base seeklhy/OmniSQL-7B \
  --subfolder "" --epochs 1 --max-steps $STEPS --train-k 32 --result-preview || die rrtrain_prev
grep -a "trained\|RERANK DEV {" "$L/rrtrain_prev.log" | cut -c1-300
[ -s $OUT/heldout_union_by_prev.jsonl ] || x score_heldout_prev $V scripts/rerank_sq2.py score --rr $OUT/rr \
  --cards $S/bird_train --dev-cands $O/heldout_union.jsonl --out $OUT/heldout_union_by_prev.jsonl \
  --base seeklhy/OmniSQL-7B --subfolder "" || die score_heldout_prev
cp $L/rrtrain_prev.log $OUT/
log VALPREV_DONE
