#!/bin/bash
# Head-score dumps (scripts/dump_heads.py) for the R1 and R2 checkpoints, one single-GPU box, ON the box, detached:
#   ROLES="r1 r2" setsid nohup bash /workspace/luthor/scripts/chains/dump_box.sh > /workspace/logs/dump.log 2>&1 < /dev/null &
# Needs on the box (pushed from the Mac, as launch_r1r2.sh does, plus the checkpoints):
#   repo at $R/luthor with runs/r1r2/<role>/{best_lora/,best_heads.pt,args.json,vocab.json} and, for the parity check,
#   runs/r1r2/<role>/rr/cands_{dev,bird_dev,train}.jsonl; $D/sys1/{bird_train,bird_dev,dbs,rrdbs,dev_dbs_bird.json,
#   bird_dev_dbs.json}; $D/dev_databases and $D/dev.json (original BIRD dev: all 1,534 questions, evaluation only).
#
# Per role, each stage skipped when its manifest exists (a re-launch resumes):
#   heldout   the 1,000 held-out training-DB questions of rr/cands_dev.jsonl (rerank_sq2 gen --split dev on bird_train)
#   bird_dev  all 1,534 BIRD dev questions of rr/cands_bird_dev.jsonl (--all-questions --questions $D/dev.json)
#   train     only with TRAIN_Q > 0: the first TRAIN_Q questions of gen's BIRD-train order (rr/cands_train.jsonl)
# Every dump compares its flip pool with the gen file (CHECK_CANDS line: share of identical candidate lists) and is
# then replayed on CPU (VERIFY line; exits non-zero unless greedy, flips and k-best come back identical).
# Last: $R/dumps/headdump_<roles>.tar (manifests + shards) and logs/dump.done (logs/dump.failed on a fatal error).
#
# Env: ROLES ("r1 r2"), TRAIN_Q (0), KBEST ("8": lazy k-best sizes recorded), FLIPS_K (8), STATE_DTYPE (float16),
#   LUTHOR_GRAMMAR (v3: the R1/R2 checkpoints are grammar v3; dump_heads.py checks it against each run's args.json),
#   SKIP_SETUP=1 (venv6 + OmniSQL-7B already there), DRY=1 (echo every command; root $TMPDIR/dump_dry).
set -euo pipefail
ROLES=${ROLES:-r1 r2}; TRAIN_Q=${TRAIN_Q:-0}; KBEST=${KBEST:-8}; FLIPS_K=${FLIPS_K:-8}; STATE_DTYPE=${STATE_DTYPE:-float16}
export LUTHOR_GRAMMAR=${LUTHOR_GRAMMAR:-v3}
DRY=${DRY:-0}
if [ "$DRY" = 1 ]; then R=${REMOTE_ROOT:-${TMPDIR:-/tmp}/dump_dry}; else R=${REMOTE_ROOT:-/workspace}; fi
D=${REMOTE_D:-/root/d}; [ "$DRY" = 1 ] && D=${REMOTE_D:-$R/d}
S=$D/sys1; L=$R/logs; REPO=$R/luthor; OUT=$R/dumps; V=$R/venv6/bin/python
mkdir -p "$L" "$OUT"
[ "$DRY" = 1 ] && mkdir -p "$REPO"
cd "$REPO"
export HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH=$R/venv6/bin:$PATH
STAGE=boot
log() { echo "$(date -u '+%m-%d %H:%M:%S') [dump] $*"; }
fail() { log "FATAL in stage $STAGE: $*"; echo "DUMP_FAIL"; echo "$STAGE: $*" > "$L/dump.failed"; exit 1; }
trap 'rc=$?; [ $rc = 0 ] || [ -e "$L/dump.failed" ] || echo "$STAGE: exit $rc" > "$L/dump.failed"' EXIT
x() {  # x NAME cmd...: all output in $L/NAME.log; DRY=1 only echoes
  local name=$1; shift
  if [ "$DRY" = 1 ]; then log "DRY: $* > $L/$name.log 2>&1"; return 0; fi
  log "run $name: $*"; local rc=0
  "$@" > "$L/$name.log" 2>&1 || rc=$?
  [ $rc = 0 ] && log "ok $name" || { log "$name exited $rc; tail:"; tail -n 25 "$L/$name.log"; }
  return $rc
}

# ---- 0. GPU check + setup --------------------------------------------------------------------------------------------
if [ "$DRY" != 1 ]; then
  g=$(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits) || fail "nvidia-smi failed"
  log "GPUs: $g"
  echo "$g" | awk -F', *' '$3 + 0 != 0 { bad = 1 } END { exit !bad }' && fail "GPU memory already in use at boot"
fi
STAGE=setup
if [ "${SKIP_SETUP:-0}" = 1 ] || [ -e "$L/setup.ok" ]; then log "setup skipped / already done"
elif [ "$DRY" = 1 ]; then log "DRY: EXPECT_GPUS=1 MODELS=seeklhy/OmniSQL-7B bash scripts/vast/setup6.sh"
else
  EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 \
    || { tail -n 30 "$L/setup.log"; fail "setup6.sh exited non-zero"; }
  grep -qa '^SETUP6_FAIL' "$L/setup.log" && fail "SETUP6_FAIL in setup.log"
  grep -qa '^DONE_SETUP6' "$L/setup.log" || fail "setup.log has no DONE_SETUP6"
  touch "$L/setup.ok"
fi
STAGE=preflight
for p in "$S/bird_train" "$S/bird_dev" "$S/dbs" "$S/dev_dbs_bird.json" "$S/bird_dev_dbs.json" "$D/dev_databases" "$D/dev.json"; do
  [ "$DRY" = 1 ] || [ -e "$p" ] || fail "missing pushed data: $p"
done
[ "$TRAIN_Q" -gt 0 ] && { [ "$DRY" = 1 ] || [ -e "$S/rrdbs" ] || fail "missing pushed data: $S/rrdbs"; }
for role in $ROLES; do
  run=runs/r1r2/$role
  for f in best_heads.pt best_lora/adapter_model.safetensors args.json vocab.json; do
    [ "$DRY" = 1 ] || [ -e "$run/$f" ] || fail "missing checkpoint file $run/$f"
  done
done
echo DONE_PREFLIGHT

# ---- 1. dumps --------------------------------------------------------------------------------------------------------
# dump ROLE NAME CANDS_FILE args...: one dump (skipped when its manifest exists), then the CPU replay check
dump() {
  local role=$1 name=$2 cands=$3; shift 3
  local out=$OUT/$role/$name
  STAGE="dump $role/$name"
  if [ -s "$out/manifest.json" ]; then log "$role/$name already dumped"
  else
    rm -rf "$out"
    # shellcheck disable=SC2086
    x "dump_${role}_$name" "$V" scripts/dump_heads.py dump --run "runs/r1r2/$role" --out "$out" --name "$name" \
      --flips-k "$FLIPS_K" --kbest-k $KBEST --state-dtype "$STATE_DTYPE" --check-cands "runs/r1r2/$role/rr/$cands" "$@" \
      || fail "dump_heads.py dump ($role/$name)"
  fi
  [ "$DRY" = 1 ] || grep -aE '^(DUMP|CHECK_CANDS)' "$L/dump_${role}_$name.log" || true
  STAGE="verify $role/$name"
  x "verify_${role}_$name" "$V" scripts/dump_heads.py verify --dump "$out" || log "VERIFY_STRICT_MISMATCH $role/$name (non-fatal; re-check on the Mac)"
  [ "$DRY" = 1 ] || grep -a '^VERIFY' "$L/verify_${role}_$name.log" | cut -c1-400 || true
}
for role in $ROLES; do
  dump "$role" heldout cands_dev.jsonl --plans "$S/bird_train" --db-roots "$S/dbs" --dev-dbs "$S/dev_dbs_bird.json" --split dev
  dump "$role" bird_dev cands_bird_dev.jsonl --plans "$S/bird_dev" --db-roots "$D/dev_databases" \
    --dev-dbs "$S/bird_dev_dbs.json" --split dev --all-questions --questions "$D/dev.json"
  if [ "$TRAIN_Q" -gt 0 ]; then
    dump "$role" "train$TRAIN_Q" cands_train.jsonl --plans "$S/bird_train" --db-roots "$S/rrdbs" \
      --dev-dbs "$S/dev_dbs_bird.json" --split train --max-q "$TRAIN_Q"
  fi
done
echo DONE_DUMPS

# ---- 2. pack for the Mac ---------------------------------------------------------------------------------------------
STAGE=pack
tag=$(echo "$ROLES" | tr ' ' '_')
if [ "$DRY" = 1 ]; then log "DRY: tar -cf $OUT/headdump_$tag.tar -C $OUT $ROLES"
else
  tar -cf "$OUT/headdump_$tag.tar" -C "$OUT" $ROLES || fail "tar"
  log "packed $(du -h "$OUT/headdump_$tag.tar" | cut -f1) -> $OUT/headdump_$tag.tar"
fi
echo "$(date -u +%FT%TZ) $ROLES" > "$L/dump.done"; STAGE=done
log "ALL DONE"; echo DONE_DUMP
