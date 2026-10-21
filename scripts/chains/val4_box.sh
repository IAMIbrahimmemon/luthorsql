#!/bin/bash
# VAL4 validator, ONE seed per box (runs/phase1/val4/PREREG.md), ON the box. A copy of valhard_box.sh (VALHARD's frozen
# script stays untouched) started by run_val4_box.sh:
#   SEED=1 VH_TAG=val4 setsid nohup bash /workspace/luthor/scripts/chains/val4_box.sh > /workspace/logs/val4.log 2>&1 < /dev/null &
# One System-1 validator (rerank LoRA r=16 + head on frozen OmniSQL-7B) trained with `rerank_sq2.py train` v2 on the
# VAL4 lists /root/d/valhard/$VH_TAG/{train_lists,val_lists}.jsonl with VALHARD's recipe exactly (2 epochs, warmup +
# cosine, --loss group, label smoothing 0.05, --tok-budget 120000, lr / head lr / q-per-db / dbs-per-step / budget at
# the defaults, best epoch by val pick accuracy) plus --seed SEED --init-seed SEED (PREREG: seeds 1 and 2). Then the model
# scores R1's two held-out pools (label-free on the box, as val3_box.sh):
#   $O/k32_by_s${SEED}.jsonl    R1's k=32 held-out pool (VALHARD's data/valhard/r1/heldout_pool.jsonl)  <- k32_pool.jsonl
#   $O/k64r_by_s${SEED}.jsonl   the k=64+repair merged pool (runs/phase1/repair/repair/k64_repair_pool.jsonl) <- k64r_pool.jsonl
# Model in $O/rr_s${SEED} (stamped VAL4_TRAINED: a model is reused only when the stamp matches seed / data / recipe).
# Ends VAL4_DONE (VAL4_FAIL <why> on errors). No BIRD dev, no sealed sets, no gate; RESERVED stays blind (no label on the
# box; SCORE lines are deleted from the score logs anyway).
# Preflight BEFORE setup uses only bash, coreutils, awk and the SYSTEM python3 (never $R/venv6, which setup creates).
# Env: SEED (int, required), VH_TAG (val4), CARDS (default /root/d/sys1/bird_train), LOSS (group) EPOCHS (2) TOK (120000)
#      MAX_HOURS (6.5; run_val4_box.sh raises it when the lists are larger), DRY=1 (echo the commands; with REMOTE_ROOT /
#      REMOTE_D / PY / REPO pointing at local paths it runs anywhere)
set -uo pipefail
R=${REMOTE_ROOT:-/workspace}; D=${REMOTE_D:-/root/d}; VH_TAG=${VH_TAG:-val4}; VH=$D/valhard/$VH_TAG; L=$R/logs; O=$R/val4
V=${PY:-$R/venv6/bin/python}; REPO=${REPO:-$R/luthor}; DRY=${DRY:-0}
CARDS=${CARDS:-$D/sys1/bird_train}; SEED=${SEED:-}
LOSS=${LOSS:-group}; EPOCHS=${EPOCHS:-2}; TOK=${TOK:-120000}; MAX_HOURS=${MAX_HOURS:-6.5}
SYSPY=$(PATH=/usr/local/bin:/usr/bin:/bin command -v python3 || true)  # system python only (not the venv)
if command -v sha256sum >/dev/null; then SHA=(sha256sum); else SHA=(shasum -a 256); fi  # (shasum: DRY runs on a Mac)
mkdir -p "$L" "$O"
log() { echo "$(date -u '+%m-%d %H:%M:%S') [val4 s${SEED}] $*"; }
die() { log "VAL4_FAIL $*"; exit 1; }
cd "$REPO" || die "no repo $REPO"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a non-negative integer (got '$SEED')"
[[ "$MAX_HOURS" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "bad MAX_HOURS '$MAX_HOURS'"
[[ "$TOK" =~ ^[0-9]+$ ]] || die "bad TOK '$TOK'"
RR=$O/rr_s$SEED; TAG=s$SEED
TRAIN_FLAGS=(--lora-r 16 --epochs "$EPOCHS" --sched cosine --loss "$LOSS" --label-smooth 0.05 --tok-budget "$TOK"
             --max-hours "$MAX_HOURS" --seed "$SEED" --init-seed "$SEED")
BASE=(--base seeklhy/OmniSQL-7B --subfolder "")
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
x() {  # x NAME cmd...: output in $L/NAME.log; no label-derived line is echoed; DRY=1 only echoes
  local n=$1; shift
  if [ "$DRY" = 1 ]; then log "DRY: $* > $L/$n.log"; return 0; fi
  log "run $n"; "$@" > "$L/$n.log" 2>&1; local rc=$?
  case "$n" in score_*) sed -i.bak -E '/^(SCORE |RERANK DEV )/d' "$L/$n.log" 2>/dev/null; rm -f "$L/$n.log.bak";; esac
  [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 20 "$L/$n.log" | grep -avE 'GATE|^(SCORE|RERANK DEV) ' | sed 's/Traceback/TB/'; }
  return $rc
}
log "tag $VH_TAG seed $SEED (init seed $SEED) cards $CARDS loss $LOSS epochs $EPOCHS tok $TOK max_hours $MAX_HOURS -> $RR"
log "train flags: ${TRAIN_FLAGS[*]}"

# ---- preflight (system tools only; seconds, before setup) ------------------------------------------------------------
if [ "$DRY" != 1 ]; then
  g=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits) || die "nvidia-smi failed"
  log "GPUs: $g"
  echo "$g" | awk -F', *' '$3 + 0 > 512 { bad = 1 } END { exit !bad }' && die "GPU memory already in use at boot (> 512 MiB)"
  echo "$g" | head -1 | awk -F', *' '$4 + 0 < 79000 { bad = 1 } END { exit !bad }' && die "GPU 0 has < 80 GB (TOK $TOK is sized for an 80 GB H100)"
fi
for f in $VH/train_lists.jsonl $VH/val_lists.jsonl $VH/k32_pool.jsonl $VH/k64r_pool.jsonl $VH/MANIFEST.sha256 \
         scripts/rerank_sq2.py src/luthor/sq2/rerank.py; do
  [ -s "$f" ] || die "missing $f"
done
[ -d $CARDS/cards ] || die "missing $CARDS/cards"
( cd $VH && "${SHA[@]}" --status -c MANIFEST.sha256 ) || die "staged files do not match MANIFEST.sha256 (partial push?)"
for p in $VH/k32_pool.jsonl $VH/k64r_pool.jsonl; do  # the pools must be label-free (RESERVED blind)
  grep -qE '"(ok|gold_sql|in_slots|covered)": ' "$p" && die "$p carries a label field: refusing (RESERVED must stay blind)"
done
for f in --seed --init-seed --tok-budget --label-smooth --sched --loss --max-hours --lora-r --val-file --train-file; do
  grep -q -- "\"$f\"" scripts/rerank_sq2.py || die "rerank_sq2.py has no $f (code not pushed?)"
done
grep -q '"group"' scripts/rerank_sq2.py || die "rerank_sq2.py has no --loss group"
if [ -n "$SYSPY" ]; then
  "$SYSPY" - "$VH" <<'PY' || die "staged data check failed"
import json, sys
vh = sys.argv[1]
for f in ("train_lists.jsonl", "val_lists.jsonl", "k32_pool.jsonl", "k64r_pool.jsonl"):
    with open(f"{vh}/{f}") as fh:
        r = json.loads(fh.readline())
    assert {"id", "db_id", "cands"} <= set(r), f
print("staged lists + pools ok")
PY
elif [ "$DRY" != 1 ]; then die "no system python3 for the preflight"
fi
STAMP_WANT="seed=$SEED init_seed=$SEED loss=$LOSS tok=$TOK epochs=$EPOCHS train=$("${SHA[@]}" < $VH/train_lists.jsonl | cut -c1-16) val=$("${SHA[@]}" < $VH/val_lists.jsonl | cut -c1-16)"
TRAINED=0
if [ -e "$RR/rerank_head.pt" ]; then
  if [ -s "$RR/VAL4_TRAINED" ] && [ "$(cat "$RR/VAL4_TRAINED")" = "$STAMP_WANT" ]; then
    TRAINED=1; log "reusing $RR: its stamp matches this seed / data / recipe"
  else
    die "$RR/rerank_head.pt exists without a matching VAL4 stamp: refusing to reuse a model (move it away first)"
  fi
elif [ -e "$RR" ]; then
  mv "$RR" "$RR.partial.$(date -u +%s)" && log "moved a partial $RR aside (no head): retraining from scratch"
fi
wc -l $VH/train_lists.jsonl $VH/val_lists.jsonl $VH/k32_pool.jsonl $VH/k64r_pool.jsonl
log "preflight ok"

# ---- setup (venv + OmniSQL-7B) ----------------------------------------------------------------------------------------
if [ "$DRY" != 1 ] && [ ! -e "$L/setup.ok" ]; then
  EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/'; die setup; }
  grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
  touch "$L/setup.ok"
fi
log DONE_SETUP
export PATH=$(dirname "$V"):$PATH
if [ "$DRY" != 1 ]; then
  [ -x "$V" ] || die "no venv python at $V after setup"
  h=$("$V" scripts/rerank_sq2.py train --help 2>&1) || die "rerank_sq2.py --help failed: $(echo "$h" | tail -2)"
  for f in --seed --init-seed --loss --label-smooth --tok-budget; do
    echo "$h" | grep -q -- "$f" || die "rerank_sq2.py train --help has no $f"
  done
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- train (VALHARD's recipe + the seed) ------------------------------------------------------------------------------
if [ $TRAINED = 0 ]; then
  # shellcheck disable=SC2086
  x val4_train "$V" scripts/rerank_sq2.py train --cards $CARDS --train-file $VH/train_lists.jsonl \
    --val-file $VH/val_lists.jsonl --out "$RR" "${BASE[@]}" "${TRAIN_FLAGS[@]}" || die val4_train
  if [ "$DRY" != 1 ]; then
    [ -s "$RR/rerank_head.pt" ] || die "no model in $RR"
    grep -q "\"seed\": $SEED," "$RR/settings.json" || die "$RR/settings.json does not record seed $SEED"
    echo "$STAMP_WANT" > "$RR/VAL4_TRAINED"
  fi
fi
if [ "$DRY" != 1 ]; then
  grep -aE '^(train lists|no card for|seed |[0-9]+ steps per epoch|VAL |SKIPPED_FOR_MEMORY|trained |time limit)' "$L/val4_train.log" | tail -16
  grep -qa 'time limit' "$L/val4_train.log" && log "RECIPE_DEVIATION: the time limit shortened the schedule (see val4_train.log)"
  cp -f "$L/val4_train.log" "$O/" 2>/dev/null
fi

# ---- score the two held-out pools (label-free; blind) ------------------------------------------------------------------
for p in k32 k64r; do
  out=$O/${p}_by_${TAG}.jsonl
  # shellcheck disable=SC2086
  [ -s "$out" ] || x score_$p "$V" scripts/rerank_sq2.py score --rr "$RR" --cards $CARDS --dev-cands $VH/${p}_pool.jsonl \
    --out "$out" "${BASE[@]}" --lora-r 16 || die score_$p
  [ "$DRY" = 1 ] || log "$p: $(wc -l < "$out" | tr -d ' ') of $(wc -l < $VH/${p}_pool.jsonl | tr -d ' ') pool rows scored"
done
# no SCORE print: the pools include RESERVED, which stays blind (scored on the Mac, DECISION only)
log VAL4_DONE
