#!/bin/bash
# VAL3 validator, ONE arm + ONE seed per box (runs/phase1/val3/PREREG.md; docs/paper_notes/research/validator/
# val3_trainer.md), ON the box (started by scripts/chains/run_val3_box.sh, which stages + pushes everything first):
#   ARM=A SEED=1 MAX_HOURS=3.8 setsid nohup bash /workspace/luthor/scripts/chains/val3_box.sh > /workspace/logs/val3.log 2>&1 < /dev/null &
# System 1 only: one System-1 validator (rerank LoRA r=16 + head on frozen OmniSQL-7B) trained with rerank_sq2.py train
# v2 on the data-v3 lists, then the tail-averaged model scores R1's two held-out pools. Nothing is generated.
#   ARM A  data v3 + recipe: 4 epochs, cosine, warmup 0.03, --loss group, label smoothing 0.05, --tok-budget TOK (60000),
#          --eval-every 0.5, --avg-last 4, --seed SEED --init-seed SEED (pre-registered: init seed = seed); VALHARD's
#          hyperparameters otherwise (lr 1e-4, head lr 1e-3, q-per-db 4, dbs-per-step 4, lora r 16, budget 9000 chars)
#   ARM B  A + --joint-results --joint-orders 2 (score reads joint scoring from settings.json)
# Output: ONE dir per arm + seed, $O/rr_${ARM}_s${SEED} (the averaged model; best/, last/, ckpt/ inside). A model
# found there is reused ONLY when this chain's stamp (VAL3_TRAINED.json: arm, seed, data sha256, recipe) matches;
# an unstamped or mismatched model is a VAL3_FAIL, never a silent reuse; a partial dir (no head) is moved aside.
# Scored (label-free pools; DECISION + RESERVED + retail_world rows, no label anywhere on the box):
#   $O/k32_by_${ARM}_s${SEED}.jsonl   R1's k=32 held-out pool        (/root/d/valhard/val3/k32_pool.jsonl)
#   $O/k64r_by_${ARM}_s${SEED}.jsonl  the k=64+repair merged pool    (/root/d/valhard/val3/k64r_pool.jsonl)
# RESERVED stays blind: the pools carry no ok / gold field, and the SCORE lines are deleted from the score logs anyway.
# Ends VAL3_DONE (VAL3_FAIL <why> on any error). No BIRD dev, no sealed set, no gate.
# Preflight BEFORE setup uses only bash, coreutils and the SYSTEM python3 (never $R/venv6, which setup creates).
# Env: ARM (A|B) SEED (int) MAX_HOURS (trainer --max-hours; run_val3_box sets ~1.3x the estimate) TOK (60000)
#      DRY=1 (echo the commands; with REMOTE_ROOT / REMOTE_D / PY pointing at local paths it runs anywhere)
set -uo pipefail
R=${REMOTE_ROOT:-/workspace}; D=${REMOTE_D:-/root/d}; VD=$D/valhard/val3; L=$R/logs; O=$R/val3
V=${PY:-$R/venv6/bin/python}; REPO=${REPO:-$R/luthor}; DRY=${DRY:-0}
ARM=${ARM:-}; SEED=${SEED:-}; MAX_HOURS=${MAX_HOURS:-}; TOK=${TOK:-60000}; EPOCHS=4
SYSPY=$(PATH=/usr/local/bin:/usr/bin:/bin command -v python3 || true)  # system python only (not the venv)
if command -v sha256sum >/dev/null; then SHA=(sha256sum); else SHA=(shasum -a 256); fi  # (shasum: DRY runs on a Mac)
mkdir -p "$L" "$O"
log() { echo "$(date -u '+%m-%d %H:%M:%S') [val3 ${ARM}_s${SEED}] $*"; }
die() { log "VAL3_FAIL $*"; exit 1; }
cd "$REPO" || die "no repo $REPO"
case "$ARM" in A|B) ;; *) die "ARM must be A or B (got '$ARM')";; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a non-negative integer (got '$SEED')"
[[ "$MAX_HOURS" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "MAX_HOURS required (got '$MAX_HOURS')"
[[ "$TOK" =~ ^[0-9]+$ ]] || die "bad TOK '$TOK'"
RR=$O/rr_${ARM}_s${SEED}; TAG=${ARM}_s${SEED}
CARDS="$D/sys1/bird_train $VD/cards"  # bird_train FIRST: the held-out cards are VALHARD's (load_cards: first dir wins)
TRAIN_FLAGS=(--epochs $EPOCHS --sched cosine --warmup 0.03 --loss group --label-smooth 0.05 --tok-budget "$TOK"
             --eval-every 0.5 --avg-last 4 --seed "$SEED" --init-seed "$SEED" --max-hours "$MAX_HOURS")
[ "$ARM" = B ] && TRAIN_FLAGS+=(--joint-results --joint-orders 2)
BASE=(--base seeklhy/OmniSQL-7B --subfolder "" --lora-r 16)
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
x() {  # x NAME cmd...: output in $L/NAME.log; no label-derived line is echoed; DRY=1 only echoes
  local n=$1; shift
  if [ "$DRY" = 1 ]; then log "DRY: $* > $L/$n.log"; return 0; fi
  log "run $n"; "$@" > "$L/$n.log" 2>&1; local rc=$?
  case "$n" in score_*) sed -i.bak -E '/^(SCORE |RERANK DEV )/d' "$L/$n.log" 2>/dev/null; rm -f "$L/$n.log.bak";; esac
  [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 20 "$L/$n.log" | grep -avE 'GATE|^(SCORE|RERANK DEV) ' | sed 's/Traceback/TB/'; }
  return $rc
}
log "arm $ARM seed $SEED (init seed $SEED) tok $TOK epochs $EPOCHS max_hours $MAX_HOURS -> $RR; cards $CARDS"
log "train flags: ${TRAIN_FLAGS[*]}"

# ---- preflight (system tools only; seconds, before setup) ------------------------------------------------------------
if [ "$DRY" != 1 ]; then
  g=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits) || die "nvidia-smi failed"
  log "GPUs: $g"
  echo "$g" | awk -F', *' '$3 + 0 > 512 { bad = 1 } END { exit !bad }' && die "GPU memory already in use at boot"
  echo "$g" | head -1 | awk -F', *' '$4 + 0 < 79000 { bad = 1 } END { exit !bad }' && die "GPU 0 has < 80 GB (TOK $TOK is sized for an 80 GB H100)"
fi
for f in $VD/train_lists.jsonl $VD/val_lists.jsonl $VD/k32_pool.jsonl $VD/k64r_pool.jsonl $VD/MANIFEST.json \
         $VD/MANIFEST.sha256 scripts/rerank_sq2.py src/luthor/sq2/rerank.py; do
  [ -s "$f" ] || die "missing $f"
done
[ -d $D/sys1/bird_train/cards ] || die "missing $D/sys1/bird_train/cards"
[ -d $VD/cards/cards ] || die "missing $VD/cards/cards"
( cd $VD && "${SHA[@]}" --status -c MANIFEST.sha256 ) || die "staged files do not match MANIFEST.sha256 (partial push?)"
for p in $VD/k32_pool.jsonl $VD/k64r_pool.jsonl; do  # the pools must be label-free (RESERVED blind)
  grep -qE '"(ok|gold_sql|in_slots|covered)": ' "$p" && die "$p carries a label field: refusing (RESERVED must stay blind)"
done
# the trainer must know every flag this chain passes (grep the source: no venv before setup)
for f in --seed --init-seed --eval-every --avg-last --joint-results --joint-orders --tok-budget --label-smooth --warmup; do
  grep -q -- "\"$f\"" scripts/rerank_sq2.py || die "rerank_sq2.py has no $f (code not pushed?)"
done
grep -q '"group"' scripts/rerank_sq2.py || die "rerank_sq2.py has no --loss group"
if [ -n "$SYSPY" ]; then
  "$SYSPY" - "$VD" <<'PY' || die "staged data check failed"
import json, sys
vd = sys.argv[1]
m = json.load(open(f"{vd}/MANIFEST.json"))
for f in ("train_lists.jsonl", "val_lists.jsonl", "k32_pool.jsonl", "k64r_pool.jsonl"):
    with open(f"{vd}/{f}") as fh:
        r = json.loads(fh.readline())
    assert {"id", "db_id", "cands"} <= set(r), f
print("staged:", json.dumps(m["counts"]), "pools", json.dumps(m["pools"]))
PY
else
  log "no system python3: skipping the JSON shape check (sha256 + grep checks passed)"
fi
STAMP_WANT="arm=$ARM seed=$SEED init_seed=$SEED tok=$TOK epochs=$EPOCHS avg_last=4 eval_every=0.5 joint=$([ "$ARM" = B ] && echo 2 || echo 0) train=$("${SHA[@]}" < $VD/train_lists.jsonl | cut -c1-16) val=$("${SHA[@]}" < $VD/val_lists.jsonl | cut -c1-16)"
TRAINED=0
if [ -e "$RR/rerank_head.pt" ]; then
  if [ -s "$RR/VAL3_TRAINED" ] && [ "$(cat "$RR/VAL3_TRAINED")" = "$STAMP_WANT" ]; then
    TRAINED=1; log "reusing $RR: its stamp matches this arm / seed / data / recipe"
  else
    die "$RR/rerank_head.pt exists without a matching VAL3 stamp: refusing to reuse a model (move it away first)"
  fi
elif [ -e "$RR" ]; then
  mv "$RR" "$RR.partial.$(date -u +%s)" && log "moved a partial $RR aside (no averaged head): retraining from scratch"
fi
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
  for f in --seed --init-seed --eval-every --avg-last --joint-results --joint-orders; do
    echo "$h" | grep -q -- "$f" || die "rerank_sq2.py train --help has no $f"
  done
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- train (one arm, one seed, one output dir) -------------------------------------------------------------------------
if [ $TRAINED = 0 ]; then
  # shellcheck disable=SC2086
  x val3_train "$V" scripts/rerank_sq2.py train --cards $CARDS --train-file $VD/train_lists.jsonl \
    --val-file $VD/val_lists.jsonl --out "$RR" "${BASE[@]}" "${TRAIN_FLAGS[@]}" || die val3_train
  if [ "$DRY" != 1 ]; then
    [ -s "$RR/rerank_head.pt" ] && [ -s "$RR/avg_steps.json" ] || die "no averaged model in $RR (avg_steps.json missing)"
    grep -q '"avg_last": 4' "$RR/settings.json" || die "$RR/settings.json does not record avg_last 4"
    if [ "$ARM" = B ]; then grep -q '"joint_results": true' "$RR/settings.json" || die "arm B model not joint"
    else grep -q '"joint_results": false' "$RR/settings.json" || die "arm A model is joint"; fi
    echo "$STAMP_WANT" > "$RR/VAL3_TRAINED"
  fi
fi
if [ "$DRY" != 1 ]; then
  grep -aE '^(train lists|no card for|seed |[0-9]+ steps per epoch|VAL |SKIPPED_FOR_MEMORY|trained |averaged |time limit)' "$L/val3_train.log" | tail -20
  grep -qa 'time limit' "$L/val3_train.log" && log "RECIPE_DEVIATION: the time limit shortened the schedule (see val3_train.log)"
  cp -f "$L/val3_train.log" "$O/" 2>/dev/null
fi

# ---- score the two held-out pools with the averaged model (label-free; blind) ------------------------------------------
for p in k32 k64r; do
  out=$O/${p}_by_${TAG}.jsonl
  # shellcheck disable=SC2086
  [ -s "$out" ] || x score_$p "$V" scripts/rerank_sq2.py score --rr "$RR" --cards $CARDS --dev-cands $VD/${p}_pool.jsonl \
    --out "$out" "${BASE[@]}" || die score_$p
  if [ "$DRY" != 1 ]; then
    if [ "$ARM" = B ]; then grep -qa 'joint results scoring, 2 table order' "$L/score_$p.log" || die "score_$p did not use joint scoring"
    else grep -qa 'joint results scoring' "$L/score_$p.log" && die "score_$p used joint scoring for arm A"; fi
    log "$p: $(wc -l < "$out" | tr -d ' ') of $(wc -l < $VD/${p}_pool.jsonl | tr -d ' ') pool rows scored"
  fi
done
# no SCORE print: the pools include RESERVED, which stays blind (scored on the Mac, DECISION only: val3_eval.py)
log VAL3_DONE
