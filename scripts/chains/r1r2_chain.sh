#!/bin/bash
# System-1 R1/R2 experiment, one role per single-GPU box (A100 80GB). Runs ON the box, detached:
#   ROLE=R1 N=6714 setsid nohup bash /workspace/luthor/scripts/chains/r1r2_chain.sh > /workspace/logs/chain.log 2>&1 < /dev/null &
# (scripts/chains/launch_r1r2.sh does the push and this launch from the Mac; watch_r1r2.sh follows both boxes.)
#
# Stages (each skipped when its output already exists, so a re-launch resumes):
#   0 boot: GPU memory.used must be 0, then setup6.sh (venv, OmniSQL-7B prefetch); any SETUP6_FAIL stops the chain.
#     preflight: the model flags this chain passes must exist in train_sq2.py / rerank_sq2.py (fails in minutes, not hours)
#   1 rare data: qwrite write + check --data $D for this box's shard (R1 -> 0/2, R2 -> 1/2) on $D/rare1 (pushed from
#     data/rare1: selected.jsonl), marker logs/rare_shard.{done,failed}. The Mac (watch_r1r2.sh) pulls both boxes'
#     $D/rare1/work/ (written/checked.sIof2.jsonl), pushes the union back to both, then writes logs/rare.go ("use" or
#     "none"). The box builds the rows on CPU (qwrite.py build -> $D/rare1/plans.jsonl + cards/). On any failure, a timeout, or
#     < 1,000 rows it continues WITHOUT rare data (loud RARE_SKIPPED line) instead of stalling.
#   2 train: sys1_full's exact recipe with --max-steps STEPS (R1: N, R2: 2N) --cand-order v2 --value-side, rare plans
#     added to --plans. sys1_full's --hours 4.0 --fit-hours are dropped: they would cut the fixed step schedule.
#     After ~50 and ~500 steps a PROJECTION line gives the measured s/step and the projected finish.
#   3 eval, exactly as round 2's rr stage for sys1_full: K=8 candidates on BIRD-train (rrdbs, 9,000 q max) and held-out
#     training DBs, BIRD dev candidates (DEVALL_FLAGS: all 1,534 questions), rerank adapter (1 h, 1 epoch), scored dev
#     -> runs/r1r2/<role>/rr/bird_dev_scored.jsonl (same format as runs/sys1/r6c/sys1_full/rr/bird_dev_scored.jsonl).
#   4 summary JSON, then logs/r1r2.done (logs/r1r2.failed with the stage name if anything fatal happened).
#
# Env: ROLE (R1|R2, required), N (base step count, default 6714 = sys1_full's schedule), TRAIN_EXTRA (more train flags),
#   DEVALL_FLAGS (default "--all-questions --questions $D/dev.json": all 1,534; "" = slot-covered questions only), QW_WRITE_ARGS / QW_CHECK_ARGS
#   (extra qwrite.py args), QW_MODELS (checkpoints to prefetch for qwrite), RARE_BUILD (build command, run in the repo),
#   RARE_PLANS (plan dir the build writes; default $D/rare1), RARE_MIN (1000), RARE_MAX_MIN (qwrite time limit,
#   180), RARE_WAIT_MIN (wait for the Mac's go, 120), PRICE_H (1.35, for the projection), EVAL_EST_H (2.0),
#   SKIP_RARE=1, DRY=1 (echo every GPU/python command instead of running it; root defaults to $TMPDIR/r1r2_dry),
#   SEED (train_sq2.py --seed; unset = the flag is not passed, i.e. seed 0, the original order),
#   BIRD_PLANS (BIRD-train plan dir for training and the rr lists; default $S/bird_train; its basename must stay
#   bird_train for --repeat bird_train=3), GATE lines (BIRD dev accuracy) are no longer copied into chain.log / summary (dev stays blind),
#   GRAMMAR (v3|v4: exported as LUTHOR_GRAMMAR for every stage and passed as train_sq2.py --grammar; unset = v3 and no
#   flag, the R1/R2 commands unchanged), VOCAB (train --vocab, default $S/templates.json; GRAMMAR=v4 requires it, e.g.
#   $D/plans_v4/templates_union.json, and v4 plan dirs in place of $S/{bird_train,spider_train}: see
#   docs/SYSTEM1_FANOUT_R3.md "grammar v4 runs"), DROP_DBS (train_sq2.py --drop-dbs, e.g. $S/contaminated_dbs.json),
#   SPIDER_PLANS / SYNSQL_PLANS (training plan dirs, default $S/spider_train, $S/synsql_train) and BIRD_DEV_PLANS (BIRD
#   dev plans/cards for the blind BIRD dev list + validator cards, default $S/bird_dev): with BIRD_PLANS and RARE_PLANS
#   these point a GRAMMAR=v4 run at the pushed v4 dirs, e.g. /root/d/plans_v4/{bird_train,spider_train,synsql_train,rare1}
#   (run_phase2_box.sh v4desc). Unset = the v3 paths, the R1/R2/R3 commands unchanged.
#   DESC=1 (phase 3): BIRD-train column descriptions in the schema cards for train AND test (luthor.sq.bird_desc).
#   DESC_ARGS holds the flag, e.g. "--bird-desc /root/d/bird_train_desc" (pushed from data/raw/bird_train_desc; a bare
#   --bird-desc would look under the repo's data/raw on the box). It is passed to train_sq2.py, rerank_sq2.py gen (all
#   lists; gen also follows the run's args.json) and rerank_sq2.py train/score, so generator and validator prompts match;
#   preflight checks every script lists the flag and the directory exists. DESC=1 with no DESC_ARGS exits at once.
#   DESC_ARGS=none: descriptions already baked into the card JSON of BIRD_PLANS / RARE_PLANS (no flag passed).
#   WIDE_K (e.g. 32): after the eval stage also write k=WIDE_K --distinct lists for the held-out (DECISION) DBs and BIRD
#   dev into $RD/wide/ (for merging into the wide R1 u R2 pool later, as phase1_w_box.sh); unset = no wide stage.
set -euo pipefail
ROLE=${ROLE:?ROLE=R1 or ROLE=R2}; N=${N:-6714}
case "$ROLE" in R1) STEPS=$N; SHARD=0/2;; R2) STEPS=$((2 * N)); SHARD=1/2;; *) echo "bad ROLE $ROLE"; exit 2;; esac
role=$(echo "$ROLE" | tr 'A-Z' 'a-z')
DRY=${DRY:-0}
case "${GRAMMAR:-v3}" in v3|v4) ;; *) echo "bad GRAMMAR $GRAMMAR (v3|v4)"; exit 2;; esac
if [ "${GRAMMAR:-v3}" = v4 ] && [ -z "${VOCAB:-}" ]; then echo "GRAMMAR=v4 needs VOCAB (the v4 templates_union.json)"; exit 2; fi
export LUTHOR_GRAMMAR=${GRAMMAR:-v3}   # every python stage (and its workers) runs this grammar
if [ "$DRY" = 1 ]; then R=${REMOTE_ROOT:-${TMPDIR:-/tmp}/r1r2_dry}; else R=${REMOTE_ROOT:-/workspace}; fi
D=${REMOTE_D:-/root/d}; [ "$DRY" = 1 ] && D=${REMOTE_D:-$R/d}
S=$D/sys1; BIRD_PLANS=${BIRD_PLANS:-$S/bird_train}; L=$R/logs
SPIDER_PLANS=${SPIDER_PLANS:-$S/spider_train}; SYNSQL_PLANS=${SYNSQL_PLANS:-$S/synsql_train}
BIRD_DEV_PLANS=${BIRD_DEV_PLANS:-$S/bird_dev}
DESC=${DESC:-0}; DESC_ARGS=${DESC_ARGS:-}
case "$DESC" in 0) DESC_ARGS="";; 1) [ -n "$DESC_ARGS" ] || { echo "DESC=1 needs DESC_ARGS, e.g. --bird-desc /root/d/bird_train_desc (see header)"; exit 2; }
  [ "$DESC_ARGS" = none ] && DESC_ARGS="";; *) echo "bad DESC $DESC (0|1)"; exit 2;; esac; REPO=$R/luthor; O=$REPO/runs/r1r2/$role; RD=$O/rr
V=$R/venv6/bin/python
mkdir -p "$L" "$O"
[ "$DRY" = 1 ] && mkdir -p "$REPO"
cd "$REPO"
export HF_HOME=$R/hf PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false \
  PATH=$R/venv6/bin:$PATH
NP=$(nproc 2>/dev/null || sysctl -n hw.ncpu); W=$(( NP > 4 ? NP - 2 : 2 ))
DEVALL_FLAGS=${DEVALL_FLAGS---all-questions --questions $D/dev.json}
RARE_PLANS=${RARE_PLANS:-$D/rare1}; RARE_MIN=${RARE_MIN:-1000}   # qwrite build writes <data>/rare1/plans.jsonl + cards/
RARE_BUILD=${RARE_BUILD:-"$V scripts/qwrite.py build --data $D"}
QW_MODELS=${QW_MODELS-Qwen/Qwen2.5-14B-Instruct}                  # qwrite's writer; its checker is OmniSQL-7B
PRICE_H=${PRICE_H:-1.35}; EVAL_EST_H=${EVAL_EST_H:-2.0}
STAGE=boot
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$ROLE] $*"; }
fail() { log "FATAL in stage $STAGE: $*"; echo "${ROLE}_FAIL"; echo "$STAGE: $*" > "$L/r1r2.failed"; exit 1; }
trap 'rc=$?; [ $rc = 0 ] || [ -e "$L/r1r2.failed" ] || echo "$STAGE: exit $rc" > "$L/r1r2.failed"' EXIT
# x NAME cmd...: run cmd with ALL output in $L/NAME.log (no pipe can hide its exit status). DRY=1 only echoes.
x() {
  local name=$1; shift
  if [ "$DRY" = 1 ]; then log "DRY: $* > $L/$name.log 2>&1"; return 0; fi
  log "run $name: $*"; local rc=0
  "$@" > "$L/$name.log" 2>&1 || rc=$?
  [ $rc = 0 ] && log "ok $name" || { log "$name exited $rc; tail:"; tail -n 25 "$L/$name.log"; }
  return $rc
}

# ---- 0. boot check + setup + preflight -------------------------------------------------------------------------------
if [ "$DRY" = 1 ]; then log "DRY: nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits (must be 0)"
else
  g=$(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits) || fail "nvidia-smi failed"
  log "GPUs: $g"
  [ "$(echo "$g" | wc -l)" -ge 1 ] || fail "no GPU"
  if echo "$g" | awk -F', *' '$3 + 0 != 0 { bad = 1 } END { exit !bad }'; then
    nvidia-smi; fail "GPU memory already in use at boot (another tenant or leftover process): abort, delete this box"
  fi
fi
STAGE=setup
if [ -e "$L/setup.ok" ]; then log "setup already done"
else
  if [ "$DRY" = 1 ]; then log "DRY: EXPECT_GPUS=1 MODELS='seeklhy/OmniSQL-7B ${QW_MODELS:-}' bash scripts/vast/setup6.sh > $L/setup.log"
  else
    EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B ${QW_MODELS:-}" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 \
      || { tail -n 30 "$L/setup.log"; fail "setup6.sh exited non-zero"; }
    grep -qa '^SETUP6_FAIL' "$L/setup.log" && fail "SETUP6_FAIL in setup.log"
    grep -qa '^DONE_SETUP6' "$L/setup.log" || fail "setup.log has no DONE_SETUP6"
  fi
  touch "$L/setup.ok"; echo DONE_SETUP
fi
STAGE=preflight
need_flag() {  # script subcmd flag
  [ "$DRY" = 1 ] && { log "DRY: check $1 ${2:-} --help lists $3"; return 0; }
  # shellcheck disable=SC2086
  $V "$1" ${2:-} --help 2>&1 | grep -q -- "$3" || fail "$1 ${2:-} has no $3 (model agent's change not pushed?)"
}
need_flag scripts/train_sq2.py "" --cand-order
need_flag scripts/train_sq2.py "" --value-side
need_flag scripts/train_sq2.py "" --max-steps
[ -n "${SEED:-}" ] && need_flag scripts/train_sq2.py "" --seed
[ -n "${GRAMMAR:-}" ] && need_flag scripts/train_sq2.py "" --grammar
[ -n "${DROP_DBS:-}" ] && need_flag scripts/train_sq2.py "" --drop-dbs
for f in ${TRAIN_EXTRA:-}; do case "$f" in --*) need_flag scripts/train_sq2.py "" "$f";; esac; done
for f in $DESC_ARGS; do case "$f" in --*)
  need_flag scripts/train_sq2.py "" "$f"; for sub in gen train score; do need_flag scripts/rerank_sq2.py $sub "$f"; done;; esac; done
[ -n "${WIDE_K:-}" ] && need_flag scripts/rerank_sq2.py gen --distinct
for f in $DEVALL_FLAGS; do case "$f" in --*) need_flag scripts/rerank_sq2.py gen "$f";; esac; done
VOCAB=${VOCAB:-$S/templates.json}
for p in "$BIRD_PLANS" "$SPIDER_PLANS" "$SYNSQL_PLANS" "$S/dbs" "$S/rrdbs" "$BIRD_DEV_PLANS" "$VOCAB" \
         "$S/dev_dbs_bird.json" "$S/bird_dev_dbs.json" "$D/dev_databases" "$D/dev.json"; do
  [ "$DRY" = 1 ] || [ -e "$p" ] || fail "missing pushed data: $p"
done
for f in ${TRAIN_EXTRA:-} $DESC_ARGS; do case "$f" in /*) p=${f%%:*}; [ "$DRY" = 1 ] || [ -e "$p" ] || fail "missing pushed data: $p";; esac; done
if [ "${GRAMMAR:-v3}" = v4 ]; then   # a v4 run must not silently train on the v3 plan dirs
  for p in "$BIRD_PLANS" "$SPIDER_PLANS" "$SYNSQL_PLANS"; do case "$p" in *plans_v4*) ;; *) fail "GRAMMAR=v4 with v3 plan dir $p";; esac; done
fi
log "plans: $BIRD_PLANS $SPIDER_PLANS $SYNSQL_PLANS rare=$RARE_PLANS bird_dev=$BIRD_DEV_PLANS vocab=$VOCAB grammar=$LUTHOR_GRAMMAR desc=$DESC ${DESC_ARGS:+($DESC_ARGS)}"
echo DONE_PREFLIGHT

# ---- 1. rare-skeleton data -------------------------------------------------------------------------------------------
STAGE=rare
RARE_ARG=""
rare_skip() { log "RARE_SKIPPED: $* -- TRAINING WITHOUT RARE DATA"; echo "none: $*" > "$L/rare.decided"; }
if [ -e "$L/rare.decided" ]; then log "rare decided earlier: $(cat "$L/rare.decided")"
elif [ "${SKIP_RARE:-0}" = 1 ]; then rare_skip "SKIP_RARE=1"
else
  if [ ! -e "$L/rare_shard.done" ] && [ ! -e "$L/rare_shard.failed" ]; then
    # shellcheck disable=SC2086
    if [ "$DRY" != 1 ] && [ ! -f scripts/qwrite.py ]; then log "scripts/qwrite.py not pushed"; touch "$L/rare_shard.failed"
    elif x qwrite_write timeout "${RARE_MAX_MIN:-180}m" $V scripts/qwrite.py write --data $D --shard $SHARD ${QW_WRITE_ARGS:-} \
      && x qwrite_check timeout 60m $V scripts/qwrite.py check --data $D --shard $SHARD ${QW_CHECK_ARGS:-}; then
      touch "$L/rare_shard.done"; echo DONE_RARE_SHARD
    else log "RARE SHARD $SHARD FAILED (see qwrite_*.log)"; touch "$L/rare_shard.failed"; echo RARE_SHARD_FAIL
    fi
  fi
  # wait for the Mac to merge both shards into data/rare1 here and write rare.go
  [ "$DRY" = 1 ] && [ ! -e "$L/rare.go" ] && echo "${DRY_RARE:-none}" > "$L/rare.go"
  t0=$(date +%s)
  while [ ! -e "$L/rare.go" ] && [ $(( $(date +%s) - t0 )) -lt $(( ${RARE_WAIT_MIN:-120} * 60 )) ]; do sleep 30; done
  if [ ! -e "$L/rare.go" ]; then rare_skip "no rare.go from the Mac within ${RARE_WAIT_MIN:-120} min"
  elif [ "$(cat "$L/rare.go")" != use ]; then rare_skip "Mac says: $(cat "$L/rare.go")"
  elif ! x rare_build bash -c "$RARE_BUILD"; then rare_skip "build failed (rare_build.log)"
  else
    rows=0; [ -s "$RARE_PLANS/plans.jsonl" ] && rows=$(wc -l < "$RARE_PLANS/plans.jsonl" | tr -d ' ')
    [ "$DRY" = 1 ] && rows=${DRY_RARE_ROWS:-0}
    if [ "$rows" -lt "$RARE_MIN" ]; then rare_skip "only $rows rows in $RARE_PLANS/plans.jsonl (< $RARE_MIN)"
    else
      sha=$( (sha256sum "$RARE_PLANS/plans.jsonl" 2>/dev/null || echo dry) | cut -c1-16)
      echo "use $rows rows sha256:$sha" > "$L/rare.decided"; log "RARE_USED: $rows rows, sha256 $sha (R1 and R2 must match)"
    fi
  fi
fi
case "$(cat "$L/rare.decided")" in use*) RARE_ARG=$RARE_PLANS;; esac
echo DONE_RARE

# ---- 2. train --------------------------------------------------------------------------------------------------------
STAGE=train
project() {  # background: PROJECTION lines after ~50 and ~500 steps, from train_sq2's "step i/T ... X.XXs/step" lines
  local f=$L/train_$role.log seen=0 line st tot sps
  for mark in 50 500; do
    [ "$seen" -ge "$mark" ] && continue
    while :; do
      [ -e "$O/train.ok" ] && return 0
      line=$(grep -aE '^step [0-9]+/[0-9]+ .*s/step' "$f" 2>/dev/null | tail -1 || true)
      st=$(echo "$line" | sed -nE 's#^step ([0-9]+)/.*#\1#p'); tot=$(echo "$line" | sed -nE 's#^step [0-9]+/([0-9]+).*#\1#p')
      sps=$(echo "$line" | sed -nE 's#.* ([0-9.]+)s/step.*#\1#p')
      if [ -n "$st" ] && [ "$st" -ge "$mark" ] && [ "$st" -gt "$seen" ] && [ -n "$sps" ]; then seen=$st; break; fi
      sleep "${PROJ_EVERY:-60}"
    done
    python3 - "$ROLE" "$st" "$tot" "$sps" "$PRICE_H" "$EVAL_EST_H" "$L/projection_$role.txt" <<'EOF'
import sys, time, datetime as dt
role, st, tot, sps, price, ev, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
train_h = (tot - st) * sps / 3600 * 1.04          # +4%: dev evals every 1,000 steps and the gate
fin = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=train_h + ev)
msg = (f"PROJECTION {role}: step {st}/{tot} at {sps:.2f} s/step -> training ends in {train_h:.1f} h, "
       f"eval +{ev:.1f} h, finish ~{fin:%m-%d %H:%M} UTC; remaining cost ~${(train_h + ev) * price:.0f} at ${price:.2f}/h")
print(msg, flush=True); open(out, "a").write(msg + "\n")
EOF
  done
}
if [ -e "$O/train.ok" ]; then log "training already done"
else
  PROJ_PID=""
  if [ "$DRY" = 1 ]; then printf 'step 25/%s ep 0 loss 9.9 3.40s/step\nstep 50/%s ep 0 loss 9.1 3.21s/step\nstep 500/%s ep 0 loss 3.0 3.15s/step\n' \
    "$STEPS" "$STEPS" "$STEPS" > "$L/train_$role.log"; PROJ_EVERY=1 project; else project & PROJ_PID=$!; fi
  # sys1_full's command (runs/sys1/logs_c/sys1_full.log) minus --hours 4.0 --fit-hours, plus the R1/R2 changes
  # shellcheck disable=SC2086
  x "train_$role" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $V scripts/train_sq2.py \
    --plans "$BIRD_PLANS" "$SPIDER_PLANS" "$SYNSQL_PLANS" $RARE_ARG --db-roots "$S/dbs" "$D/dev_databases" \
    --base seeklhy/OmniSQL-7B --subfolder "" --vocab "$VOCAB" --dev-dbs "$S/dev_dbs_bird.json" --out "$O" \
    --eval-every 1000 --budget-chars 9000 --gate "$S/bird_dev" --gate-total 1534 --col-mentions --repeat bird_train=3 \
    --max-steps "$STEPS" --cand-order v2 --value-side ${SEED:+--seed "$SEED"} ${GRAMMAR:+--grammar "$GRAMMAR"} \
    ${DROP_DBS:+--drop-dbs "$DROP_DBS"} ${TRAIN_EXTRA:-} $DESC_ARGS \
    || { [ -n "$PROJ_PID" ] && kill "$PROJ_PID" 2>/dev/null; fail "train_sq2.py crashed (train_$role.log)"; }
  [ -n "$PROJ_PID" ] && { kill "$PROJ_PID" 2>/dev/null || true; }
  [ "$DRY" = 1 ] || [ -d "$O/best_lora" ] || fail "no best_lora after training"
  touch "$O/train.ok"; echo DONE_TRAIN
fi
[ "$DRY" = 1 ] || grep -aE '^done in' "$L/train_$role.log" || true

# ---- 3. eval: candidates, validator, scored BIRD dev (round 2's rr stage) ---------------------------------------------
STAGE="eval"
mkdir -p "$RD"
G=("$V" scripts/rerank_sq2.py gen --run "$O" --k 8 --workers "$W" $DESC_ARGS)
[ -s "$RD/cands_train.jsonl" ] || x "rrgen_train_$role" "${G[@]}" --plans "$BIRD_PLANS" --db-roots "$S/rrdbs" \
  --dev-dbs "$S/dev_dbs_bird.json" --split train --max-q 9000 --out "$RD/cands_train.jsonl" || fail "gen train"
[ -s "$RD/cands_dev.jsonl" ] || x "rrgen_dev_$role" "${G[@]}" --plans "$BIRD_PLANS" --db-roots "$S/dbs" \
  --dev-dbs "$S/dev_dbs_bird.json" --split dev --out "$RD/cands_dev.jsonl" || fail "gen held-out dev"
# shellcheck disable=SC2086
[ -s "$RD/cands_bird_dev.jsonl" ] || x "rrgen_bird_dev_$role" "${G[@]}" --plans "$BIRD_DEV_PLANS" --db-roots "$D/dev_databases" \
  --dev-dbs "$S/bird_dev_dbs.json" --split dev $DEVALL_FLAGS --out "$RD/cands_bird_dev.jsonl" || fail "gen BIRD dev"
echo DONE_RRGEN
[ -s "$RD/rerank_head.pt" ] || x "rrtrain_$role" "$V" scripts/rerank_sq2.py train --cards "$BIRD_PLANS" "$BIRD_DEV_PLANS" \
  --train-cands "$RD/cands_train.jsonl" --dev-cands "$RD/cands_dev.jsonl" --out "$RD" --base seeklhy/OmniSQL-7B \
  --subfolder "" --hours 1.0 --epochs 1 $DESC_ARGS || fail "validator training"
[ -s "$RD/bird_dev_scored.jsonl" ] || x "rrscore_$role" "$V" scripts/rerank_sq2.py score --rr "$RD" --cards "$BIRD_DEV_PLANS" \
  --dev-cands "$RD/cands_bird_dev.jsonl" --out "$RD/bird_dev_scored.jsonl" --base seeklhy/OmniSQL-7B --subfolder "" \
  $DESC_ARGS || fail "validator scoring"
echo DONE_EVAL
if [ -n "${WIDE_K:-}" ]; then   # wide lists for the later R1 u R2 u <this run> pool (phase1_w_box.sh's gen flags)
  STAGE=wide; mkdir -p "$RD/wide"
  GW=("$V" scripts/rerank_sq2.py gen --run "$O" --k "$WIDE_K" --distinct --workers "$W" $DESC_ARGS)
  [ -s "$RD/wide/cands_dev_k$WIDE_K.jsonl" ] || x "wide_dev_$role" "${GW[@]}" --plans "$BIRD_PLANS" --db-roots "$S/dbs" \
    --dev-dbs "$S/dev_dbs_bird.json" --split dev --out "$RD/wide/cands_dev_k$WIDE_K.jsonl" || fail "wide gen held-out"
  # shellcheck disable=SC2086
  [ -s "$RD/wide/cands_bird_dev_k$WIDE_K.jsonl" ] || x "wide_bird_dev_$role" "${GW[@]}" --plans "$BIRD_DEV_PLANS" \
    --db-roots "$D/dev_databases" --dev-dbs "$S/bird_dev_dbs.json" --split dev $DEVALL_FLAGS \
    --out "$RD/wide/cands_bird_dev_k$WIDE_K.jsonl" || fail "wide gen BIRD dev"
  rm -f "$L/wide_bird_dev_$role.log"   # BIRD dev blind
  echo DONE_WIDE
fi

# ---- 4. summary ------------------------------------------------------------------------------------------------------
STAGE=summary
python3 - "$ROLE" "$N" "$STEPS" "$L" "$O" "$role" "$DEVALL_FLAGS" "$LUTHOR_GRAMMAR" "$DESC $DESC_ARGS" "${WIDE_K:-}" <<'EOF' || fail "summary"
import json, re, sys, glob, os
role, n, steps, L, O, r, devall, grammar, desc, wide_k = sys.argv[1:11]
def lines(name, pat):
    p = os.path.join(L, name)
    return [l.strip() for l in open(p, errors="replace") if re.match(pat, l)] if os.path.exists(p) else []
def rd(p):
    try: return open(p).read().strip()
    except OSError: return None
s = {"role": role, "N": int(n), "max_steps": int(steps), "dev_all_flags": devall,
     "rare": rd(f"{L}/rare.decided"), "projection": lines(f"projection_{r}.txt", "PROJECTION"),
     "train": lines(f"train_{r}.log", r"^done in"), "best": json.loads(rd(f"{O}/best.json") or "null"),
     "gen": {k: lines(f"rrgen_{k}_{r}.log", "^GEN") for k in ("train", "dev", "bird_dev")},
     "validator": lines(f"rrtrain_{r}.log", r"^(\[|RERANK|SCORE)"), "score": lines(f"rrscore_{r}.log", "^SCORE"),
     "scored_file": f"runs/r1r2/{r}/rr/bird_dev_scored.jsonl", "grammar": grammar, "desc": desc.strip(),
     "wide_k": wide_k or None, "wide_gen": lines(f"wide_dev_{r}.log", "^GEN") if wide_k else []}
try: s["args"] = json.load(open(f"{O}/args.json"))
except OSError: pass
json.dump(s, open(f"{O}/summary.json", "w"), indent=1)
print(json.dumps(s, indent=1))
EOF
cp "$O/summary.json" "$L/summary_$role.json" 2>/dev/null || true
echo "$ROLE $(date -u +%FT%TZ)" > "$L/r1r2.done"; STAGE="done"
log "ALL DONE: results in $O, marker $L/r1r2.done"; echo DONE_R1R2
