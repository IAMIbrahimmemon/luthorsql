#!/bin/bash
# v4guard (runs/phase1/v4guard/PREREG.md; docs/paper_notes/research/grammar_v4_fix_2026-10-11.md), ON the box:
#   setsid nohup bash /workspace/luthor/scripts/chains/v4guard_box.sh > /workspace/logs/v4guard.log 2>&1 < /dev/null &
# (started by scripts/chains/run_v4guard_box.sh on the Mac, which pushes everything below first.)
#
# Re-decode the EXISTING phase-3 v4desc generator (grammar v4, --bird-desc; no retraining) with the join-route guard
# LUTHOR_JOIN_GUARD=direct (src/luthor/sq/plan.py, off by default) and score with v4desc's own validator and with
# VALHARD-v4desc. Same flags as the v4desc chain (runs/phase1/v4desc/logs/chain.log rrgen_dev_r1 / wide_dev_r1), only
# the guard differs. Stages, each skipped when its output exists (a re-launch resumes):
#   0 k8_off_gen   (REPRO_GEN=1) k=8, guard OFF: decode-determinism check vs v4desc's own cands_dev.jsonl
#   1 k8_guard     k=8, guard direct, held-out split (dev_dbs_bird.json = DECISION + RESERVED ids)  -> $O/k8_guard.jsonl
#   2 score        k8_guard by v4desc's own validator                                 -> $O/k8_guard_by_own.jsonl
#                  (REPRO_SCORE=1) v4desc's unguarded k=8 list by the same `score` path -> $O/k8_off_by_own.jsonl
#   3 k32_guard    k=32 --distinct, guard direct                                       -> $O/k32_guard.jsonl
#   4 score k32    by the own validator and by VALHARD-v4desc   -> $O/k32_guard_by_own.jsonl, $O/k32_guard_by_valhard.jsonl
# Ends with V4GUARD_DONE (V4GUARD_FAIL <stage> on any error). No BIRD dev, no sealed set, no GATE, nothing trained.
# Blind: every list holds RESERVED rows, so no label-derived line is printed: the GEN / SCORE summary lines are deleted
# from the stage logs. Scored on the Mac, DECISION only (scripts/analysis/v4guard_eval.py).
#
# Box layout (run_v4guard_box.sh):
#   /workspace/luthor/{src,scripts,pyproject.toml}
#   /root/d/sys1/{dbs,dev_dbs_bird.json}  /root/d/plans_v4/bird_train  /root/d/bird_train_desc
#   /root/d/v4guard/{gen/{best_lora,best_heads.pt,args.json,vocab.json}, rr_own/, rr_valhard/, k8_off.jsonl}
# Env: REPRO_GEN (1), REPRO_SCORE (1), DRY=1 (echo the commands only; with REMOTE_ROOT / REMOTE_D / PY pointing at local
#   paths it runs anywhere). Preflight before setup uses SYSTEM python3 only (the venv does not exist yet).
set -uo pipefail
R=${REMOTE_ROOT:-/workspace}; D=${REMOTE_D:-/root/d}; S=$D/sys1; G=$D/v4guard; L=$R/logs; O=$R/v4guard
V=${PY:-$R/venv6/bin/python}; REPO=${REPO:-$R/luthor}
DRY=${DRY:-0}; REPRO_GEN=${REPRO_GEN:-1}; REPRO_SCORE=${REPRO_SCORE:-1}
RUN=$G/gen; BP=$D/plans_v4/bird_train; DESC=$D/bird_train_desc; DEVDBS=$S/dev_dbs_bird.json
W=$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8) - 2 )); [ "$W" -lt 2 ] && W=2; [ "$W" -gt 18 ] && W=18
mkdir -p "$L" "$O"; cd "$REPO" || { echo "V4GUARD_FAIL no repo $REPO"; exit 1; }
export LUTHOR_GRAMMAR=v4 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH=$(dirname "$V"):$PATH
log() { echo "$(date -u '+%m-%d %H:%M:%S') [v4guard] $*"; }
die() { log "V4GUARD_FAIL $*"; exit 1; }
x() {  # x NAME cmd...: output in $L/NAME.log; DRY=1 only echoes. No label-derived line is kept or echoed.
  local n=$1; shift
  if [ "$DRY" = 1 ]; then log "DRY: $* > $L/$n.log"; return 0; fi
  log "run $n"; "$@" > "$L/$n.log" 2>&1; local rc=$?
  sed -i.bak -E '/^(GEN |SCORE |RERANK DEV |VAL |greedy|oracle|gold SQL failed)/d' "$L/$n.log" 2>/dev/null; rm -f "$L/$n.log.bak"
  [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 20 "$L/$n.log" | grep -avE 'GATE|^(GEN|SCORE) ' | sed 's/Traceback/TB/'; }
  return $rc
}
GENF=(--workers "$W" --bird-desc "$DESC" --plans "$BP" --db-roots "$S/dbs" --dev-dbs "$DEVDBS" --split dev)
SCOREF=(--cards "$BP" --base seeklhy/OmniSQL-7B --subfolder "" --lora-r 16 --bird-desc "$DESC")
log "v4desc generator $RUN (grammar v4, --bird-desc), guard direct, workers $W, outputs -> $O, repro gen $REPRO_GEN score $REPRO_SCORE"

if [ "$DRY" != 1 ]; then
  # ---- preflight BEFORE setup: system python3 + coreutils only (no venv yet) ----
  g=$(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits) || die "nvidia-smi failed"
  log "GPUs: $g"
  if echo "$g" | awk -F', *' '$3 + 0 != 0 { bad = 1 } END { exit !bad }'; then die "GPU memory already in use at boot"; fi
  for f in $RUN/best_lora/adapter_model.safetensors $RUN/best_lora/adapter_config.json $RUN/best_heads.pt $RUN/args.json \
           $RUN/vocab.json $G/k8_off.jsonl "$DEVDBS" \
           $G/rr_own/rerank_head.pt $G/rr_own/rerank_lora/rerank/adapter_model.safetensors \
           $G/rr_valhard/rerank_head.pt $G/rr_valhard/rerank_lora/rerank/adapter_model.safetensors; do
    [ -s "$f" ] || die "missing $f"
  done
  [ -d $BP/cards ] && [ -s $BP/plans.jsonl ] || die "missing $BP/{cards,plans.jsonl}"
  [ -d "$DESC" ] || die "missing $DESC"
  [ "$(find $S/dbs -name '*.sqlite' | wc -l)" -ge 9 ] || die "held-out sqlite files missing under $S/dbs"
  command -v python3 >/dev/null || die "no system python3 for the preflight"
  # the v4desc checkpoint is grammar v4 with --bird-desc /root/d/bird_train_desc; refuse anything else
  python3 -c "import json,sys; a=json.load(open('$RUN/args.json')); sys.exit(0 if a.get('grammar')=='v4' and a.get('bird_desc')=='$DESC' else 1)" \
    || die "$RUN/args.json is not the v4desc run (grammar v4, bird_desc $DESC)"
  log "k8_off (v4desc's own k=8 list): $(wc -l < $G/k8_off.jsonl) questions"
  if [ ! -e "$L/setup.ok" ]; then
    EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/'; die setup; }
    grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
    touch "$L/setup.ok"
  fi
  log DONE_SETUP
  # ---- preflight AFTER setup (venv): the pushed code has the guard and the flags this chain passes ----
  [ -x "$V" ] || die "no venv python $V after setup"
  LUTHOR_JOIN_GUARD=direct "$V" -c "import sys; sys.path.insert(0, 'src'); import luthor.sq.plan as p; sys.exit(0 if p.JOIN_GUARD == 'direct' and p.GRAMMAR_V4 else 1)" \
    || die "luthor.sq.plan has no LUTHOR_JOIN_GUARD=direct under grammar v4 (code not pushed?)"
  "$V" scripts/rerank_sq2.py --help 2>&1 | grep -q -- "--distinct" || die "rerank_sq2.py has no --distinct"
  "$V" scripts/rerank_sq2.py --help 2>&1 | grep -q -- "--bird-desc" || die "rerank_sq2.py has no --bird-desc"
  log DONE_PREFLIGHT
fi

# 0 decode-determinism check: guard OFF must reproduce v4desc's own k=8 list (SQL strings only; no labels printed)
if [ "$REPRO_GEN" = 1 ] && [ ! -s $O/k8_off_gen.jsonl ]; then
  x k8_off_gen env LUTHOR_JOIN_GUARD=off "$V" scripts/rerank_sq2.py gen --run $RUN --k 8 "${GENF[@]}" \
    --out $O/k8_off_gen.jsonl || die k8_off_gen
fi
if [ "$REPRO_GEN" = 1 ] && [ "$DRY" != 1 ]; then
  python3 - "$O/k8_off_gen.jsonl" "$G/k8_off.jsonl" <<'PY'
import json, sys
def sqls(p):
    return {r["id"]: [c.get("sql") for c in r["cands"]] for r in map(json.loads, open(p))}
a, b = sqls(sys.argv[1]), sqls(sys.argv[2])
same = sum(1 for i in a if i in b and a[i] == b[i])
print(f"repro gen (guard off vs v4desc k8): {len(a)} vs {len(b)} questions, identical SQL lists {same}, "
      f"only-new {len(set(a) - set(b))}, only-old {len(set(b) - set(a))}")
PY
fi
# 1 guarded k=8 list (v4desc's rrgen_dev_r1 flags + the guard)
[ -s $O/k8_guard.jsonl ] || x k8_guard env LUTHOR_JOIN_GUARD=direct "$V" scripts/rerank_sq2.py gen --run $RUN --k 8 \
  "${GENF[@]}" --out $O/k8_guard.jsonl || die k8_guard
[ "$DRY" = 1 ] || python3 - "$O/k8_guard.jsonl" "$G/k8_off.jsonl" <<'PY'
import json, sys
def sqls(p):
    return {r["id"]: [c.get("sql") for c in r["cands"]] for r in map(json.loads, open(p))}
a, b = sqls(sys.argv[1]), sqls(sys.argv[2])
ch = sum(1 for i in a if i in b and a[i] != b[i])
print(f"guarded k8: {len(a)} questions, lists changed vs unguarded {ch} (SQL strings only)")
PY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 2 score k=8 by the own validator (the guarded list; then the unguarded list through the same path, as a check)
[ -s $O/k8_guard_by_own.jsonl ] || x score_k8_guard_own env LUTHOR_JOIN_GUARD=direct "$V" scripts/rerank_sq2.py score \
  --rr $G/rr_own --dev-cands $O/k8_guard.jsonl --out $O/k8_guard_by_own.jsonl "${SCOREF[@]}" || die score_k8_guard_own
if [ "$REPRO_SCORE" = 1 ]; then
  [ -s $O/k8_off_by_own.jsonl ] || x score_k8_off_own "$V" scripts/rerank_sq2.py score --rr $G/rr_own \
    --dev-cands $G/k8_off.jsonl --out $O/k8_off_by_own.jsonl "${SCOREF[@]}" || die score_k8_off_own
fi
# 3 guarded k=32 --distinct list (v4desc's wide_dev_r1 flags + the guard)
[ -s $O/k32_guard.jsonl ] || x k32_guard env LUTHOR_JOIN_GUARD=direct "$V" scripts/rerank_sq2.py gen --run $RUN --k 32 \
  --distinct "${GENF[@]}" --out $O/k32_guard.jsonl || die k32_guard
# 4 score k=32 by the own validator and VALHARD-v4desc
for v in own valhard; do
  out=$O/k32_guard_by_$v.jsonl
  [ -s "$out" ] || x score_k32_guard_$v env LUTHOR_JOIN_GUARD=direct "$V" scripts/rerank_sq2.py score --rr $G/rr_$v \
    --dev-cands $O/k32_guard.jsonl --out "$out" "${SCOREF[@]}" || die score_k32_guard_$v
done
[ "$DRY" = 1 ] || log "outputs: $(cd $O && ls -1 ./*.jsonl | tr '\n' ' ')"
log V4GUARD_DONE
