#!/bin/bash
# BIRD dev look #1 (runs/phase1/devlook/FROZEN.md), ON the box:
#   VARIANT=A|B setsid nohup bash /workspace/luthor/scripts/chains/devlook_box.sh > /workspace/logs/devlook.log 2>&1 < /dev/null &
# (started by scripts/chains/run_devlook_box.sh on the Mac, which pushes everything below first.)
#
# Frozen System 1: ONE generator, R1 (OmniSQL-7B + slot LoRA, grammar v3, runs/r1r2/r1), ONE validator, VALHARD
# (runs/phase1/valhard/valhard/rr). The box is BLIND: it holds no BIRD dev gold query and no label (the pushed files are
# audited by scripts/devlook/prep.py audit; blind_gen.py refuses any row carrying ok / gold_sql / covered / ...), so
# no EX / ok / oracle can be computed or printed here. The Mac picks and executes against gold once
# (scripts/analysis/devlook_score.py).
#   VARIANT=A  pool A = R1's own k=32 distinct BIRD dev list (prep.py pool-a)              -> score -> $O/poolA_by_valhard.jsonl
#   VARIANT=B  1 k64     blind_gen.py gen --k 64 --distinct (R1)                            -> $O/k64_gen.jsonl
#              2 k64pool k64_gen + pool A's candidates it lacks (src=k32)                   -> $O/k64_pool.jsonl
#              3 repair  blind_gen.py repair on k64_pool (repair_gen.py's triggers/variants, unchanged settings)
#                                                                                           -> $O/repair_extra.jsonl
#              4 merge   k64_pool + repair_extra (src=repair)                               -> $O/poolB.jsonl
#              5 score   VALHARD                                                            -> $O/poolB_by_valhard.jsonl
# Each stage is skipped when its output exists (a re-launch resumes). Ends DEVLOOK_DONE (DEVLOOK_FAIL <stage> on error).
#
# Box layout (run_devlook_box.sh):
#   /workspace/luthor/{src,scripts,pyproject.toml}
#   /root/d/devlook/r1/{poolA.jsonl,qs/{cards/,plans.jsonl},rr_valhard/}
#   B only: /workspace/luthor/runs/r1r2/r1/{best_lora,best_heads.pt,args.json,vocab.json}, /root/d/dev_databases/*/*.sqlite
# Env: VARIANT (required), K (64), DRY=1 (echo the commands only; with REMOTE_ROOT / REMOTE_D / PY / REPO pointing at
#   local paths it runs anywhere).
set -uo pipefail
R=${REMOTE_ROOT:-/workspace}; D=${REMOTE_D:-/root/d}; DL=$D/devlook/r1; L=$R/logs; O=$R/devlook
V=${PY:-$R/venv6/bin/python}; REPO=${REPO:-$R/luthor}; RUN=runs/r1r2/r1
DRY=${DRY:-0}; K=${K:-64}; VARIANT=${VARIANT:-}
QS=$DL/qs; POOLA=$DL/poolA.jsonl; RR=$DL/rr_valhard; DBS=$D/dev_databases
W=$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8) - 2 )); [ "$W" -lt 2 ] && W=2
mkdir -p "$L" "$O"; cd "$REPO" || { echo "DEVLOOK_FAIL no repo $REPO"; exit 1; }
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH=$(dirname "$V"):$PATH
log() { echo "$(date -u '+%m-%d %H:%M:%S') [devlook $VARIANT] $*"; }
die() { log "DEVLOOK_FAIL $*"; exit 1; }
case "$VARIANT" in A|B) ;; *) die "VARIANT must be A or B (got '$VARIANT')";; esac
x() {  # x NAME cmd...: output in $L/NAME.log; DRY=1 only echoes. Label-looking lines are deleted (there are no labels
       # on this box, so rerank_sq2 score's SCORE lines are all zeros; they are dropped anyway).
  local n=$1; shift
  if [ "$DRY" = 1 ]; then log "DRY: $* > $L/$n.log"; return 0; fi
  log "run $n"; "$@" > "$L/$n.log" 2>&1; local rc=$?
  sed -i.bak -E '/^(GEN |SCORE|RERANK DEV |greedy|oracle|REPAIR )/d' "$L/$n.log" 2>/dev/null; rm -f "$L/$n.log.bak"
  [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 20 "$L/$n.log" | grep -avE '^(GEN|SCORE)' | sed 's/Traceback/TB/'; }
  return $rc
}
log "frozen System 1: generator R1 ($RUN, grammar v3), validator VALHARD ($RR); variant $VARIANT -> $O"
if [ "$DRY" != 1 ]; then
  g=$(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits) || die "nvidia-smi failed"
  log "GPUs: $g"
  if echo "$g" | awk -F', *' '$3 + 0 != 0 { bad = 1 } END { exit !bad }'; then die "GPU memory already in use at boot"; fi
  need=("$POOLA" "$QS/plans.jsonl" "$RR/rerank_head.pt" "$RR/rerank_lora/rerank/adapter_model.safetensors")
  [ "$VARIANT" = B ] && need+=($RUN/best_lora/adapter_model.safetensors $RUN/best_heads.pt $RUN/args.json $RUN/vocab.json)
  for f in "${need[@]}"; do [ -s "$f" ] || die "missing $f"; done
  [ "$(find "$QS/cards" -name '*.json' | wc -l)" -eq 11 ] || die "expected 11 BIRD dev cards under $QS/cards"
  # blind: no gold / label anywhere under the pushed data (dev.json is never pushed; the audit refuses label fields)
  [ -z "$(find "$D" \( -name 'dev.json' -o -name 'dev_*.json' -o -name '*.sql' -o -name '*gold*' \) 2>/dev/null | head -1)" ] \
    || die "a gold-looking file is on the box under $D: refusing"
  python3 scripts/devlook/prep.py audit "$POOLA" "$QS/plans.jsonl" > "$L/audit.log" 2>&1 || { cat "$L/audit.log"; die audit; }
  if [ "$VARIANT" = B ]; then
    [ "$(find "$DBS" -name '*.sqlite' | wc -l)" -eq 11 ] || die "expected 11 BIRD dev sqlite files under $DBS"
    python3 -c "import json,sys; g=json.load(open('$RUN/args.json')).get('grammar'); sys.exit(0 if g in (None,'','None','v3') else 1)" \
      || die "$RUN/args.json is not a grammar-v3 run"
  fi
  if [ ! -e "$L/setup.ok" ]; then
    EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/'; die setup; }
    grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
    touch "$L/setup.ok"
  fi
  log DONE_SETUP
  "$V" scripts/devlook/blind_gen.py merge --help > /dev/null 2>&1 || die "blind_gen.py does not run (code not pushed?)"
  "$V" scripts/rerank_sq2.py --help 2>&1 | grep -q -- "--dev-cands" || die "rerank_sq2.py has no --dev-cands"
  log "pool A: $(wc -l < "$POOLA") questions"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
score() {  # score POOL OUT
  [ -s "$2" ] || x "score_$(basename "$1" .jsonl)" "$V" scripts/rerank_sq2.py score --rr "$RR" --cards "$QS" \
    --dev-cands "$1" --out "$2" --base seeklhy/OmniSQL-7B --subfolder "" --lora-r 16 || die "score $(basename "$1")"
}
if [ "$VARIANT" = A ]; then
  score "$POOLA" "$O/poolA_by_valhard.jsonl"
  FINAL=$O/poolA_by_valhard.jsonl
else
  # 1 k=64 distinct lists (CPU-bound decode + execution; GPU encoder per chunk of 8 questions)
  [ -s $O/k64_gen.jsonl ] || x k64_gen "$V" scripts/devlook/blind_gen.py gen --run $RUN --k "$K" --distinct \
    --workers "$W" --plans "$QS" --db-roots "$DBS" --out $O/k64_gen.jsonl || die k64_gen
  # 2 + pool A's candidates the k=64 list lacks (src=k32; a question the k=64 run lost is kept whole from pool A)
  [ -s $O/k64_pool.jsonl ] || x k64_pool "$V" scripts/devlook/blind_gen.py merge --pool $O/k64_gen.jsonl \
    --extra "$POOLA" --src k32 --add-missing --out $O/k64_pool.jsonl || die k64_pool
  # 3 repair (same generator, constrained re-decoding), every BIRD dev question that triggers
  [ -s $O/repair_extra.jsonl ] || x repair "$V" scripts/devlook/blind_gen.py repair --run $RUN --pool $O/k64_pool.jsonl \
    --plans "$QS" --db-roots "$DBS" --workers "$W" --out $O/repair_extra.jsonl || die repair
  # 4 merge
  [ -s $O/poolB.jsonl ] || x poolB "$V" scripts/devlook/blind_gen.py merge --pool $O/k64_pool.jsonl \
    --extra $O/repair_extra.jsonl --src repair --out $O/poolB.jsonl || die poolB
  # 5 score
  score "$O/poolB.jsonl" "$O/poolB_by_valhard.jsonl"
  FINAL=$O/poolB_by_valhard.jsonl
fi
if [ "$DRY" != 1 ]; then
  for f in "$L"/*.log; do grep -aE '^(gen blind|repair blind|merged|POOL|AUDIT)' "$f" 2>/dev/null | tail -2; done
  "$V" - "$FINAL" <<'PY' || die "scored file check"
import json, sys
n = nc = nrr = 0
for l in open(sys.argv[1]):
    r = json.loads(l); n += 1
    for c in r["cands"]:
        nc += 1; nrr += "rr" in c
        assert "ok" not in c and "gold_sql" not in r
print(f"scored file: {n} questions, {nc} candidates, {nrr} with a VALHARD score")
PY
  log "outputs: $(cd $O && ls -1 ./*.jsonl | tr '\n' ' ')"
fi
log DEVLOOK_DONE
