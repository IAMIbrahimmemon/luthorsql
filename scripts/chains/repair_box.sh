#!/bin/bash
# System-1 repair stage (runs/phase1/repair/PREREG.md; docs/paper_notes/research/repair_stage_2026-10-10.md), ON the box:
#   setsid nohup bash /workspace/luthor/scripts/chains/repair_box.sh > /workspace/logs/repair.log 2>&1 < /dev/null &
# (started by scripts/chains/run_repair_box.sh on the Mac, which pushes everything below first.)
#
# System 1 only: ONE grammar generator, the R1 checkpoint (OmniSQL-7B + slot LoRA, grammar v3; phase 3 / v4 was a
# KILL, so no v4 and no R2 here), and ONE validator per arm. Stages, each skipped when its output exists (re-launch
# resumes):
#   1 k64      rerank_sq2.py gen --k 64 --distinct on the held-out split (the heldout_v2 universe: the same 1,000
#              questions as R1's k=32 wide list = DECISION + RESERVED + retail_world)      -> $O/k64_gen.jsonl
#   2 k64pool  k64_gen + anything of R1's own k=32 pool it lacks (src=k32; only when lazy k-best fell short of 32 and
#              samples filled in, or a question the k=64 run dropped)                     -> $O/k64_pool.jsonl
#   3 repair   repair_gen.py gen on k64_pool (constrained re-decoding with the same generator: F1-F4 execution-feedback
#              bans, E1-E3 evidence forcing; banned -1e4 / forced +30; lazy k-best per variant), on DECISION + RESERVED
#              ids only, --blind                                                           -> $O/repair_extra.jsonl
#   4 merge    k64_pool + repair_extra (src=repair)                                        -> $O/k64_repair_pool.jsonl
#   5 score    each pool by both validators: VALHARD ($RP/rr_valhard) and the R1 wide validator ($RP/rr_r1w)
#              -> $O/k64_repair_pool_by_{valhard,r1w}.jsonl, then (SCORE_K64=1) $O/k64_pool_by_{valhard,r1w}.jsonl
# Ends with REPAIR_DONE (REPAIR_FAIL <stage> on any error). No BIRD dev, no sealed set, nothing trained.
# Blind: every pool holds RESERVED rows, so no label-derived number is printed here: the GEN summary line of the k64
# log and the SCORE lines of the score logs are deleted, repair runs with --blind. Scored on the Mac, DECISION only
# (scripts/analysis/repair_eval.py).
#
# Box layout (run_repair_box.sh):
#   /workspace/luthor/{src,scripts,pyproject.toml}  /workspace/luthor/runs/r1r2/r1/{best_lora,best_heads.pt,args.json,vocab.json}
#   /root/d/sys1/{bird_train,dbs,dev_dbs_bird.json,heldout_v2}
#   /root/d/repair/r1/{k32_pool.jsonl,rr_valhard/,rr_r1w/}
# Env: SCORE_K64 (1), REPAIR_ARGS (extra repair_gen.py gen flags), K (64), DRY=1 (echo the commands only; with
#   REMOTE_ROOT / REMOTE_D / PY pointing at local paths it runs anywhere).
set -uo pipefail
R=${REMOTE_ROOT:-/workspace}; D=${REMOTE_D:-/root/d}; S=$D/sys1; RP=$D/repair/r1; L=$R/logs; O=$R/repair
V=${PY:-$R/venv6/bin/python}; REPO=${REPO:-$R/luthor}; RUN=runs/r1r2/r1
DRY=${DRY:-0}; SCORE_K64=${SCORE_K64:-1}; K=${K:-64}
K32=$RP/k32_pool.jsonl; DEC=$S/heldout_v2/decision.json; RES=$S/heldout_v2/reserved.json
W=$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8) - 2 )); [ "$W" -lt 2 ] && W=2
mkdir -p "$L" "$O"; cd "$REPO" || { echo "REPAIR_FAIL no repo $REPO"; exit 1; }
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH=$(dirname "$V"):$PATH
log() { echo "$(date -u '+%m-%d %H:%M:%S') [repair] $*"; }
die() { log "REPAIR_FAIL $*"; exit 1; }
x() {  # x NAME cmd...: output in $L/NAME.log; DRY=1 only echoes. No label-derived line is echoed.
  local n=$1; shift
  if [ "$DRY" = 1 ]; then log "DRY: $* > $L/$n.log"; return 0; fi
  log "run $n"; "$@" > "$L/$n.log" 2>&1; local rc=$?
  sed -i.bak -E '/^(GEN |SCORE |RERANK DEV |greedy|oracle)/d' "$L/$n.log" 2>/dev/null; rm -f "$L/$n.log.bak"
  [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 20 "$L/$n.log" | grep -avE 'GATE|^(GEN|SCORE) ' | sed 's/Traceback/TB/'; }
  return $rc
}
log "R1 generator $RUN (grammar v3), k=$K, pools -> $O, score k64-only pool: $SCORE_K64"
if [ "$DRY" != 1 ]; then
  g=$(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits) || die "nvidia-smi failed"
  log "GPUs: $g"
  if echo "$g" | awk -F', *' '$3 + 0 != 0 { bad = 1 } END { exit !bad }'; then die "GPU memory already in use at boot"; fi
  for f in $RUN/best_lora/adapter_model.safetensors $RUN/best_heads.pt $RUN/args.json $RUN/vocab.json "$K32" "$DEC" "$RES" \
           $S/dev_dbs_bird.json $RP/rr_valhard/rerank_head.pt $RP/rr_valhard/rerank_lora/rerank/adapter_model.safetensors \
           $RP/rr_r1w/rerank_head.pt $RP/rr_r1w/rerank_lora/rerank/adapter_model.safetensors; do
    [ -s "$f" ] || die "missing $f"
  done
  [ -d $S/bird_train/cards ] || die "missing $S/bird_train/cards"
  [ "$(find $S/dbs -name '*.sqlite' | wc -l)" -ge 9 ] || die "held-out sqlite files missing under $S/dbs"
  # the R1 checkpoint is grammar v3 (args.json has no grammar key); refuse anything else
  python3 -c "import json,sys; g=json.load(open('$RUN/args.json')).get('grammar'); sys.exit(0 if g in (None,'','None','v3') else 1)" \
    || die "$RUN/args.json is not a grammar-v3 run"
  if [ ! -e "$L/setup.ok" ]; then
    EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/'; die setup; }
    grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
    touch "$L/setup.ok"
  fi
  log DONE_SETUP
  # preflight: the flags this chain passes (fails in seconds, not after an hour)
  "$V" scripts/repair/repair_gen.py gen --help 2>&1 | grep -q -- "--blind" || die "repair_gen.py has no --blind (code not pushed?)"
  "$V" scripts/rerank_sq2.py --help 2>&1 | grep -q -- "--distinct" || die "rerank_sq2.py has no --distinct"
  log "k32 pool: $(wc -l < "$K32") questions"
fi

# 1 k=64 lists (CPU-bound decode + execution; GPU encoder per chunk of 8 questions)
[ -s $O/k64_gen.jsonl ] || x k64_gen "$V" scripts/rerank_sq2.py gen --run $RUN --k "$K" --distinct --workers "$W" \
  --plans $S/bird_train --db-roots $S/dbs --dev-dbs $S/dev_dbs_bird.json --split dev --out $O/k64_gen.jsonl || die k64_gen
# 2 + R1's own k=32 candidates the k=64 list lacks (src=k32); a question the k=64 run dropped (a gold that timed out)
#   is kept whole from the k=32 pool, so every arm covers the same questions (counts only are printed)
if [ ! -s $O/k64_pool.jsonl ]; then
  if [ "$DRY" != 1 ]; then
    "$V" - "$O/k64_gen.jsonl" "$K32" <<'PY'
import json, sys
a = {json.loads(l)["id"] for l in open(sys.argv[1])}
b = {json.loads(l)["id"] for l in open(sys.argv[2])}
print(f"k64 {len(a)} questions, k32 {len(b)}, only-k64 {len(a - b)}, only-k32 {len(b - a)} (kept from k32)")
PY
  fi
  x k64_pool "$V" scripts/repair/repair_gen.py merge --pool $O/k64_gen.jsonl --extra "$K32" --src k32 --add-missing \
    --out $O/k64_pool.jsonl || die k64_pool
fi
# 3 repair (same generator, constrained re-decoding), DECISION + RESERVED only, blind
# shellcheck disable=SC2086
[ -s $O/repair_extra.jsonl ] || x repair_gen "$V" scripts/repair/repair_gen.py gen --grammar v3 --run $RUN \
  --pool $O/k64_pool.jsonl --plans $S/bird_train --db-roots $S/dbs --ids "$DEC" "$RES" --blind --workers "$W" \
  --out $O/repair_extra.jsonl ${REPAIR_ARGS:-} || die repair_gen
[ "$DRY" = 1 ] || grep -aE '^(repair:|REPAIR )' "$L/repair_gen.log" | tail -2
# 4 merge
[ -s $O/k64_repair_pool.jsonl ] || x repair_merge "$V" scripts/repair/repair_gen.py merge --pool $O/k64_pool.jsonl \
  --extra $O/repair_extra.jsonl --src repair --out $O/k64_repair_pool.jsonl || die repair_merge
# 5 score: primary pool first, then the k=64-only pool (secondary)
pools=(k64_repair_pool); [ "$SCORE_K64" = 1 ] && pools+=(k64_pool)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for p in "${pools[@]}"; do
  for v in valhard r1w; do
    out=$O/${p}_by_$v.jsonl
    [ -s "$out" ] || x score_${p}_$v "$V" scripts/rerank_sq2.py score --rr $RP/rr_$v --cards $S/bird_train \
      --dev-cands $O/$p.jsonl --out "$out" --base seeklhy/OmniSQL-7B --subfolder "" --lora-r 16 || die score_${p}_$v
  done
done
[ "$DRY" = 1 ] || log "outputs: $(cd $O && ls -1 ./*.jsonl | tr '\n' ' ')"
log REPAIR_DONE
