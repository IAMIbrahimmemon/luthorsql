#!/bin/bash
# VAL3 data v3 (docs/paper_notes/research/validator/04_training_data.md #1, 00_synthesis.md #1), ON the box:
#   setsid nohup bash /workspace/luthor/scripts/chains/val3_lists_box.sh > /workspace/logs/val3_lists.log 2>&1 < /dev/null &
# (started by scripts/chains/run_val3_lists_box.sh on the Mac, which pushes everything below first.)
#
# System 1, generator R1 only (runs/r1r2/r1: OmniSQL-7B + slot LoRA, grammar v3; nothing is trained here). For each NEW
# question set staged by scripts/val3/sources.py it writes R1's k=32 distinct lists, executed and labelled against
# each set's own gold on the box (rerank_sq2.py gen; these labels are TRAINING labels), and R2's greedy (k=1) on the
# same questions (for the label-noise mask in scripts/val3/build.py):
#   bird_rr   BIRD-train questions on the 47 BIRD-train DBs of data/sys1/rrdbs (in-sample a + out-of-sample b)
#   synsql    unseen SynSQL-2.5M questions (verified plans; DBs pushed as data/val3/dbs/synsql)
#   bird_big  BIRD-train questions on 10 large BIRD-train DBs, only when the public BIRD train archive downloads
#             (scripts/val3/fetch_big_dbs.py, system python3, started first, runs during setup); else skipped
# Outputs -> $O = /workspace/val3: cands_<set>_r1.jsonl (k=32), cands_<set>_r2.jsonl (k=1), counts.txt.
# Ends VAL3_LISTS_DONE (VAL3_LISTS_FAIL <stage> when bird_rr or synsql fails; bird_big and R2 are optional and only
# warn). No held-out DB, no BIRD dev, no sealed set is on this box.
#
# Box layout (run_val3_lists_box.sh):
#   /workspace/luthor/{src,scripts,pyproject.toml}
#   /workspace/luthor/runs/r1r2/{r1,r2}/{best_lora,best_heads.pt,args.json,vocab.json}
#   /root/d/sys1/rrdbs/**.sqlite   /root/d/val3/{q/{bird_rr,synsql,bird_big},dbs/synsql,big_dbs.json}
# Env: BIG (1: fetch + generate bird_big; 0 skips), BIG_WAIT_MIN (90: give up on the download after this many minutes
#   from the start), BIG_MAX_MIN (100: time limit of the bird_big k=32 gen), R2_GREEDY (1), K (32), SECONDS_GEN (3.0),
#   DRY=1 (echo the commands only; with REMOTE_ROOT / REMOTE_D / PY pointing at local paths it runs anywhere).
# Preflight before setup uses the SYSTEM python3 only (venv6 does not exist before setup6.sh).
set -uo pipefail
R=${REMOTE_ROOT:-/workspace}; D=${REMOTE_D:-/root/d}; Q=$D/val3/q; L=$R/logs; O=$R/val3
V=${PY:-$R/venv6/bin/python}; REPO=${REPO:-$R/luthor}; R1=runs/r1r2/r1; R2RUN=runs/r1r2/r2
DRY=${DRY:-0}; BIG=${BIG:-1}; BIG_WAIT_MIN=${BIG_WAIT_MIN:-90}; BIG_MAX_MIN=${BIG_MAX_MIN:-100}; DO_R2=${R2_GREEDY:-1}
K=${K:-32}; SEC=${SECONDS_GEN:-3.0}; BIGDB=$D/val3/bigdbs
W=$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8) - 2 )); [ "$W" -lt 2 ] && W=2
T0=$(date +%s)
mkdir -p "$L" "$O"; cd "$REPO" || { echo "VAL3_LISTS_FAIL no repo $REPO"; exit 1; }
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH=$(dirname "$V"):$PATH
log() { echo "$(date -u '+%m-%d %H:%M:%S') [val3] $*"; }
die() { log "VAL3_LISTS_FAIL $*"; exit 1; }
x() {  # x NAME cmd...: output in $L/NAME.log; DRY=1 only echoes
  local n=$1; shift
  if [ "$DRY" = 1 ]; then log "DRY: $* > $L/$n.log"; return 0; fi
  log "run $n"; "$@" > "$L/$n.log" 2>&1; local rc=$?
  [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 20 "$L/$n.log" | grep -av GATE | sed 's/Traceback/TB/; s/_FAIL/_fail/g'; }
  return $rc
}
gen() {  # gen RUN K SET OUT DBROOTS... : one rerank_sq2 gen over a staged plan dir (uncovered rows included)
  local run=$1 k=$2 set=$3 out=$4; shift 4
  local extra=(); [ "$k" -gt 1 ] && extra=(--distinct)
  "$V" scripts/rerank_sq2.py gen --run "$run" --k "$k" "${extra[@]}" --workers "$W" --seconds "$SEC" \
    --plans "$Q/$set" --db-roots "$@" --dev-dbs "$Q/$set/dbs.json" --split dev --all-questions --out "$out"
}
roots_of() { case "$1" in bird_rr) echo "$D/sys1/rrdbs";; synsql) echo "$D/val3/dbs/synsql";; bird_big) echo "$BIGDB";; esac; }
nq() { [ -s "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

log "R1 $R1 (grammar v3) k=$K distinct, R2 greedy $DO_R2, big DBs $BIG (wait ${BIG_WAIT_MIN}m, limit ${BIG_MAX_MIN}m), workers $W"
# ---- preflight, system python3 only (no venv yet) --------------------------------------------------------------
if [ "$DRY" != 1 ]; then
  g=$(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits) || die "nvidia-smi failed"
  log "GPUs: $g"
  if echo "$g" | awk -F', *' '$3 + 0 != 0 { bad = 1 } END { exit !bad }'; then die "GPU memory already in use at boot"; fi
  command -v python3 >/dev/null || die "no system python3"
  need=($R1/best_lora/adapter_model.safetensors $R1/best_heads.pt $R1/args.json $R1/vocab.json
        $Q/bird_rr/plans.jsonl $Q/bird_rr/dbs.json $Q/synsql/plans.jsonl $Q/synsql/dbs.json)
  [ "$DO_R2" = 1 ] && need+=($R2RUN/best_lora/adapter_model.safetensors $R2RUN/best_heads.pt $R2RUN/args.json $R2RUN/vocab.json)
  [ "$BIG" = 1 ] && need+=($Q/bird_big/plans.jsonl $Q/bird_big/dbs.json $D/val3/big_dbs.json)
  for f in "${need[@]}"; do [ -s "$f" ] || die "missing $f"; done
  for r in $R1 $([ "$DO_R2" = 1 ] && echo $R2RUN); do  # grammar-v3 runs only (args.json without a grammar key = v3)
    python3 -c "import json,sys; g=json.load(open('$r/args.json')).get('grammar'); sys.exit(0 if g in (None,'','None','v3') else 1)" \
      || die "$r/args.json is not a grammar-v3 run"
  done
  # every DB a staged set names must be on the box (gen silently drops questions whose sqlite is missing)
  python3 - "$Q" "$D/sys1/rrdbs" "$D/val3/dbs/synsql" <<'PY' || die "staged DBs missing (see above)"
import json, pathlib, sys
q, roots = pathlib.Path(sys.argv[1]), {"bird_rr": sys.argv[2], "synsql": sys.argv[3]}
bad = 0
for s, root in roots.items():
    have = {p.stem for p in pathlib.Path(root).rglob("*.sqlite")}
    want = json.load(open(q / s / "dbs.json"))
    cards = {p.stem for p in (q / s / "cards").glob("*.json")}
    miss, nocard = sorted(set(want) - have), sorted(set(want) - cards)
    print(f"{s}: {sum(1 for _ in open(q / s / 'plans.jsonl'))} questions, {len(want)} DBs, {len(miss)} sqlite missing, "
          f"{len(nocard)} cards missing")
    bad += len(miss) + len(nocard)
sys.exit(1 if bad else 0)
PY
  df -h "$R" | tail -1
fi
# ---- big DBs: public BIRD train archive, in the background during setup ------------------------------------------
FETCH_PID=""
if [ "$BIG" = 1 ] && [ "$DRY" != 1 ] && [ ! -e "$BIGDB/FETCH_OK" ]; then
  python3 scripts/val3/fetch_big_dbs.py --want "$D/val3/big_dbs.json" --out "$BIGDB" --work "$R/bird_dl" \
    > "$L/fetch_big.log" 2>&1 < /dev/null &
  FETCH_PID=$!
  log "big-DB download started (pid $FETCH_PID, log $L/fetch_big.log)"
fi
# ---- setup -------------------------------------------------------------------------------------------------------
if [ "$DRY" != 1 ]; then
  if [ ! -e "$L/setup.ok" ]; then
    EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/; s/_FAIL/_fail/g'; die setup; }
    grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
    touch "$L/setup.ok"
  fi
  log DONE_SETUP
  # preflight 2 (venv exists now): the flags this chain passes
  h=$("$V" scripts/rerank_sq2.py --help 2>&1)
  for f in --distinct --all-questions --dev-dbs --seconds; do echo "$h" | grep -q -- "$f" || die "rerank_sq2.py has no $f (code not pushed?)"; done
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ---- 1 R1 k=32 on the sets whose DBs are here ------------------------------------------------------------------------
for s in bird_rr synsql; do
  out=$O/cands_${s}_r1.jsonl
  # shellcheck disable=SC2046
  [ -s "$out" ] || x gen_${s}_r1 gen $R1 "$K" $s "$out" $(roots_of $s) || die gen_${s}_r1
  [ "$DRY" = 1 ] || { log "$s: $(nq "$out") lists"; grep -aE '^GEN ' "$L/gen_${s}_r1.log" | tail -1 | cut -c1-400; }
done
# ---- 2 bird_big: wait for the download (bounded), then its k=32 gen in the background (time-limited) ----------------
BIG_GEN_PID=""; HAVE_BIG=0
if [ "$BIG" = 1 ] && [ "$DRY" != 1 ]; then
  while [ -n "$FETCH_PID" ] && kill -0 "$FETCH_PID" 2>/dev/null; do
    el=$(( ($(date +%s) - T0) / 60 ))
    if [ "$el" -ge "$BIG_WAIT_MIN" ]; then log "big-DB download still running after ${el} min: giving up on bird_big"; kill "$FETCH_PID" 2>/dev/null; break; fi
    log "waiting for the big-DB download (${el} min): $(tail -n 1 "$L/fetch_big.log" 2>/dev/null | cut -c1-160)"
    sleep 120
  done
  if [ -e "$BIGDB/FETCH_OK" ]; then
    HAVE_BIG=1; out=$O/cands_bird_big_r1.jsonl
    if [ ! -s "$out" ]; then
      log "run gen_bird_big_r1 (background, limit ${BIG_MAX_MIN} min)"
      ( timeout "${BIG_MAX_MIN}m" bash -c "$(declare -f gen); V='$V' W='$W' SEC='$SEC' Q='$Q'; gen $R1 $K bird_big '$out.tmp' '$BIGDB'" \
          > "$L/gen_bird_big_r1.log" 2>&1 && mv -f "$out.tmp" "$out" ) &
      BIG_GEN_PID=$!
    fi
  else
    why="no FETCH_OK"; [ -s "$BIGDB/FETCH_FAIL" ] && why=$(head -c 200 "$BIGDB/FETCH_FAIL" | sed 's/_FAIL/_fail/g')
    log "big DBs not available ($why): bird_big skipped"
  fi
fi
# ---- 3 R2 greedy (k=1) on the same questions (optional: only the label-noise mask uses it) ------------------------------
if [ "$DO_R2" = 1 ]; then
  for s in bird_rr synsql; do
    out=$O/cands_${s}_r2.jsonl
    # shellcheck disable=SC2046
    [ -s "$out" ] || x gen_${s}_r2 gen $R2RUN 1 $s "$out" $(roots_of $s) || log "WARN gen_${s}_r2 failed: no R2 greedy for $s"
  done
fi
if [ -n "$BIG_GEN_PID" ]; then
  while kill -0 "$BIG_GEN_PID" 2>/dev/null; do
    log "bird_big k=32 gen running: $(grep -aE 'decoded|executing' "$L/gen_bird_big_r1.log" 2>/dev/null | tail -1 | cut -c1-120)"
    sleep 120
  done
  wait "$BIG_GEN_PID"; rc=$?
  [ -s $O/cands_bird_big_r1.jsonl ] && log "bird_big: $(nq $O/cands_bird_big_r1.jsonl) lists" \
    || log "WARN gen_bird_big_r1 exit $rc (124 = time limit): bird_big skipped"
fi
if [ "$HAVE_BIG" = 1 ] && [ "$DO_R2" = 1 ] && [ -s $O/cands_bird_big_r1.jsonl ] && [ ! -s $O/cands_bird_big_r2.jsonl ]; then
  x gen_bird_big_r2 timeout "${BIG_MAX_MIN}m" bash -c "$(declare -f gen); V='$V' W='$W' SEC='$SEC' Q='$Q'; gen $R2RUN 1 bird_big '$O/cands_bird_big_r2.jsonl' '$BIGDB'" \
    || log "WARN gen_bird_big_r2 failed: no R2 greedy for bird_big"
fi
if [ "$DRY" != 1 ]; then
  for f in "$O"/cands_*.jsonl; do echo "$(basename "$f") $(nq "$f")"; done | tee "$O/counts.txt"
  cp -f "$L"/gen_*.log "$L/fetch_big.log" "$O/" 2>/dev/null
  rm -rf "$R/bird_dl"
fi
log "VAL3_LISTS_DONE in $(( ($(date +%s) - T0) / 60 )) min"
