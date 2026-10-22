#!/bin/bash
# Fresh unseen-set System 1 run (experiments/fresh_unseen/PREREG.md): frozen package, gold-free.
# RESERVED questions with no SQL field, for gold-free candidates and margins (experiments/cost_confirmation/PREREG.md).
#   setsid nohup bash /workspace/luthor/scripts/chains/heldout_s1_box.sh > /workspace/logs/heldout.log 2>&1 < /dev/null &
# Layout (scripts/chains/run_bird_check.sh pushes it):
#   /root/d/bird_submit/{luthorsql_submission/,weights/,../fresh_blind/blind/{questions.json,tables.json}}  /root/d/heldout_databases/
# Output: /workspace/external/{predict_external.json,predict_external.json.cands.jsonl,env.txt,requirement_box.txt,run.log}
# BLIND: no gold SQL is on this box (refused if found); the Mac scores the candidates against gold after the run.
# Ends DONE_EXTERNAL (EXTERNAL_FAIL <why> on errors).
set -uo pipefail
D=${REMOTE_D:-/root/d}; B=$D/bird_submit; P=$B/luthorsql_submission; O=/workspace/external
mkdir -p "$O" /workspace/logs
log() { echo "$(date -u '+%m-%d %H:%M:%S') [heldout] $*"; }
die() { log "$*"; echo "EXTERNAL_FAIL"; exit 1; }
for f in "$P/run_bird.sh" "$P/requirement.txt" "$B/../fresh_blind/blind/questions.json" "$B/../fresh_blind/blind/tables.json"; do
  [ -s "$f" ] || die "missing $f"
done
[ "$(find "$D/fresh_blind/databases" -name '*.sqlite' | wc -l | tr -d ' ')" -eq ${NDB:-8} ] || die "expected ${NDB:-8} sqlite files"
grep -q '"SQL"' "$B/../fresh_blind/blind/questions.json" && die "questions.json carries SQL: refusing"
[ -z "$(find "$D" \( -name 'dev.json' -o -name '*.sql' -o -name '*gold*' \) | head -1)" ] || die "gold-like file on the box: refusing"
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/CUDA \1/p' | head -1; nproc; } > "$O/env.txt"
log "env: $(tr '\n' ' ' < "$O/env.txt")"
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0
t=$(date +%s)
cd "$P" || die "no package dir"
RUN=(bash run_bird.sh "$B/../fresh_blind/blind/questions.json" "$B/../fresh_blind/blind/tables.json" "$D/fresh_blind/databases" "$O/predict_external.json" "$B/weights")
if [ "${RESUME_TEST:-0}" = 1 ]; then
  # BIRD asked for incremental saving + resume: stop the run after RESUME_AFTER databases are saved, rerun the same
  # command, and require that it resumes from them (predict.py logs "resume: N of M databases already done").
  N=${RESUME_AFTER:-3}
  setsid "${RUN[@]}" > "$O/run_part1.log" 2>&1 &
  pid=$!
  until [ "$(ls "$O/predict_external.json.work/dbs/"*.done.jsonl 2>/dev/null | wc -l)" -ge "$N" ] || ! kill -0 $pid 2>/dev/null; do sleep 10; done
  kill -0 $pid 2>/dev/null || die "run ended before $N databases were saved (see run_part1.log)"
  ndone=$(ls "$O/predict_external.json.work/dbs/"*.done.jsonl | wc -l | tr -d ' ')
  npred=$(python3 -c "import json; print(len(json.load(open('$O/predict_external.json'))))" 2>/dev/null || echo 0)
  kill -TERM -- -$pid 2>/dev/null; sleep 5; kill -KILL -- -$pid 2>/dev/null; wait $pid 2>/dev/null
  log "RESUME_TEST: stopped after $ndone databases saved; predict_external.json already held $npred predictions"
  [ "$npred" -gt 0 ] || die "RESUME_TEST: no incremental predictions were written before the stop"
fi
"${RUN[@]}" > "$O/run.log" 2>&1
rc=$?
if [ "${RESUME_TEST:-0}" = 1 ]; then
  line=$(grep -a "resume:" "$O/run.log" | head -1)
  log "RESUME_TEST rerun: ${line#*INFO }"
  echo "$line" | grep -qE "resume: [1-9][0-9]* of [0-9]+ databases already done" || die "RESUME_TEST: rerun did not resume"
fi
sed -i -E '/^(GEN |SCORE|RERANK DEV |greedy|oracle)/d' "$O/run.log"
VIRTUAL_ENV=$P/venv "$HOME/.local/bin/uv" pip freeze > "$O/requirement_box.txt" 2>/dev/null || "$P/venv/bin/python" -m pip freeze > "$O/requirement_box.txt" 2>/dev/null
tail -n 5 "$O/run.log"
[ $rc = 0 ] && grep -q PREDICT_DONE "$O/run.log" || die "run_bird.sh exit $rc"
log "total $(( $(date +%s) - t ))s (setup + model download + questions)"
echo "DONE_EXTERNAL"
