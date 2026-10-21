#!/bin/bash
# BIRD package check, ON the box (one A100 80GB, CUDA 12.2 driver): run the submission package exactly as BIRD would
# (run_bird.sh: fresh venv from requirement.txt, then predict.py) on the BIRD dev questions with the SQL field removed.
#   setsid nohup bash /workspace/luthor/scripts/chains/bird_check_box.sh > /workspace/logs/birdcheck.log 2>&1 < /dev/null &
# Layout (scripts/chains/run_bird_check.sh pushes it):
#   /root/d/bird_submit/{luthorsql_submission/,weights/,dev_blind/{questions.json,dev_tables.json}}  /root/d/dev_databases/
# Output: /workspace/birdcheck/{predict_dev.json,predict_dev.json.cands.jsonl,env.txt,requirement_box.txt,run.log}
# BLIND: no gold SQL is on this box (refused if found); the Mac compares picks with the scored dev run by SQL only.
# Ends DONE_BIRDCHECK (BIRDCHECK_FAIL <why> on errors).
set -uo pipefail
D=${REMOTE_D:-/root/d}; B=$D/bird_submit; P=$B/luthorsql_submission; O=/workspace/birdcheck
mkdir -p "$O" /workspace/logs
log() { echo "$(date -u '+%m-%d %H:%M:%S') [birdcheck] $*"; }
die() { log "$*"; echo "BIRDCHECK_FAIL"; exit 1; }
for f in "$P/run_bird.sh" "$P/requirement.txt" "$B/dev_blind/questions.json" "$B/dev_blind/dev_tables.json"; do
  [ -s "$f" ] || die "missing $f"
done
[ "$(find "$D/dev_databases" -name '*.sqlite' | wc -l | tr -d ' ')" -eq 11 ] || die "expected 11 dev sqlite files"
grep -q '"SQL"' "$B/dev_blind/questions.json" && die "questions.json carries SQL: refusing"
[ -z "$(find "$D" \( -name 'dev.json' -o -name '*.sql' -o -name '*gold*' \) | head -1)" ] || die "gold-like file on the box: refusing"
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/CUDA \1/p' | head -1; nproc; } > "$O/env.txt"
log "env: $(tr '\n' ' ' < "$O/env.txt")"
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0
t=$(date +%s)
cd "$P" || die "no package dir"
RUN=(bash run_bird.sh "$B/dev_blind/questions.json" "$B/dev_blind/dev_tables.json" "$D/dev_databases" "$O/predict_dev.json" "$B/weights")
if [ "${RESUME_TEST:-0}" = 1 ]; then
  # BIRD asked for incremental saving + resume: stop the run after RESUME_AFTER databases are saved, rerun the same
  # command, and require that it resumes from them (predict.py logs "resume: N of M databases already done").
  N=${RESUME_AFTER:-3}
  setsid "${RUN[@]}" > "$O/run_part1.log" 2>&1 &
  pid=$!
  until [ "$(ls "$O/predict_dev.json.work/dbs/"*.done.jsonl 2>/dev/null | wc -l)" -ge "$N" ] || ! kill -0 $pid 2>/dev/null; do sleep 10; done
  kill -0 $pid 2>/dev/null || die "run ended before $N databases were saved (see run_part1.log)"
  ndone=$(ls "$O/predict_dev.json.work/dbs/"*.done.jsonl | wc -l | tr -d ' ')
  npred=$(python3 -c "import json; print(len(json.load(open('$O/predict_dev.json'))))" 2>/dev/null || echo 0)
  kill -TERM -- -$pid 2>/dev/null; sleep 5; kill -KILL -- -$pid 2>/dev/null; wait $pid 2>/dev/null
  log "RESUME_TEST: stopped after $ndone databases saved; predict_dev.json already held $npred predictions"
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
log "total $(( $(date +%s) - t ))s (setup + model download + 1,534 questions)"
echo "DONE_BIRDCHECK"
