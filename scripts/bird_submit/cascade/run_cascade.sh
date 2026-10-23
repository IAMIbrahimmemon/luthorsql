#!/bin/bash
# LuthorSQL cascade on BIRD: System 1 (7B, one GPU) answers every question; questions where it is unsure go to
# Claude Opus 5.5 (API) with System 1's candidates.
#   ANTHROPIC_API_KEY=... bash run_cascade.sh <test.json> <test_tables.json> <test_databases/> <predict_test.json> [weights_dir]
# Stage 1 (GPU code, run_bird.sh) writes <predict_test.json>.s1.json and its .cands.jsonl.
# Stage 2 (API code, cascade/cascade.py, its own venv ./venv_api) writes <predict_test.json>.
# Both stages save progress and resume: rerun the same command after an interruption.
set -euo pipefail
[ $# -ge 4 ] || { sed -n 2,7p "$0"; exit 2; }
H=$(cd "$(dirname "$0")" && pwd)
OUT=$4; S1="$OUT.s1.json"; WORK="$OUT.cascade_work"
# ---- stage 1: GPU (no API) ----
s1_done() {
  python3 - "$1" "$S1" <<'EOF'
import json, sys
from pathlib import Path
n = len(json.load(open(sys.argv[1])))
c, p = Path(sys.argv[2] + ".cands.jsonl"), Path(sys.argv[2])
ok = c.exists() and p.exists() and sum(1 for _ in open(c)) == n and len(json.load(open(p))) == n
log = Path(sys.argv[2] + ".log")
if ok and log.exists():
    ok = "PREDICT_INCOMPLETE" not in "".join(log.read_text().splitlines(True)[-50:])
sys.exit(0 if ok else 1)
EOF
}
if s1_done "$1"; then
  echo "stage 1 complete: $S1 (delete it to rerun System 1)"
else
  bash "$H/run_bird.sh" "$1" "$2" "$3" "$S1" ${5:+"$5"}
  s1_done "$1" || { echo "stage 1 incomplete; rerun the same command"; exit 1; }
fi
# ---- stage 2: API (no GPU) ----
DRY=${CASCADE_DRY:+--dry-run}   # CASCADE_DRY=1: pipeline test with fake answers, no API calls
[ -n "${ANTHROPIC_API_KEY:-}" ] || [ -n "$DRY" ] || { echo "set ANTHROPIC_API_KEY for stage 2"; exit 1; }
if [ ! -x "$H/venv_api/bin/python" ]; then
  command -v uv >/dev/null || python3 -m pip install -q --user uv
  export PATH=$HOME/.local/bin:$PATH
  uv venv -q --python 3.12 "$H/venv_api"
  VIRTUAL_ENV=$H/venv_api uv pip install -q -r "$H/cascade/requirement_api.txt"
fi
"$H/venv_api/bin/python" "$H/cascade/cascade.py" all --cands "$S1.cands.jsonl" --db-root "$3" --work "$WORK" \
  --out "$OUT" --jobs "${CASCADE_JOBS:-4}" $DRY 2>&1 | tee -a "$OUT.cascade.log"
