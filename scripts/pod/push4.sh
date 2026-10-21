#!/bin/bash
# Round 4: everything push3.sh sends, plus the round-3 pilot slot model (runs/sq3_pilot: LoRA + heads only)
# and the BIRD train SQLite files used for execution checks (5 dev databases + the small train databases
# listed in $BIRD_LIST). Public data only. Code-only refresh: CODE_ONLY=1 bash scripts/pod/push4.sh
set -e
cd "$(dirname "$0")/../.."
E="ssh -i $HOME/.ssh/runpod_luthor -o StrictHostKeyChecking=accept-new -p ${POD_PORT:?set POD_PORT}"
H="root@${POD_HOST:?set POD_HOST}"
rsync -az --no-owner --no-group -e "$E" --exclude .venv --exclude .git --exclude __pycache__ --exclude 'data/' \
  --exclude models --exclude reports --exclude runs --exclude '.env*' --exclude graphify-out ./ "$H:/workspace/luthor/"
[ -n "$CODE_ONLY" ] && { echo PUSHED_CODE; exit 0; }
POD_HOST=$POD_HOST POD_PORT=$POD_PORT bash scripts/pod/push3.sh
$E "$H" mkdir -p /workspace/luthor/runs/sq3_pilot /workspace/bt/train/train_databases
rsync -az --no-owner --no-group -e "$E" runs/sq3_pilot/best_lora runs/sq3_pilot/best_heads.pt runs/sq3_pilot/args.json \
  "$H:/workspace/luthor/runs/sq3_pilot/"
rsync -a --no-owner --no-group -e "$E" --include '*/' --include '*.sqlite' --exclude '*' \
  $(for d in car_retails cs_semester food_inspection_2 movie_platform donor $(cat "${BIRD_LIST:?}"); do
      printf 'data/raw/bird_train/train_databases/%s ' "$d"; done) \
  "$H:/workspace/bt/train/train_databases/"
echo PUSHED
