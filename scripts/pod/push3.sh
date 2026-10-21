#!/bin/bash
# Round-3 pilot: push code, data/mix3 (public, checked), the dev plan sets (Spider/BIRD train plans; only
# their --dev-dbs rows are used), the dev database list and the Spider SQLite files. BIRD train databases
# are downloaded on the pod from the public BIRD URL (fetch_bird_dbs in run3.sh). Nothing else leaves the Mac.
set -e
cd "$(dirname "$0")/../.."
E="ssh -i $HOME/.ssh/runpod_luthor -o StrictHostKeyChecking=accept-new -p ${POD_PORT:?set POD_PORT}"
H="root@${POD_HOST:?set POD_HOST}"
TR=/Volumes/MacSSD/AI/datasets/text-to-sql/plans_v2
rsync -az --no-owner --no-group -e "$E" --exclude .venv --exclude .git --exclude __pycache__ --exclude 'data/' \
  --exclude models --exclude reports --exclude runs --exclude '.env*' --exclude graphify-out ./ "$H:/workspace/luthor/"
$E "$H" mkdir -p /root/d/plans /root/d/db
rsync -az --no-owner --no-group -e "$E" data/mix3 "$H:/root/d/"
rsync -az --no-owner --no-group -e "$E" $TR/spider_train $TR/bird_train "$H:/root/d/plans/"
rsync -az --no-owner --no-group -e "$E" data/mix2_dev_dbs.json "$H:/root/d/"
rsync -az --no-owner --no-group -e "$E" --include '*/' --include '*.sqlite' --exclude '*' data/raw/spider_data/database "$H:/root/d/db/"
echo PUSHED
