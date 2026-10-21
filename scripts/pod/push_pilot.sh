#!/bin/bash
# Pilot C: push code + Spider-train v2 plans, the Spider dev gate set and the Spider SQLite files
# (all public). Nothing else leaves the Mac.
set -e
cd "$(dirname "$0")/../.."
E="ssh -i $HOME/.ssh/runpod_luthor -o StrictHostKeyChecking=accept-new -p ${POD_PORT:?set POD_PORT}"
H="root@${POD_HOST:?set POD_HOST}"
TR=/Volumes/MacSSD/AI/datasets/text-to-sql/plans_v2
rsync -az --no-owner --no-group -e "$E" --exclude .venv --exclude .git --exclude __pycache__ --exclude 'data/' \
  --exclude models --exclude reports --exclude runs --exclude '.env*' ./ "$H:/workspace/luthor/"
$E "$H" mkdir -p /root/d/plans /root/d/gate /root/d/db
rsync -az --no-owner --no-group -e "$E" $TR/spider_train $TR/spider_syn_train "$H:/root/d/plans/"
rsync -az --no-owner --no-group -e "$E" data/eval_plans_v2/spider_dev "$H:/root/d/gate/"
rsync -az --no-owner --no-group -e "$E" data/pilot/dev_dbs.json "$H:/root/d/"
rsync -az --no-owner --no-group -e "$E" --include '*/' --include '*.sqlite' --exclude '*' data/raw/spider_data/database "$H:/root/d/db/"
echo PUSHED
