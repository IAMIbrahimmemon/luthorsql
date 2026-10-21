#!/bin/bash
# Convert every downloaded public dataset into execution-verified plans (data/plans/<format>_<split>/).
cd "$(dirname "$0")/.."
P=(.venv/bin/python scripts/prepare_plans.py --out data/plans --workers 6)
run() { "${P[@]}" --format "$1" --root "$2" --split "$3" "${@:4}" > "data/plans_$1_$3.log" 2>&1; echo "$1 $3: $(grep -m1 '"coverage"' data/plans_$1_$3.log)"; }
run bird data/bench/dev_20240627 dev
for s in train dev test; do run spider data/raw/spider_data $s; done
for s in train dev; do run spider_syn data/raw $s; done
for s in train dev test; do run wikisql data/raw/wikisql $s; done
[ -f data/raw/synsql/data.json ] && [ -s data/raw/synsql/data.json ] && run synsql data/raw/synsql train --unsup-sample 0.05
# BIRD train: scripts/fetch_bird_train.py (streams and converts per database)
