#!/bin/bash
# Step A: re-measure plan coverage with grammar v2 on every dataset.
cd "$(dirname "$0")/.."
EV=data/eval_plans_v2; TR=/Volumes/MacSSD/AI/datasets/text-to-sql/plans_v2; mkdir -p $EV $TR reports/coverage_v2
P=".venv/bin/python scripts/prepare_plans.py --workers 6"
run() { $P --format "$1" --root "$2" --split "$3" --out "$4" "${@:5}" > "reports/coverage_v2/$1_$3.log" 2>&1; echo "$1 $3: $(grep -m1 '"coverage"' reports/coverage_v2/$1_$3.log)" >> reports/coverage_v2/summary.txt; }
: > reports/coverage_v2/summary.txt
run bird data/bench/dev_20240627 dev $EV
run spider data/eval_raw/spider_data dev $EV
run spider data/eval_raw/spider_data test $EV
run spider_syn data/eval_raw dev $EV
run wikisql data/eval_raw/wikisql test $EV
run spider data/raw/spider_data train $TR
run spider_syn data/raw train $TR
run wikisql data/raw/wikisql train $TR
.venv/bin/python scripts/fetch_bird_train.py --out $TR --keep-dbs 3 > reports/coverage_v2/bird_train.log 2>&1
echo "bird train: $(tail -1 reports/coverage_v2/bird_train.log)" >> reports/coverage_v2/summary.txt
curl -sL https://huggingface.co/datasets/seeklhy/SynSQL-2.5M/resolve/main/data.json | .venv/bin/python scripts/trim_synsql.py > data/raw/synsql/data.json 2>/dev/null
run synsql data/raw/synsql train $TR --unsup-sample 0.05
rm -f data/raw/synsql/data.json
echo DONE >> reports/coverage_v2/summary.txt
