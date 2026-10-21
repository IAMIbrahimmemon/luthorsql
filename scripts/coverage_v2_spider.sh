#!/bin/bash
# Re-run the Spider-family conversions after the require + PK-edge fixes (round-2 inputs).
cd "$(dirname "$0")/.."
EV=data/eval_plans_v2; TR=/Volumes/MacSSD/AI/datasets/text-to-sql/plans_v2; mkdir -p reports/coverage_v2
P=".venv/bin/python scripts/prepare_plans.py --workers 5"
run() { $P --format "$1" --root "$2" --split "$3" --out "$4" "${@:5}" > "reports/coverage_v2/$1_$3.log" 2>&1; echo "$1 $3: $(grep -m1 '"coverage"' reports/coverage_v2/$1_$3.log)" >> reports/coverage_v2/summary2.txt; }
: > reports/coverage_v2/summary2.txt
run spider data/eval_raw/spider_data dev $EV
run spider data/raw/spider_data train $TR
run spider_syn data/raw train $TR
run spider data/eval_raw/spider_data test $EV
run spider_syn data/eval_raw dev $EV
run bird data/bench/dev_20240627 dev $EV
echo DONE >> reports/coverage_v2/summary2.txt
