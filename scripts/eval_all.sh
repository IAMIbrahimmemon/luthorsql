#!/bin/bash
# Round-1 sealed evaluation, one set at a time (MLX memory). Logs in reports/.
cd "$(dirname "$0")/.."
E=".venv/bin/python scripts/eval_sealed.py --run runs/sq4b_v1"; mkdir -p reports/noabstain; cd . 
echo "start $(date)" > reports/noabstain/eval_all.status
$E --sets bird_dev   2>&1 | grep -v -i warn > reports/noabstain/sealed_bird_dev.log;   echo "bird_dev done $(date)" >> reports/noabstain/eval_all.status
$E --sets spider_dev 2>&1 | grep -v -i warn > reports/noabstain/sealed_spider_dev.log; echo "spider_dev done $(date)" >> reports/noabstain/eval_all.status
$E --sets spider_test spider_syn_dev wikisql_test --limit 500 2>&1 | grep -v -i warn > reports/noabstain/sealed_sampled.log
echo "ALL_DONE $(date)" >> reports/noabstain/eval_all.status
