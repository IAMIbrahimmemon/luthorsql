#!/bin/bash
# Round 5 session B + final scoring: writer v2 data (data/sqlw5), the sealed question files and their
# databases and cards. Public data only. Sealed files are scored once by the final system (EVAL_LEDGER.md).
set -e
cd "$(dirname "$0")/../.."
E="ssh -i $HOME/.ssh/runpod_luthor -o StrictHostKeyChecking=accept-new -p ${POD_PORT:?set POD_PORT}"
H="root@${POD_HOST:?set POD_HOST}"
P=/Volumes/MacSSD/AI/datasets/text-to-sql/plans
$E "$H" mkdir -p /root/d/sqlw5 /root/d/eval /root/d/sealed/spider_dev /root/d/sealed/spider_test /root/d/sealed/bird_dev /root/d/db_test /root/d/bird_dev
rsync -az --no-owner --no-group -e "$E" data/sqlw5/ "$H:/root/d/sqlw5/"
rsync -az --no-owner --no-group -e "$E" data/eval/sealed_spider_dev.jsonl data/eval/sealed_spider_test.jsonl \
  data/eval/sealed_spider_syn_dev.jsonl data/eval/sealed_bird_dev.jsonl "$H:/root/d/eval/"
for s in spider_dev spider_test bird_dev; do rsync -az --no-owner --no-group -e "$E" $P/$s/cards "$H:/root/d/sealed/$s/"; done
rsync -a --no-owner --no-group -e "$E" --include '*/' --include '*.sqlite' --exclude '*' data/raw/spider_data/test_database "$H:/root/d/db_test/"
rsync -a --no-owner --no-group -e "$E" --include '*/' --include '*.sqlite' --exclude '*' data/bench/dev_20240627/dev_databases "$H:/root/d/bird_dev/"
echo PUSHED5B
