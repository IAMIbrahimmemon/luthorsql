#!/bin/bash
# Re-convert every dataset (after a grammar/compiler change) and rebuild the mix.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -s data/raw/synsql/data.json ] || curl -sL https://huggingface.co/datasets/seeklhy/SynSQL-2.5M/resolve/main/data.json \
  | .venv/bin/python scripts/trim_synsql.py > data/raw/synsql/data.json 2> data/synsql_stream.log
rm -rf data/plans/synsql_train data/plans/bird_train data/raw/bird_train
scripts/prepare_all.sh
.venv/bin/python scripts/fetch_bird_train.py --out data/plans --keep-dbs 3 > data/bird_train.log 2>&1
tail -1 data/bird_train.log
rm -f data/raw/synsql/data.json
.venv/bin/python scripts/build_mix.py --out data/mix
