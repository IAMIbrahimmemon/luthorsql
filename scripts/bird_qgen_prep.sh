#!/bin/bash
# After the BIRD train stream finishes: sample plans, build batches, split DeepSeek / Claude.
cd "$(dirname "$0")/.."
until grep -q '"plans_verified"' reports/coverage_v2/bird_train_keepall.log; do sleep 30; done
V2=/Volumes/MacSSD/AI/datasets/text-to-sql/plans_v2
mkdir -p data/qgen/bird
.venv/bin/python scripts/sample_plans.py --cards $V2/bird_train/cards --db-root data/raw/bird_train/train_databases \
  --shapes $V2/bird_train/plans.jsonl --per-db 175 --out data/qgen/bird/plans.jsonl > reports/qgen_bird_sample.log 2>&1
.venv/bin/python scripts/qgen.py batches --plans data/qgen/bird/plans.jsonl --cards $V2/bird_train/cards --out data/qgen/bird/all --size 25 >> reports/qgen_bird_sample.log 2>&1
mkdir -p data/qgen/bird/ds data/qgen/bird/claude
for f in data/qgen/bird/all/writer_*.json; do n=$(basename $f .json | cut -d_ -f2); if [ $((10#$n % 5)) -eq 4 ]; then mv $f data/qgen/bird/claude/; else mv $f data/qgen/bird/ds/; fi; done
cp data/qgen/bird/all/plans_with_ids.jsonl data/qgen/bird/ds/; cp data/qgen/bird/all/plans_with_ids.jsonl data/qgen/bird/claude/
echo "BIRD_PREP_DONE $(ls data/qgen/bird/ds/writer_*.json | wc -l) ds / $(ls data/qgen/bird/claude/writer_*.json | wc -l) claude" >> reports/qgen_bird_sample.log
