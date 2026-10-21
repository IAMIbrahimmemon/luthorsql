#!/bin/bash
# Second-stage verifier over round 2's generated questions (1 DeepSeek session; the qgen3 pipeline uses 2).
cd "$(dirname "$0")/.."
.venv/bin/python scripts/qgen.py verifies --batches data/qgen/bird/claude --cards /Volumes/MacSSD/AI/datasets/text-to-sql/plans_v2/bird_train/cards
.venv/bin/python scripts/qgen.py verifies --batches data/qgen/bird/ds --cards /Volumes/MacSSD/AI/datasets/text-to-sql/plans_v2/bird_train/cards
.venv/bin/python scripts/qgen.py verifies --batches data/qgen/spider/ds --cards /Volumes/MacSSD/AI/datasets/text-to-sql/plans_v3/spider_train/cards
for d in bird/claude bird/ds spider/claude spider/ds; do
  for pass in 1 2; do .venv/bin/python -u scripts/qgen_ollama.py verify --batches data/qgen/$d --workers 1 >> reports/verify_round2.log 2>&1; done
done
echo VERIFY_R2_DONE >> reports/verify_round2.log
