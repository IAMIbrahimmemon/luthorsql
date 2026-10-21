#!/bin/bash
# After the Spider top-up finishes, DeepSeek writes the BIRD batches the Claude agents left (2 sessions).
cd "$(dirname "$0")/.."
while pgrep -f "qgen_ollama.py topup" >/dev/null; do sleep 60; done
.venv/bin/python -u scripts/qgen_ollama.py write --batches data/qgen/bird/claude --workers 2 > reports/qgen_bird_claude_by_ds.log 2>&1
echo CHAIN_DONE >> reports/qgen_bird_claude_by_ds.log
