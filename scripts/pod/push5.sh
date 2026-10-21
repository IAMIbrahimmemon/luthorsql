#!/bin/bash
# Round 5 session A: everything push4.sh sends, plus the frozen dev list, the training-question file and the
# round-4 writer LoRAs (SFT and RL). Public data only. Code-only refresh: CODE_ONLY=1 bash scripts/pod/push5.sh
set -e
cd "$(dirname "$0")/../.."
E="ssh -i $HOME/.ssh/runpod_luthor -o StrictHostKeyChecking=accept-new -p ${POD_PORT:?set POD_PORT}"
H="root@${POD_HOST:?set POD_HOST}"
BIRD_LIST=${BIRD_LIST:-data/sel/bird_pod_list.txt} POD_HOST=$POD_HOST POD_PORT=$POD_PORT CODE_ONLY=$CODE_ONLY bash scripts/pod/push4.sh
[ -n "$CODE_ONLY" ] && exit 0
$E "$H" mkdir -p /root/d/eval /root/d/sel /workspace/luthor/runs/sqlw_sft /workspace/luthor/runs/sqlw_rl
rsync -az --no-owner --no-group -e "$E" data/eval/dev_frozen.jsonl "$H:/root/d/eval/"
rsync -az --no-owner --no-group -e "$E" data/sel/train_questions.jsonl "$H:/root/d/sel/"
rsync -az --no-owner --no-group -e "$E" runs/sqlw_sft/lora "$H:/workspace/luthor/runs/sqlw_sft/"
rsync -az --no-owner --no-group -e "$E" runs/sqlw_rl/lora "$H:/workspace/luthor/runs/sqlw_rl/"
echo PUSHED5
