#!/bin/bash
# Arctic-Text2SQL-R1-7B off the shelf on BIRD dev (corrected) + sealed sets. Pulls results, deletes the instance.
cd ~/Documents/GitHub/luthor-general; ID=${1:?id}; SP=${SP:-/tmp}
S() { bash scripts/vast/ssh.sh $ID "$@" 2>/dev/null | grep -av "Welcome\|Have fun"; }
pull() { bash -c 'source scripts/vast/ssh.sh '$ID'; mkdir -p runs/arc; "${RS[@]}" "$H:/workspace/luthor/runs/r6/" runs/arc/ ; "${RS[@]}" "$H:/workspace/logs/" runs/arc/logs/' >/dev/null 2>&1; }
end() { pull; CONFIRM=yes bash scripts/vast/destroy.sh $ID | grep -v Welcome; echo "$1"; exit 0; }
spent() { python3 scripts/shade/sf.py info $ID 2>/dev/null | python3 -c "import json,sys;print('spent', json.load(sys.stdin).get('cost_estimate'))"; }
until S 'command -v rsync && command -v python3' | grep -q python3; do REFRESH=1 bash scripts/vast/ssh.sh $ID true >/dev/null 2>&1; sleep 15; done
echo "$(date -u +%H:%M) booted"; S 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
bash scripts/vast/push6.sh $ID data/eval/arc 2>&1 | grep -E "to send|PUSHED|FAIL|leak|LEAK" | tee /dev/stderr | grep -q "^PUSHED6" || end PUSH_FAILED
S 'test -s /workspace/logs/setup.log' || S '(MODELS="Snowflake/Arctic-Text2SQL-R1-7B" setsid nohup bash /workspace/luthor/scripts/vast/setup6.sh > /workspace/logs/setup.log 2>&1 < /dev/null &)'
until S 'grep -qaE "^(DONE_SETUP6|SETUP6_FAIL)" /workspace/logs/setup.log && echo y' | grep -q y; do sleep 30; done
S 'grep -qa ^SETUP6_FAIL /workspace/logs/setup.log && echo F' | grep -q F && { S 'tail -20 /workspace/logs/setup.log'; end SETUP_FAILED; }
echo "$(date -u +%H:%M) setup done $(spent)"
S "(STAGE=arctic setsid nohup bash /workspace/luthor/scripts/vast/run6.sh > /workspace/logs/arctic.log 2>&1 < /dev/null &)"
while true; do
  st=$(S "grep -aE '^(DONE_ARCTIC|ARCTIC_FAIL|Traceback|STYLE|GENERATED)|Processed prompts' /workspace/logs/arctic.log | tail -2 | cut -c1-200")
  echo "$(date -u +%H:%M) $(spent) | $(echo "$st" | tail -1)"
  echo "$st" | grep -q "^DONE_ARCTIC$" && end ALL_DONE
  echo "$st" | grep -qE "^ARCTIC_FAIL|Traceback" && { S 'tail -30 /workspace/logs/arctic.log'; end ARCTIC_FAILED; }
  sleep 180
done
