#!/bin/bash
# Round 3 (7B finish): old-checkpoint diagnostics + robustness, new training run, robustness, wide top-32 validator.
# Runs detached on the Mac; destroys the instance at the end, on failure, or when credit < $1.
cd ~/Documents/GitHub/luthor-general; ID=${1:?id}
S() { bash scripts/vast/ssh.sh $ID "$@" 2>/dev/null | grep -av "Welcome\|Have fun"; }
pull() { bash -c 'source scripts/vast/ssh.sh '$ID'; mkdir -p runs/sys1/r7; for r in sys1_full sys1_v4; do "${RS[@]}" --exclude "*_lora" --exclude "*.pt" "$H:/workspace/luthor/runs/r6/$r/" runs/sys1/r7/$r/; done; "${RS[@]}" "$H:/workspace/luthor/runs/r6/sys1_v4/best_lora" "$H:/workspace/luthor/runs/r6/sys1_v4/best_heads.pt" runs/sys1/r7/sys1_v4/; "${RS[@]}" "$H:/workspace/luthor/runs/r6/sys1_v4/rr32w/rerank_lora" "$H:/workspace/luthor/runs/r6/sys1_v4/rr32w/rerank_head.pt" runs/sys1/r7/sys1_v4/rr32w/; "${RS[@]}" "$H:/workspace/luthor/runs/r6/sys1_v4/rr32w_prev/rerank_lora" "$H:/workspace/luthor/runs/r6/sys1_v4/rr32w_prev/rerank_head.pt" "$H:/workspace/luthor/runs/r6/sys1_v4/rr32w_prev/settings.json" runs/sys1/r7/sys1_v4/rr32w_prev/; "${RS[@]}" "$H:/workspace/logs/" runs/sys1/logs_g/' >/dev/null 2>&1; }
credit() {  # Vast: account credit. Shadeform: spent so far (its auto_delete cap guards the money), so never "low"
  if [[ "$ID" == *-* ]]; then python3 scripts/shade/sf.py info $ID 2>/dev/null | python3 -c "import json,sys;print('spent', json.load(sys.stdin).get('cost_estimate'))"
  else vastai show user --raw 2>/dev/null | python3 -c "import json,sys;print(round(json.load(sys.stdin).get('credit',0),2))"; fi; }
end() { pull; CONFIRM=yes bash scripts/vast/destroy.sh $ID | grep -v Welcome; echo "$1"; exit 0; }
low() { [[ "$ID" == *-* ]] && return 1; python3 -c "import sys; sys.exit(0 if float('${1:-99}') < 1.0 else 1)"; }
run() { S "(STAGE=$1 $3 setsid nohup bash /workspace/luthor/scripts/vast/run6.sh > /workspace/logs/$2.log 2>&1 < /dev/null &)"; }
wait_for() {  # $1 log, $2 done, $3 fail
  while true; do
    st=$(S "grep -aE '^($2|$3)\$|^Traceback|^SCORE|^\[|^GEN|^eval step|^step [0-9]+/|^normal|^off|^cap2|^inv|^GATE|^done in' /workspace/logs/$1.log | tail -2")
    c=$(credit); echo "$(date -u +%H:%M) $1 credit $c | $(echo "$st" | tail -1 | cut -c1-200)"
    echo "$st" | grep -q "^$2$" && return 0
    echo "$st" | grep -qE "^$3$|Traceback" && { S "tail -25 /workspace/logs/$1.log"; return 1; }
    low $c && end STOPPED_LOW_CREDIT
    sleep 180
  done
}
until S 'command -v rsync && command -v python3' | grep -q python3; do REFRESH=1 bash scripts/vast/ssh.sh $ID true >/dev/null 2>&1; sleep 15; done
echo "$(date -u +%H:%M) booted"; S 'nvidia-smi --query-gpu=name,memory.used --format=csv,noheader'
until grep -q "^FINAL" data/sys1/autodesc2.log 2>/dev/null; do sleep 30; done; echo "$(date -u +%H:%M) auto-descriptions ready"
bash scripts/vast/push6.sh $ID data/sys1 data/bench/dev_20240627/dev_databases 2>&1 | grep -E "to send|PUSHED|FAIL|leak"
S '(MODELS="seeklhy/OmniSQL-7B" setsid nohup bash /workspace/luthor/scripts/vast/setup6.sh > /workspace/logs/setup.log 2>&1 < /dev/null &)'
until S 'grep -qaE "^(DONE_SETUP6|SETUP6_FAIL)" /workspace/logs/setup.log && echo y' | grep -q y; do sleep 30; done
S 'grep -qa ^SETUP6_FAIL /workspace/logs/setup.log && echo F' | grep -q F && end SETUP_FAILED
echo "$(date -u +%H:%M) setup done"
S 'for n in full col pilot; do mkdir -p /workspace/luthor/runs/r6/sys1_$n && cp -r /root/d/sys1/ckpt_$n/* /workspace/luthor/runs/r6/sys1_$n/; done'
run f1diag f1diag "OUTN=sys1_full"; wait_for f1diag DONE_F1DIAG NOPE; S 'grep -a "^normal\|^off\|^cap2\|^inv" /workspace/logs/f1diag.log'
run evalset evalset_full "OUTN=sys1_full K=8"; wait_for evalset_full DONE_EVALSET NOPE; S 'grep -a "^\[" /workspace/logs/evalset_full.log'
run sys1 sys1_v4 "OUTN=sys1_v4 BUDGET=9000 HOURS=${TRAINH:-3.5} DEVDBS=dev_dbs_mix.json EVAL_EVERY=1000 EXTRA=\"--col-mentions --mention-weight inv --desc-drop 0.2 --schema-aug 0.2 --repeat bird_train=3 --fit-hours --mention-drop 0.3 --bird-filter /root/d/sys1/bird_filtered.jsonl\""
wait_for sys1_v4 DONE_SYS1 SYS1_FAIL || end TRAIN_FAILED
S "grep -aE '^(GATE|done in|eval step)' /workspace/logs/sys1_v4.log"; pull
run evalset evalset_v4 "OUTN=sys1_v4 K=8"; wait_for evalset_v4 DONE_EVALSET NOPE; S 'grep -a "^\[" /workspace/logs/evalset_v4.log'
run rr rr_v4 "OUTN=sys1_v4 DEVDBS=dev_dbs_mix.json K=32 DISTINCT=1 RRD=rr32w RRWIDE=1 RREP=2 RRH=${RRH:-1.0}"
wait_for rr_v4 DONE_RR RR_FAIL; S "grep -aE '^(SCORE|GEN|\[)' /workspace/logs/rr_v4.log"
pull
run rrtrain rr_v4_prev "OUTN=sys1_v4 RRD=rr32w_prev RRSRC=rr32w RRFLAGS=--result-preview RREP=2 RRH=${RRH:-1.0}"
wait_for rr_v4_prev DONE_RR RR_FAIL; S "grep -aE '^\[' /workspace/logs/rr_v4_prev.log"; pull
for v in rr32w rr32w_prev; do
  run pool pool_$v "OUTN=sys1_v4 RRD=$v RRSRC=rr32w K=32 POOLK=48 POOLRUNS=\"sys1_full sys1_col sys1_pilot\""
  wait_for pool_$v DONE_POOL POOL_FAIL; S "grep -aE '^\[' /workspace/logs/pool_$v.log"
done
end ALL_DONE
