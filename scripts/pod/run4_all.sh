# Runs ON the pod: the whole round-4 night in order, each stage logged to /workspace/logs/<stage>.log.
#   sft -> diag -> wgen (needs sft + diag) -> rl (needs sft) -> rlgen (needs rl)
# A stage that fails does not stop independent later stages.
cd /workspace/luthor
L=/workspace/logs
(while true; do echo "$(date -u +%H:%M) gpu $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) host $(free -g | awk '/Mem/{print $3}')G" >> $L/gpu.log; sleep 60; done) &
run() { STAGE=$1 bash scripts/pod/run4.sh > $L/$1.log 2>&1; grep -q "^DONE_" $L/$1.log; }
ok_sft=0; ok_diag=0; ok_rl=0
SFTH=${SFTH:-4} run sft && ok_sft=1
run diag && ok_diag=1
[ $ok_sft = 1 ] && [ $ok_diag = 1 ] && run wgen
[ $ok_sft = 1 ] && RLH=${RLH:-4.5} run rl && ok_rl=1
[ $ok_rl = 1 ] && [ $ok_diag = 1 ] && run rlgen
echo ALL_DONE sft=$ok_sft diag=$ok_diag rl=$ok_rl > $L/all.done
