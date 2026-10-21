# Session A chain: merge -> devgen -> traingen -> seldata -> seltrain -> seleval. Logs in /workspace/logs.
L=/workspace/logs; R=/workspace/luthor/scripts/pod/run5.sh; mkdir -p $L
( while true; do echo "$(date -u +%H:%M) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) host $(free -g | awk '/Mem/{print $3}')G" >> $L/gpu.log; sleep 60; done ) &
for s in merge devgen traingen seldata seltrain seleval; do
  STAGE=$s bash $R > $L/$s.log 2>&1
  grep -q "DONE_$(echo $s | tr a-z A-Z)" $L/$s.log || { echo "STOP at $s" > $L/all.done; exit 1; }
done
echo ALL_DONE > $L/all.done
