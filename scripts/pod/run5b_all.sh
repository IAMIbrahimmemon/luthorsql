# Session B chain: writer v2 SFT -> dev candidates -> vote / repair comparisons. Logs in /workspace/logs.
L=/workspace/logs; R=/workspace/luthor/scripts/pod/run5.sh
for s in sft5 devgen5 seleval5; do
  SFTH=${SFTH:-3.5} STAGE=$s bash $R > $L/$s.log 2>&1
  grep -q "DONE_$(echo $s | tr a-z A-Z)" $L/$s.log || { echo "STOP at $s" > $L/b.done; exit 1; }
done
echo B_DONE > $L/b.done
