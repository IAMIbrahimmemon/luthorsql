#!/bin/bash
# Round 6 watcher (runs on the Mac): wait until a NEW DONE_<X> / <X>_FAIL line appears in the instance's
# $LOGS/*.log, or the done-file $DONEF exists on the instance. Prints the new markers and exits 0.
#   DONEF=/workspace/logs/all.done bash scripts/vast/watch.sh <id> [n0]
# n0 = marker count already seen (default: count now). Env: LOGS (default /workspace/logs), EVERY (s, default 120).
# Exits 2 after 15 failed ssh polls in a row (instance gone or host down).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/vast/ssh.sh "${1:?usage: watch.sh <instance_id> [n0]}"
LOGS=${LOGS:-/workspace/logs}; EVERY=${EVERY:-120}; DONEF=${DONEF:-}
RE='^(DONE_[A-Z0-9]+|[A-Z0-9]+_FAIL)$'
POLL="n=\$(cat $LOGS/*.log 2>/dev/null | grep -cE '$RE' || true); f=N; [ -n '$DONEF' ] && [ -e '$DONEF' ] && f=Y; echo \$n \$f"
N0=${2:-}
if [ -z "$N0" ]; then read -r N0 _ < <("${S[@]}" "$H" "$POLL") || { echo "ssh failed"; exit 2; }; echo "baseline: $N0 markers"; fi
fails=0
while true; do
  if out=$("${S[@]}" "$H" "$POLL" 2>/dev/null); then
    fails=0; read -r n f <<< "$out"
    if [ "${n:-0}" -gt "$N0" ] || [ "$f" = Y ]; then
      echo "$(date -u +%H:%M) markers $N0 -> $n, donefile=$f"
      "${S[@]}" "$H" "grep -HE '$RE' $LOGS/*.log 2>/dev/null | tail -n $(( n > N0 ? n - N0 : 1 ))"
      [ "$f" = Y ] && "${S[@]}" "$H" "cat '$DONEF'"
      exit 0
    fi
  else
    fails=$((fails + 1)); echo "$(date -u +%H:%M) ssh poll failed ($fails/15)"
    [ $fails -lt 15 ] || exit 2
  fi
  sleep "$EVERY"
done
