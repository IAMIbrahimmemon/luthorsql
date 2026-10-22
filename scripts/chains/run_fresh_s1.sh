#!/bin/bash
# Mac side of the held-out System 1 run (experiments/cost_confirmation): instance created first (sf.py create); never creates one.
#   CAP=6 PRICE_H=2.50 bash scripts/chains/run_heldout_s1.sh <instance_id>
# Builds the package (scripts/bird_submit/build_package.sh), waits for ssh, pushes code + data/bird_submit +
# the BIRD dev SQLite databases (push6 runs leakcheck), starts bird_check_box.sh detached, polls the log, pulls
# /workspace/birdcheck into runs/phase1/bird_check/, then DESTROYS the instance (on done, fail, stall or spend cap).
# Spend guard: elapsed hours x PRICE_H >= CAP -> pull + destroy. Never touches any other instance.
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=${1:?usage: run_bird_check.sh <vast_instance_id>}
CAP=${CAP:-6}; PRICE_H=${PRICE_H:?PRICE_H required}; POLL=${POLL:-120}; STALL_MIN=${STALL_MIN:-40}
OUT=runs/phase2/${OUT_NAME:-fresh_s1}; mkdir -p "$OUT"   # OUT_NAME: a separate results dir (e.g. bird_check_single); SINGLE=<dir> packs the single-LoRA weights
log() { echo "$(date -u '+%m-%d %H:%M:%S') [fresh] $*" | tee -a "$OUT/watch.log"; }
T0=$(date +%s)
spent() { python3 -c "print(f'{($(date +%s) - $T0) / 3600 * $PRICE_H:.2f}')"; }
over() { python3 -c "import sys; sys.exit(0 if ($(date +%s) - $T0) / 3600 * $PRICE_H >= $CAP else 1)"; }
destroy() { log "destroying $ID ($1, ~\$$(spent))"; echo "$1" > "$OUT/VERDICT"
  CONFIRM=yes bash scripts/vast/destroy.sh "$ID" 2>&1 | tee -a "$OUT/watch.log" | grep -q VERIFIED_DELETED \
    && log VERIFIED_DESTROYED || log "STILL_LISTED: check vastai show instances / sf.py list"; }   # Vast or Shadeform id
pull() { ( source scripts/vast/ssh.sh "$ID" >/dev/null 2>&1 || exit 1
  "${RS[@]}" "$H:/workspace/external/" "$OUT/external/" >/dev/null 2>&1; "${RS[@]}" "$H:/workspace/logs/" "$OUT/logs/" >/dev/null 2>&1 ) \
  || log "pull failed"; }
bash scripts/bird_submit/build_package.sh > "$OUT/build.log" 2>&1 || { log "build failed: not starting"; destroy build_failed; exit 1; }
log "$(tail -1 "$OUT/build.log")"
for i in $(seq 1 80); do
  REFRESH=1 bash scripts/vast/ssh.sh "$ID" 'command -v rsync' >/dev/null 2>&1 && break
  over && { destroy cap_before_ssh; exit 1; }
  [ "$i" = 80 ] && { log "ssh/rsync never came up"; destroy no_ssh; exit 1; }
  sleep 15
done
log "ssh up"
REMOTE_D=/root/d bash scripts/vast/push6.sh "$ID" data/bird_submit data/sys1/fresh_blind >> "$OUT/push.log" 2>&1 \
  || { log "push failed"; destroy push_failed; exit 1; }
bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (RESUME_TEST=${RESUME_TEST:-0} RESUME_AFTER=${RESUME_AFTER:-3} setsid nohup bash /workspace/luthor/scripts/chains/fresh_s1_box.sh > /workspace/logs/external.log 2>&1 < /dev/null &)"
log "started (\$$PRICE_H/h, cap \$$CAP)"
last=""; tlast=$(date +%s); fails=0
while true; do
  sleep "$POLL"
  if s=$(bash scripts/vast/ssh.sh "$ID" 'tail -c 400 /workspace/logs/external.log; echo; tail -n 1 /workspace/external/run.log 2>/dev/null' 2>/dev/null); then
    fails=0
    [ "$s" != "$last" ] && { last=$s; tlast=$(date +%s); log "$(echo "$s" | tail -n 1 | cut -c1-160)"; }
    echo "$s" | grep -q '^DONE_EXTERNAL' && { pull; destroy finished; exit 0; }
    echo "$s" | grep -q 'EXTERNAL_FAIL' && { pull; destroy failed; exit 1; }
    [ $(( $(date +%s) - tlast )) -gt $(( STALL_MIN * 60 )) ] && { pull; destroy stalled; exit 1; }
  else
    fails=$((fails + 1)); [ $fails -ge 15 ] && { log "unreachable"; destroy lost; exit 1; }
  fi
  over && { pull; destroy cap; exit 1; }
done
