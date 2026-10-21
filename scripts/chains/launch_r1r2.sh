#!/bin/bash
# Mac side: wait for a Shadeform box's ssh, push code + the data the R1/R2 chain needs, start the chain detached.
#   N=6714 bash scripts/chains/launch_r1r2.sh <instance_id> R1        (and ... <other_id> R2 for the second box)
# Forwarded to the box when set here: N TRAIN_EXTRA DEVALL_FLAGS QW_WRITE_ARGS QW_CHECK_ARGS QW_MODELS RARE_BUILD
#   RARE_PLANS RARE_MIN RARE_MAX_MIN RARE_WAIT_MIN PRICE_H EVAL_EST_H SKIP_RARE SEED GRAMMAR VOCAB DROP_DBS BIRD_PLANS
#   SPIDER_PLANS SYNSQL_PLANS BIRD_DEV_PLANS DESC DESC_ARGS WIDE_K (see r1r2_chain.sh).
# Pushes (all through push6.sh + leakcheck.sh): code; data/sys1 parts the chain reads (no sealed sets, no checkpoints);
# the original BIRD dev databases + dev.json (evaluation only: all 1,534 questions); data/rare1 (rare-skeleton inputs).
# Never creates or deletes instances.
set -euo pipefail
cd "$(dirname "$0")/../.."
ID=${1:?usage: launch_r1r2.sh <instance_id> R1|R2}; ROLE=${2:?usage: launch_r1r2.sh <instance_id> R1|R2}
case "$ROLE" in R1|R2) ;; *) echo "ROLE must be R1 or R2"; exit 2;; esac
log() { echo "$(date -u +%H:%M:%S) [$ROLE $ID] $*"; }
for i in $(seq 1 120); do   # up to ~30 min for boot + root login
  REFRESH=1 bash scripts/vast/ssh.sh "$ID" true >/dev/null 2>&1 && break
  [ "$i" = 120 ] && { log "ssh never came up"; exit 1; }
  sleep 15
done
log "ssh up: $(bash scripts/vast/ssh.sh "$ID" 'nvidia-smi --query-gpu=name,memory.used --format=csv,noheader' 2>/dev/null | grep -av Welcome)"
if bash scripts/vast/ssh.sh "$ID" 'pgrep -f "[r]1r2_chain.sh" >/dev/null && echo RUNNING' 2>/dev/null | grep -q RUNNING; then
  log "a chain is already running on this box: not starting another"; exit 1
fi
SYS1=(data/sys1/bird_train data/sys1/spider_train data/sys1/synsql_train data/sys1/dbs data/sys1/rrdbs data/sys1/bird_dev
      data/sys1/templates.json data/sys1/templates_nums.json data/sys1/dev_dbs_bird.json data/sys1/bird_dev_dbs.json
      data/sys1/contaminated_dbs.json)
REMOTE_D=/root/d/sys1 bash scripts/vast/push6.sh "$ID" "${SYS1[@]}"
bash scripts/vast/push6.sh "$ID" data/bench/dev_20240627/dev_databases data/bench/dev_20240627/dev.json
if [ -s data/rare1/selected.jsonl ]; then bash scripts/vast/push6.sh "$ID" data/rare1   # -> /root/d/rare1 (qwrite --data /root/d)
else log "WARNING: no data/rare1/selected.jsonl (build_rare_mix.py select not run): the box's qwrite stage will fail and the run continues without rare data"; fi
envs="ROLE=$ROLE "
for v in N TRAIN_EXTRA DEVALL_FLAGS QW_WRITE_ARGS QW_CHECK_ARGS QW_MODELS RARE_BUILD RARE_PLANS RARE_MIN RARE_MAX_MIN \
         RARE_WAIT_MIN PRICE_H EVAL_EST_H SKIP_RARE SEED GRAMMAR VOCAB DROP_DBS BIRD_PLANS SPIDER_PLANS SYNSQL_PLANS \
         BIRD_DEV_PLANS DESC DESC_ARGS WIDE_K; do
  if [ -n "${!v+x}" ]; then envs+="$v=$(printf %q "${!v}") "; fi
done
log "starting chain: $envs"
bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (${envs} setsid nohup bash /workspace/luthor/scripts/chains/r1r2_chain.sh \
  >> /workspace/logs/chain.log 2>&1 < /dev/null &)" 2>/dev/null
sleep 10
bash scripts/vast/ssh.sh "$ID" 'tail -n 5 /workspace/logs/chain.log' 2>/dev/null | grep -av Welcome || true
log "LAUNCHED"
