#!/bin/bash
# Mac side of the free-form pilot ff1 (plan S1 + S2; box script scripts/chains/ff1_box.sh):
#   CAP=<dollars> bash scripts/chains/run_ff1_box.sh <instance id>
# 1. stage data/ff1 (scripts/freeform/ff1_stage.py: DECISION pool + ids, BIRD-train bug-set ids, DB list, optional
#    arithmetic mutants from data/pool/decision_arith.jsonl); STAGE_ONLY=1 stops here.
# 2. wait for ssh (the box's IP may be reused: its stale known_hosts entries are removed first),
# 3. push code + data with push6.sh (leakcheck first; REMOTE_D=/root/d layout):
#      data/ff1 -> /root/d/ff1, data/raw/bird_train/train.json -> /root/d/ff1/train.json,
#      data/raw/bird_train/train_databases/<db> (dbs.txt, SQLITE_ONLY) -> /root/d/ffdbs/<db>,
#      data/ens3/{v_r1,v_r2,sys1} (R1/R2 validators + BIRD-train cards) -> /root/d/ens3/...
# 4. start ff1_box.sh detached and exec watch_box.sh (pulls /workspace/ff1 + logs into runs/phase1/ff1/, deletes the box).
# No RESERVED record, no BIRD-dev file and no sealed set is pushed.
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=${1:?usage: CAP=<dollars> bash scripts/chains/run_ff1_box.sh <instance id>}; NAME=ff1
mkdir -p runs/phase1/$NAME
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a runs/phase1/$NAME/watch.log; }
[ -n "${STAGE_ONLY:-}" ] || : "${CAP:?CAP dollars required (watch_box.sh spend guard)}"
uv run python scripts/freeform/ff1_stage.py 2>&1 | tee -a runs/phase1/$NAME/stage.log || { log "staging failed"; exit 1; }
for f in data/ff1/decision_ids.json data/ff1/decision_pool.jsonl data/ff1/bug_train_ids.json data/ff1/dbs.txt \
         data/raw/bird_train/train.json data/ens3/v_r1/rerank_head.pt data/ens3/v_r2/rerank_head.pt data/ens3/sys1/bird_train/cards; do
  [ -e "$f" ] || { log "missing $f (the box would fail): not starting"; exit 1; }
done
DBS=(); while read -r d; do [ -n "$d" ] && DBS+=("data/raw/bird_train/train_databases/$d"); done < data/ff1/dbs.txt
bash scripts/vast/leakcheck.sh data/ff1 data/raw/bird_train/train.json "${DBS[@]}" data/ens3/v_r1 data/ens3/v_r2 data/ens3/sys1 \
  > runs/phase1/$NAME/leakcheck.log 2>&1 || { tail -5 runs/phase1/$NAME/leakcheck.log; log "leakcheck failed: not starting"; exit 1; }
log "staged + leakcheck ok (${#DBS[@]} DBs)"
[ -n "${STAGE_ONLY:-}" ] && exit 0

hostport() {  # -> "HOST PORT" of the instance, or nothing while it is starting
  if [[ "$ID" == *-* ]]; then python3 scripts/shade/sf.py hostline "$ID" 2>/dev/null | awk '{print $1, $2}'
  else vastai ssh-url "$ID" 2>/dev/null | sed -nE 's#.*ssh://[^@]+@([^:]+):([0-9]+).*#\1 \2#p' | head -1; fi
}
cleared=0
for i in $(seq 1 240); do
  if [ $cleared = 0 ] && read -r HOST PORT < <(hostport) && [ -n "${HOST:-}" ]; then
    for k in "$HOST" "[$HOST]:${PORT:-22}"; do ssh-keygen -R "$k" >/dev/null 2>&1 || true; done
    log "cleared known_hosts for $HOST (port ${PORT:-22})"; cleared=1
  fi
  [ $cleared = 1 ] && REFRESH=1 bash scripts/vast/ssh.sh "$ID" true >/dev/null 2>&1 && break
  [ "$i" = 240 ] && { log "ssh never came up: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
  sleep 15
done
log "ssh up"
ok=0
for t in 1 2 3; do
  REMOTE_D=/root/d bash scripts/vast/push6.sh "$ID" data/ff1 >> runs/phase1/$NAME/push.log 2>&1 \
    && REMOTE_D=/root/d/ff1 bash scripts/vast/push6.sh "$ID" data/raw/bird_train/train.json >> runs/phase1/$NAME/push.log 2>&1 \
    && SQLITE_ONLY=1 REMOTE_D=/root/d/ffdbs bash scripts/vast/push6.sh "$ID" "${DBS[@]}" >> runs/phase1/$NAME/push.log 2>&1 \
    && REMOTE_D=/root/d/ens3 bash scripts/vast/push6.sh "$ID" data/ens3/v_r1 data/ens3/v_r2 data/ens3/sys1 >> runs/phase1/$NAME/push.log 2>&1 \
    && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (QWEN_NOEV='${QWEN_NOEV:-1}' MAX_MUT='${MAX_MUT:-0}' setsid nohup bash /workspace/luthor/scripts/chains/ff1_box.sh > /workspace/logs/ff1.log 2>&1 < /dev/null &)" 2>/dev/null
log "started"
STALL_MIN=${STALL_MIN:-60} FAIL_REGEX='FF1_FAIL' exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/ff1.log FF1_DONE /workspace/ff1
