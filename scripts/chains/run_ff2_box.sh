#!/bin/bash
# Mac side of free-form phase 2 (ff2; box script scripts/chains/ff2_box.sh; prereg runs/phase1/ff2/PREREG.md):
#   CAP=<dollars> bash scripts/chains/run_ff2_box.sh <instance id>
# 0. refuses to start without runs/phase1/ff2/PREREG.md (written before any launch).
# 1. stage data/ff2 (scripts/freeform/ff2_stage.py: BIRD-train subset questions + grammar lists + relabelled arithmetic
#    mutants + the phase-1 DECISION pools) when data/ff2/stage_stats.json is missing (FORCE_STAGE=1 re-stages);
#    STAGE_ONLY=1 stops after staging + leakcheck.
# 2. wait for ssh (the box's IP may be reused: its stale known_hosts entries are removed first),
# 3. push code + data with push6.sh (leakcheck first; REMOTE_D layout):
#      data/ff2 -> /root/d/ff2, data/val2/heldout_union.jsonl -> /root/d/ff2/heldout_union.jsonl,
#      data/raw/bird_train/train_databases/<db> (data/ff2/dbs.txt, SQLITE_ONLY) -> /root/d/ff2dbs/<db>,
#      data/sys1/bird_train/cards -> /root/d/sys1/bird_train/cards
# 4. start ff2_box.sh detached (env N TEMP TOK MAX_HOURS MAX_Q passed through) and exec watch_box.sh (pulls
#    /workspace/ff2 + logs into runs/phase1/ff2/, deletes the box).
# No DECISION/RESERVED question is generated for or trained on; the held-out union (val2's, RESERVED rows included) is
# only scored, and its SCORE line is removed on the box. No BIRD-dev file and no sealed set is pushed.
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=${1:?usage: CAP=<dollars> bash scripts/chains/run_ff2_box.sh <instance id>}; NAME=ff2
mkdir -p runs/phase1/$NAME
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a runs/phase1/$NAME/watch.log; }
[ -n "${STAGE_ONLY:-}" ] || : "${CAP:?CAP dollars required (watch_box.sh spend guard)}"
[ -s runs/phase1/$NAME/PREREG.md ] || { log "runs/phase1/$NAME/PREREG.md missing: write the prereg first"; exit 1; }
if [ -n "${FORCE_STAGE:-}" ] || [ ! -s data/ff2/stage_stats.json ]; then
  uv run python scripts/freeform/ff2_stage.py 2>&1 | tee -a runs/phase1/$NAME/stage.log || { log "staging failed"; exit 1; }
fi
for f in data/ff2/gen_questions.jsonl data/ff2/grammar_lists.jsonl data/ff2/dbs.txt data/ff2/dec/superset.jsonl \
         data/ff2/dec/m_both_mut.jsonl data/val2/heldout_union.jsonl data/sys1/bird_train/cards; do
  [ -e "$f" ] || { log "missing $f (the box would fail): not starting"; exit 1; }
done
DBS=(); while read -r d; do [ -n "$d" ] && DBS+=("data/raw/bird_train/train_databases/$d"); done < data/ff2/dbs.txt
bash scripts/vast/leakcheck.sh data/ff2 data/val2/heldout_union.jsonl "${DBS[@]}" data/sys1/bird_train/cards \
  > runs/phase1/$NAME/leakcheck.log 2>&1 || { tail -5 runs/phase1/$NAME/leakcheck.log; log "leakcheck failed: not starting"; exit 1; }
log "staged + leakcheck ok (${#DBS[@]} DBs, $(wc -l < data/ff2/gen_questions.jsonl) questions)"
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
  REMOTE_D=/root/d bash scripts/vast/push6.sh "$ID" data/ff2 >> runs/phase1/$NAME/push.log 2>&1 \
    && REMOTE_D=/root/d/ff2 bash scripts/vast/push6.sh "$ID" data/val2/heldout_union.jsonl >> runs/phase1/$NAME/push.log 2>&1 \
    && SQLITE_ONLY=1 REMOTE_D=/root/d/ff2dbs bash scripts/vast/push6.sh "$ID" "${DBS[@]}" >> runs/phase1/$NAME/push.log 2>&1 \
    && REMOTE_D=/root/d/sys1/bird_train bash scripts/vast/push6.sh "$ID" data/sys1/bird_train/cards >> runs/phase1/$NAME/push.log 2>&1 \
    && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (N='${N:-8}' TEMP='${TEMP:-1.0}' TOK='${TOK:-120000}' MAX_HOURS='${MAX_HOURS:-2.0}' MAX_Q='${MAX_Q:-0}' setsid nohup bash /workspace/luthor/scripts/chains/ff2_box.sh > /workspace/logs/ff2.log 2>&1 < /dev/null &)" 2>/dev/null
log "started"
STALL_MIN=${STALL_MIN:-60} FAIL_REGEX='FF2_FAIL' exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/ff2.log FF2_DONE /workspace/ff2
