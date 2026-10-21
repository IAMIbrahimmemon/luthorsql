#!/bin/bash
# Mac side: follow the two R1/R2 boxes, pull their logs and results into runs/r1r2/<role>/, and do the rare-data
# hand-off. It NEVER deletes an instance: when a box finishes it says so and the owner deletes (and verifies) it.
#   caffeinate -i bash scripts/chains/watch_r1r2.sh <R1_instance_id> <R2_instance_id>    ("-" for a role not running)
# Env: EVERY (poll seconds, default 600), DEAD_POLLS (ssh failures in a row before a box counts as lost, default 18).
#
# Rare-data hand-off: each box writes its qwrite shard, then logs/rare_shard.done (or .failed). Once both boxes
# have one (or a box is lost), this pulls both boxes' /root/d/rare1/work/ (written/checked.sIof2.jsonl, shard-specific
# names) into runs/r1r2/rare_merge/rare1/work, pushes the union to /root/d/rare1 on every live box (push6.sh +
# leakcheck) and writes
# logs/rare.go = "use" (or "none" when no shard succeeded or a push failed) on each, so both train on the same data.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
A=${1:?usage: watch_r1r2.sh <R1_id|-> <R2_id|->}; B=${2:?usage: watch_r1r2.sh <R1_id|-> <R2_id|->}
EVERY=${EVERY:-600}; DEAD_POLLS=${DEAD_POLLS:-18}
OUT=runs/r1r2; ST=$OUT/.watch; M=$OUT/rare_merge/rare1
mkdir -p "$ST" "$M"
log() { echo "$(date -u '+%m-%d %H:%M') $*"; }
id_of() { if [ "$1" = r1 ]; then echo "$A"; else echo "$B"; fi; }
S() { bash scripts/vast/ssh.sh "$1" "$2" 2>/dev/null | grep -av "Welcome\|Have fun"; }
alive() { bash scripts/vast/ssh.sh "$1" true >/dev/null 2>&1; }
note() { osascript -e "display notification \"$1\" with title \"luthor r1r2\"" >/dev/null 2>&1 || true; }
pull() {  # role id final(0|1)
  mkdir -p "$OUT/$1/logs"
  ( source scripts/vast/ssh.sh "$2" >/dev/null 2>&1 || exit 1
    "${RS[@]}" "$H:/workspace/logs/" "$OUT/$1/logs/" || exit 1
    if [ "$3" = 1 ]; then "${RS[@]}" --exclude last_lora --exclude last_heads.pt "$H:/workspace/luthor/runs/r1r2/$1/" "$OUT/$1/"
    else "${RS[@]}" --exclude '*_lora' --exclude '*.pt' "$H:/workspace/luthor/runs/r1r2/$1/" "$OUT/$1/"; fi ) >/dev/null 2>&1
}
for r in r1 r2; do [ "$(id_of $r)" = - ] && touch "$ST/$r.finished"; done
rare_handoff() {
  [ -e "$ST/rare.merged" ] && return 0
  local r id st states="" ready=1 any=0 dec
  for r in r1 r2; do
    id=$(id_of $r)
    if [ "$id" = - ] || [ -e "$ST/$r.lost" ]; then st=lost
    else st=$(S "$id" 'cd /workspace/logs 2>/dev/null || exit 0
        if [ -e rare.decided ]; then echo "decided:$(cut -d" " -f1 rare.decided)"; elif [ -e rare.go ]; then echo go
        elif [ -e rare_shard.done ]; then echo done; elif [ -e rare_shard.failed ]; then echo failed; else echo wait; fi')
      st=${st:-unreachable}; fi
    states+="$r=$st "
    case "$st" in
      go|decided:use) touch "$ST/rare.merged"; log "rare: decision already sent ($states)"; return 0;;
      decided:*) # this box gave up waiting and trains without rare data: tell the other box now, so both match
        for o in r1 r2; do
          id=$(id_of $o); { [ "$o" = "$r" ] || [ "$id" = - ] || [ -e "$ST/$o.lost" ]; } && continue
          S "$id" '[ -e /workspace/logs/rare.go ] || echo "none (other box went without)" > /workspace/logs/rare.go' >/dev/null
        done
        touch "$ST/rare.merged"; log "rare: $r went without rare data; told the other box to match ($states)"; return 0;;
      done) any=1;; failed|lost) ;; *) ready=0;; esac
  done
  [ $ready = 1 ] || { log "rare: waiting for shards ($states)"; return 0; }
  dec=none
  if [ $any = 1 ]; then
    dec=use
    for r in r1 r2; do
      id=$(id_of $r); [ "$id" = - ] && continue
      case "$states" in *"$r=done"*) ( source scripts/vast/ssh.sh "$id" >/dev/null 2>&1 &&
        mkdir -p "$M/work" && "${RS[@]}" "$H:/root/d/rare1/work/" "$M/work/" ) >/dev/null 2>&1 || { log "rare: pull from $r failed"; dec=none; };; esac
    done
    for r in r1 r2; do
      id=$(id_of $r); { [ "$id" = - ] || [ -e "$ST/$r.lost" ]; } && continue
      [ $dec = use ] || break
      ok=0; for _ in 1 2 3; do
        bash scripts/vast/push6.sh "$id" "$M" > "$ST/push_rare_$r.log" 2>&1 \
          && grep -q '^PUSHED6' "$ST/push_rare_$r.log" && { ok=1; break; }; sleep 30; done
      [ $ok = 1 ] || { log "rare: push to $r failed 3 times (see $ST/push_rare_$r.log): both boxes go WITHOUT rare data"; dec=none; }
    done
  fi
  for r in r1 r2; do
    id=$(id_of $r); { [ "$id" = - ] || [ -e "$ST/$r.lost" ]; } && continue
    S "$id" "echo $dec > /workspace/logs/rare.go && echo ok" | grep -q ok || log "rare: could not write rare.go on $r (the box times out and goes without)"
  done
  touch "$ST/rare.merged"; log "rare: shards ($states) -> rare.go=$dec"; note "rare data: $dec"
}
# shellcheck disable=SC2034  # read through eval
fails_r1=0 fails_r2=0
while true; do
  for r in r1 r2; do
    [ -e "$ST/$r.finished" ] && continue
    id=$(id_of $r)
    cost=$(python3 scripts/shade/sf.py info "$id" 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status"), "spent $%s" % d.get("cost_estimate"))' 2>/dev/null || echo "?")
    if ! alive "$id"; then
      eval "fails_$r=\$(( fails_$r + 1 ))"; n=$(eval echo "\$fails_$r")
      st=$(python3 scripts/shade/sf.py status "$id" 2>/dev/null || echo unknown)
      log "$r ($id): ssh failed ($n/$DEAD_POLLS), API status=$st"
      if [ "$st" = gone ] || [ "$st" = deleted ] || [ "$n" -ge "$DEAD_POLLS" ]; then
        log "$r LOST (status $st): last pulled results stay in $OUT/$r"; note "$r box lost ($st)"
        touch "$ST/$r.lost" "$ST/$r.finished"
      fi
      continue
    fi
    eval "fails_$r=0"
    pull "$r" "$id" 0 || log "$r: pull failed (will retry)"
    prog=$(S "$id" 'L=/workspace/logs; grep -aE "^(DONE_[A-Z0-9_]+|[A-Z0-9_]+_FAIL|PROJECTION.*|.*RARE_(USED|SKIPPED).*|.*FATAL.*)$" $L/chain.log 2>/dev/null | tail -2;
      f=$(ls -t $L/train_*.log 2>/dev/null | head -1); [ -n "$f" ] && grep -aE "^(step [0-9]+/|GATE|done in)" "$f" | tail -1 | cut -c1-150;
      [ -e $L/r1r2.done ] && echo "@@DONE"; [ -e $L/r1r2.failed ] && echo "@@FAILED $(cat $L/r1r2.failed)"; true')
    log "$r ($id) [$cost]"; echo "$prog" | grep -v '^@@' | sed 's/^/    /'
    if echo "$prog" | grep -q '^@@DONE\|^@@FAILED'; then
      pull "$r" "$id" 1 || log "$r: final pull failed, retrying next poll"
      if [ -s "$OUT/$r/logs/r1r2.done" ] || [ -s "$OUT/$r/logs/r1r2.failed" ]; then
        what=$(echo "$prog" | grep '^@@' | head -1 | sed 's/^@@//')
        log "$r $what: everything pulled to $OUT/$r. NOT deleted. Delete it yourself and verify:"
        log "    CONFIRM=yes bash scripts/vast/destroy.sh $id      (prints VERIFIED_DELETED)"
        note "$r $what: pulled, ready to delete"; touch "$ST/$r.finished"
      fi
    fi
  done
  rare_handoff
  [ -e "$ST/r1.finished" ] && [ -e "$ST/r2.finished" ] && { log "both roles finished: results in $OUT/r1, $OUT/r2"; exit 0; }
  sleep "$EVERY"
done
