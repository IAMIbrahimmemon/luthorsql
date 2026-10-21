#!/bin/bash
# Mac side of the System-1 latency box: stage data/latency (scripts/analysis/latency/sys1_latency.py stage: a seeded
# sample of 200 + 3 DECISION questions, labels stripped, the 8 DECISION cards, a VALHARD copy, manifest), leakcheck,
# wait for the box, push code + the R1 checkpoint + data/latency + the 8 DECISION SQLite DBs (push6 runs leakcheck on
# every path), start latency_box.sh detached, then watch (watch_box.sh pulls /workspace/latency + logs into
# runs/phase1/latency/ and deletes ONLY this instance id, verified); latency.json / latency.txt are copied up to
# runs/phase1/latency/.
#   DRY=1 bash scripts/chains/run_latency_box.sh -               # stage + leakcheck + print the push plan; no ssh, no spend
#   PRICE_H=2.50 bash scripts/chains/run_latency_box.sh <instance_id>   # CAP default 6
# Box (one H100 80 GB, created first by hand; this script never creates one):
#   python3 scripts/shade/sf.py offers H100 --n 5
#   python3 scripts/shade/sf.py create <cloud> <region> <type> --max-spend 8 --name luthor-latency
# Generator: R1 only (runs/r1r2/r1: best_lora + best_heads.pt + args.json + vocab.json, grammar v3). Validator: VALHARD
#   (runs/phase1/valhard/valhard/rr, copied into data/latency/rr_valhard so leakcheck's allowlist covers it).
# Nothing from BIRD dev, sealed, gate or RESERVED is staged or pushed; no label travels (latency only).
# Env: PRICE_H ($/h of the box, 2.50; passed to the box for $/1000 questions), CAP (watch_box hard stop, 6),
#      STALL_MIN (30), N / SQ_MAX_MIN (passed to latency_box.sh).
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=${1:-}; DRY=${DRY:-0}; NAME=${NAME:-latency}; OUT=runs/phase1/$NAME
PRICE_H=${PRICE_H:-2.50}; CAP=${CAP:-6}
[ -n "$ID" ] && { [ "$ID" != - ] || [ "$DRY" = 1 ]; } || { echo "usage: bash scripts/chains/run_latency_box.sh <instance_id>   (or DRY=1 ... -)"; exit 2; }
[[ "$PRICE_H" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "bad PRICE_H '$PRICE_H'"; exit 2; }
mkdir -p "$OUT"
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a "$OUT/watch.log"; }
[ "$DRY" = 1 ] && log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME DRY] $*"; }
if [ "$DRY" != 1 ] && { [ -e "$OUT/VERDICT" ] || [ -d "$OUT/latency" ]; }; then
  log "$OUT already holds a run (VERDICT / latency/): move it away first; not starting"; exit 1
fi
GEN=runs/r1r2/r1; LD=data/latency
for f in $GEN/best_lora/adapter_model.safetensors $GEN/best_heads.pt $GEN/args.json $GEN/vocab.json \
         runs/phase1/valhard/valhard/rr/rerank_head.pt data/sys1/bird_train/cards data/sys1/heldout_v2/decision.json; do
  [ -e "$f" ] || { log "missing $f: not starting"; exit 1; }
done

# ---- stage data/latency ----------------------------------------------------------------------------------------------
st=$(python3 scripts/analysis/latency/sys1_latency.py stage 2>&1) || { echo "$st" | tail -n 20; log "staging failed: nothing sent"; exit 1; }
log "$(echo "$st" | grep '^STAGED' | sed "s#$(pwd)/##")"
read -r -a DBS < <(python3 -c "import json; print(' '.join(json.load(open('$LD/sample.json'))['databases']))")
[ "${#DBS[@]}" = 8 ] || { log "expected 8 DECISION DBs, got ${#DBS[@]}: not starting"; exit 1; }
DBP=(); for d in "${DBS[@]}"; do [ -s "data/sys1/dbs/$d/$d.sqlite" ] || { log "missing data/sys1/dbs/$d/$d.sqlite"; exit 1; }; DBP+=("data/sys1/dbs/$d"); done

# ---- what goes where ("REMOTE_D|paths") --------------------------------------------------------------------------------
pushes=("/root/d|$LD"
        "/root/d/sys1/dbs|${DBP[*]}"
        "/workspace/luthor/$GEN|$GEN/best_lora $GEN/best_heads.pt $GEN/args.json $GEN/vocab.json")
all=(src scripts pyproject.toml)
for p in "${pushes[@]}"; do read -r -a ps <<< "${p#*|}"; all+=("${ps[@]}"); done
bash scripts/vast/leakcheck.sh "${all[@]}" > /dev/null || { bash scripts/vast/leakcheck.sh "${all[@]}" | tail -n 20; log "leakcheck failed: nothing sent"; exit 1; }
log "leakcheck ok: ${#all[@]} paths, $(du -shc "${all[@]}" 2>/dev/null | tail -1 | cut -f1) on disk"
START="mkdir -p /workspace/logs && (SETUP=${SETUP:-setup6} PRICE_H=$PRICE_H N='${N:-0}' SQ_MAX_MIN='${SQ_MAX_MIN:-35}' setsid nohup bash /workspace/luthor/scripts/chains/latency_box.sh > /workspace/logs/latency.log 2>&1 < /dev/null &)"
WATCH=(scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/latency.log LATENCY_DONE /workspace/latency)
if [ "$DRY" = 1 ]; then
  for p in "${pushes[@]}"; do
    read -r -a ps <<< "${p#*|}"
    log "would push -> ${p%%|*}: ${ps[*]} ($(du -shc "${ps[@]}" | tail -1 | cut -f1))"
  done
  log "would push code: src scripts pyproject.toml ($(du -shc src scripts pyproject.toml 2>/dev/null | tail -1 | cut -f1), __pycache__ excluded by push6)"
  log "would start: $START"
  log "would watch: STALL_MIN=${STALL_MIN:-30} FAIL_REGEX=LATENCY_FAIL CAP=$CAP ${WATCH[*]}"
  exit 0
fi

for i in $(seq 1 240); do
  REFRESH=1 bash scripts/vast/ssh.sh "$ID" true >/dev/null 2>&1 && break
  [ "$i" = 240 ] && { log "ssh never came up: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
  sleep 15
done
log "ssh up"
ok=0
for t in 1 2 3; do
  good=1
  for p in "${pushes[@]}"; do
    read -r -a ps <<< "${p#*|}"
    REMOTE_D=${p%%|*} bash scripts/vast/push6.sh "$ID" "${ps[@]}" >> "$OUT/push.log" 2>&1 || { good=0; break; }
  done
  [ $good = 1 ] && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
bash scripts/vast/ssh.sh "$ID" "$START" 2>/dev/null
log "started (price \$$PRICE_H/h, cap \$$CAP)"
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') id=$ID price_h=$PRICE_H cap=$CAP sample=$(shasum -a 256 $LD/sample.json | cut -c1-16)" >> "$OUT/launches.log"
CAP=$CAP STALL_MIN=${STALL_MIN:-30} FAIL_REGEX='LATENCY_FAIL' bash "${WATCH[@]}"; rc=$?
for f in latency.json latency.txt; do [ -s "$OUT/latency/$f" ] && cp -f "$OUT/latency/$f" "$OUT/$f"; done
[ -s "$OUT/latency.txt" ] && cat "$OUT/latency.txt"
exit $rc
