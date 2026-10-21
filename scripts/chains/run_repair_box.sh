#!/bin/bash
# Mac side of the System-1 repair stage (runs/phase1/repair/PREREG.md): stage data/repair/r1, wait for a box, push code
# + the R1 generator checkpoint + held-out data + the two validators (push6 runs leakcheck on every path), start
# repair_box.sh detached, then watch (watch_box.sh pulls /workspace/repair + logs, then deletes the box, verified).
#   DRY=1 bash scripts/chains/run_repair_box.sh                 # stage + leakcheck + print the push plan; no ssh, no spend
#   CAP=10 bash scripts/chains/run_repair_box.sh <id>           # results -> runs/phase1/repair/{repair,logs}/
# Box (H100, Shadeform; created first by hand):
#   python3 scripts/shade/sf.py offers H100 --n 5
#   python3 scripts/shade/sf.py create <cloud> <region> <type> --max-spend 12 --name luthor-repair
# Generator: R1 only (runs/r1r2/r1: best_lora + best_heads.pt + args.json + vocab.json, grammar v3; on the Mac, no
#   retrain or download). Validators (copied into data/repair/r1 so leakcheck's allowlist covers them):
#   rr_valhard <- runs/phase1/valhard/valhard/rr   (VALHARD, DECISION 59.05 on R1's k=32 pool)
#   rr_r1w     <- runs/phase1/p1w/p1w/r1/rr        (R1 wide validator, DECISION 57.29)
#   k32_pool.jsonl <- data/valhard/r1/heldout_pool.jsonl (R1's own k=32 held-out list, no scores)
# Nothing from BIRD dev, sealed or gate is staged or pushed (no data/sys1/bird_dev, no dev_databases).
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=${1:-}; DRY=${DRY:-0}; NAME=repair
[ -n "$ID" ] || [ "$DRY" = 1 ] || { echo "usage: CAP=10 bash scripts/chains/run_repair_box.sh <id>   (or DRY=1)"; exit 2; }
mkdir -p runs/phase1/$NAME
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a runs/phase1/$NAME/watch.log; }
[ "$DRY" = 1 ] && log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME DRY] $*"; }
GEN=runs/r1r2/r1; RP=data/repair/r1
VH=runs/phase1/valhard/valhard/rr; R1W=runs/phase1/p1w/p1w/r1/rr; K32SRC=data/valhard/r1/heldout_pool.jsonl
for f in $GEN/best_lora/adapter_model.safetensors $GEN/best_heads.pt $GEN/args.json $GEN/vocab.json \
         $VH/rerank_head.pt $VH/rerank_lora/rerank/adapter_model.safetensors \
         $R1W/rerank_head.pt $R1W/rerank_lora/rerank/adapter_model.safetensors "$K32SRC" \
         data/sys1/bird_train/cards data/sys1/dbs data/sys1/dev_dbs_bird.json \
         data/sys1/heldout_v2/decision.json data/sys1/heldout_v2/reserved.json; do
  [ -e "$f" ] || { log "missing $f: not starting"; exit 1; }
done
[ "$ID" ] && [ "$DRY" != 1 ] && [ -z "${CAP:-}" ] && { echo "CAP= dollars required (watch_box hard stop)"; exit 2; }

# ---- stage data/repair/r1 (copies; only the files the box reads: adapters + head + settings, the k=32 pool) --------
stage_rr() {  # stage_rr SRC DST
  mkdir -p "$2/rerank_lora/rerank"
  cp -f "$1/rerank_head.pt" "$2/"
  cp -f "$1/rerank_lora/rerank/adapter_config.json" "$1/rerank_lora/rerank/adapter_model.safetensors" "$2/rerank_lora/rerank/"
  [ -f "$1/settings.json" ] && cp -f "$1/settings.json" "$2/"
  return 0
}
mkdir -p $RP
stage_rr $VH $RP/rr_valhard
stage_rr $R1W $RP/rr_r1w
cmp -s "$K32SRC" $RP/k32_pool.jsonl || cp -f "$K32SRC" $RP/k32_pool.jsonl
log "staged $RP: $(du -sh $RP | cut -f1) ($(wc -l < $RP/k32_pool.jsonl | tr -d ' ') k=32 questions)"

# ---- what goes where ("REMOTE_D|paths") ----------------------------------------------------------------------------
pushes=("/root/d/sys1|data/sys1/bird_train data/sys1/dbs data/sys1/dev_dbs_bird.json data/sys1/heldout_v2"
        "/workspace/luthor/$GEN|$GEN/best_lora $GEN/best_heads.pt $GEN/args.json $GEN/vocab.json"
        "/root/d/repair|$RP")
all=(src scripts pyproject.toml)
for p in "${pushes[@]}"; do read -r -a ps <<< "${p#*|}"; all+=("${ps[@]}"); done
bash scripts/vast/leakcheck.sh "${all[@]}" > /dev/null || { bash scripts/vast/leakcheck.sh "${all[@]}" | tail -n 20; log "leakcheck failed: nothing sent"; exit 1; }
log "leakcheck ok: ${#all[@]} paths, $(du -shc "${all[@]}" 2>/dev/null | tail -1 | cut -f1) on disk"
if [ "$DRY" = 1 ]; then
  for p in "${pushes[@]}"; do
    read -r -a ps <<< "${p#*|}"
    log "would push -> ${p%%|*}: ${ps[*]} ($(du -shc "${ps[@]}" | tail -1 | cut -f1))"
  done
  log "would start: SCORE_K64=${SCORE_K64:-1} setsid nohup bash /workspace/luthor/scripts/chains/repair_box.sh > /workspace/logs/repair.log"
  log "would watch: STALL_MIN=${STALL_MIN:-60} FAIL_REGEX=REPAIR_FAIL CAP=\$CAP watch_box.sh <id> $NAME /workspace/logs/repair.log REPAIR_DONE /workspace/repair"
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
    REMOTE_D=${p%%|*} bash scripts/vast/push6.sh "$ID" "${ps[@]}" >> runs/phase1/$NAME/push.log 2>&1 || { good=0; break; }
  done
  [ $good = 1 ] && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (SCORE_K64='${SCORE_K64:-1}' REPAIR_ARGS='${REPAIR_ARGS:-}' setsid nohup bash /workspace/luthor/scripts/chains/repair_box.sh > /workspace/logs/repair.log 2>&1 < /dev/null &)" 2>/dev/null
log "started"
STALL_MIN=${STALL_MIN:-60} FAIL_REGEX='REPAIR_FAIL' exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/repair.log REPAIR_DONE /workspace/repair
