#!/bin/bash
# Mac side of BIRD dev look #1 (runs/phase1/devlook/FROZEN.md): stage data/devlook/r1 (blind), wait for a box, push code
# + the blind inputs (push6 runs leakcheck on every path; prep.py audit refuses any label field first), start
# devlook_box.sh detached, then watch (watch_box.sh pulls /workspace/devlook + logs, then deletes THIS box only, verified).
#   DRY=1 VARIANT=A bash scripts/chains/run_devlook_box.sh            # stage + audit + leakcheck + push plan; no ssh, no spend
#   VARIANT=A CAP=4  bash scripts/chains/run_devlook_box.sh <id>      # score R1's own k=32 pool with VALHARD
#   VARIANT=B CAP=10 bash scripts/chains/run_devlook_box.sh <id>      # k=64 + repair (blind), merged with A, then score
# Results -> runs/phase1/devlook/{devlook,logs}/; then, on the Mac, ONCE:
#   .venv/bin/python scripts/analysis/devlook_score.py --scored runs/phase1/devlook/devlook/pool<V>_by_valhard.jsonl --variant <V>
# Box (H100, Shadeform; created first BY HAND, never by this script):
#   python3 scripts/shade/sf.py offers H100 --n 5
#   python3 scripts/shade/sf.py create <cloud> <region> <type> --max-spend 12 --name luthor-devlook
# What leaves the Mac: code (src, scripts, pyproject.toml) and data/devlook/r1 = {poolA.jsonl (SQL + execution facts),
# qs/ (questions + evidence + schema cards; no plan, no SQL), rr_valhard/ (VALHARD adapter + head)}; B also the R1
# checkpoint and the 11 BIRD dev *.sqlite files (SQLITE_ONLY). Never dev.json, gold SQL, labels, scores or RESERVED.
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=${1:-}; DRY=${DRY:-0}; NAME=devlook; VARIANT=${VARIANT:-}
case "$VARIANT" in A|B) ;; *) echo "VARIANT=A or VARIANT=B required"; exit 2;; esac
[ -n "$ID" ] || [ "$DRY" = 1 ] || { echo "usage: VARIANT=A|B CAP=<\$> bash scripts/chains/run_devlook_box.sh <id>   (or DRY=1)"; exit 2; }
mkdir -p runs/phase1/$NAME
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME $VARIANT] $*" | tee -a runs/phase1/$NAME/watch.log; }
[ "$DRY" = 1 ] && log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME $VARIANT DRY] $*"; }
PY=.venv/bin/python
GEN=runs/r1r2/r1; DL=data/devlook/r1; VH=runs/phase1/valhard/valhard/rr
DEVDB=data/bench/dev_20240627/dev_databases
if [ "$DRY" != 1 ] && [ -e runs/phase1/$NAME/LOOK_USED ]; then
  echo "runs/phase1/$NAME/LOOK_USED exists: the one approved BIRD dev look is spent; not starting a box"; exit 1
fi
[ "$ID" ] && [ "$DRY" != 1 ] && [ -z "${CAP:-}" ] && { echo "CAP= dollars required (watch_box hard stop)"; exit 2; }
for f in $VH/rerank_head.pt $VH/rerank_lora/rerank/adapter_model.safetensors runs/phase1/p1w/p1w/bird_dev_union.jsonl \
         data/sys1/bird_dev/plans.jsonl data/bench/dev_20240627/dev.json; do
  [ -e "$f" ] || { log "missing $f: not starting"; exit 1; }
done
if [ "$VARIANT" = B ]; then
  for f in $GEN/best_lora/adapter_model.safetensors $GEN/best_heads.pt $GEN/args.json $GEN/vocab.json; do
    [ -e "$f" ] || { log "missing $f: not starting"; exit 1; }
  done
  [ "$(find $DEVDB -name '*.sqlite' | wc -l | tr -d ' ')" = 11 ] || { log "expected 11 sqlite files under $DEVDB"; exit 1; }
fi

# ---- stage data/devlook/r1 (blind copies; prep.py prints counts only) ----------------------------------------------
$PY scripts/devlook/prep.py pool-a > /dev/null || { log "pool-a failed"; exit 1; }
$PY scripts/devlook/prep.py questions | grep -q "order vs poolA.jsonl: SAME" || { log "questions: order differs from pool A"; exit 1; }
mkdir -p $DL/rr_valhard/rerank_lora/rerank
cp -f $VH/rerank_head.pt $DL/rr_valhard/
cp -f $VH/rerank_lora/rerank/adapter_config.json $VH/rerank_lora/rerank/adapter_model.safetensors $DL/rr_valhard/rerank_lora/rerank/
cp -f $VH/settings.json $DL/rr_valhard/
$PY scripts/devlook/prep.py audit $DL/poolA.jsonl $DL/qs/plans.jsonl || { log "audit failed: nothing sent"; exit 1; }
log "staged $DL: $(du -sh $DL | cut -f1) ($(wc -l < $DL/poolA.jsonl | tr -d ' ') questions)"
# code + inputs this look runs on, hashed (FROZEN.md cites this file)
{ date -u '+%Y-%m-%d %H:%M:%S UTC'; echo "VARIANT=$VARIANT K=${K:-64}"
  shasum -a 256 scripts/devlook/*.py scripts/chains/devlook_box.sh scripts/chains/run_devlook_box.sh scripts/rerank_sq2.py \
    scripts/train_sq2.py scripts/eval_sq2.py scripts/repair/repair_gen.py scripts/repair/evidence.py \
    scripts/analysis/devlook_score.py $DL/poolA.jsonl $DL/qs/plans.jsonl $DL/rr_valhard/rerank_head.pt \
    $DL/rr_valhard/rerank_lora/rerank/adapter_model.safetensors $GEN/best_heads.pt $GEN/best_lora/adapter_model.safetensors \
    $GEN/args.json $GEN/vocab.json 2>/dev/null
  git rev-parse HEAD 2>/dev/null; } > runs/phase1/$NAME/launch_hashes_$VARIANT.txt

# ---- what goes where ("REMOTE_D|SQLITE_ONLY|paths") ------------------------------------------------------------------
pushes=("/root/d/devlook|0|$DL")
[ "$VARIANT" = B ] && pushes+=("/workspace/luthor/$GEN|0|$GEN/best_lora $GEN/best_heads.pt $GEN/args.json $GEN/vocab.json"
                               "/root/d|1|$DEVDB")
all=(src scripts pyproject.toml)
for p in "${pushes[@]}"; do read -r -a ps <<< "${p##*|}"; all+=("${ps[@]}"); done
bash scripts/vast/leakcheck.sh "${all[@]}" > /dev/null || { bash scripts/vast/leakcheck.sh "${all[@]}" | tail -n 20; log "leakcheck failed: nothing sent"; exit 1; }
log "leakcheck ok: ${#all[@]} paths, $(du -shc "${all[@]}" 2>/dev/null | tail -1 | cut -f1) on disk"
if [ "$DRY" = 1 ]; then
  for p in "${pushes[@]}"; do
    read -r -a ps <<< "${p##*|}"; so=$(echo "$p" | cut -d'|' -f2)
    log "would push -> ${p%%|*} (SQLITE_ONLY=$so): ${ps[*]} ($(du -shc "${ps[@]}" | tail -1 | cut -f1))"
  done
  log "would start: VARIANT=$VARIANT setsid nohup bash /workspace/luthor/scripts/chains/devlook_box.sh > /workspace/logs/devlook.log"
  log "would watch: STALL_MIN=${STALL_MIN:-60} FAIL_REGEX=DEVLOOK_FAIL CAP=\$CAP watch_box.sh <id> $NAME /workspace/logs/devlook.log DEVLOOK_DONE /workspace/devlook"
  exit 0
fi

for i in $(seq 1 240); do
  REFRESH=1 bash scripts/vast/ssh.sh "$ID" true >/dev/null 2>&1 && break
  [ "$i" = 240 ] && { log "ssh never came up: deleting $ID"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
  sleep 15
done
log "ssh up"
ok=0
for t in 1 2 3; do
  good=1
  for p in "${pushes[@]}"; do
    read -r -a ps <<< "${p##*|}"; so=$(echo "$p" | cut -d'|' -f2)
    if [ "$so" = 1 ]; then
      SQLITE_ONLY=1 REMOTE_D=${p%%|*} bash scripts/vast/push6.sh "$ID" "${ps[@]}" >> runs/phase1/$NAME/push.log 2>&1 || { good=0; break; }
    else
      REMOTE_D=${p%%|*} bash scripts/vast/push6.sh "$ID" "${ps[@]}" >> runs/phase1/$NAME/push.log 2>&1 || { good=0; break; }
    fi
  done
  [ $good = 1 ] && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting $ID"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (VARIANT='$VARIANT' K='${K:-64}' setsid nohup bash /workspace/luthor/scripts/chains/devlook_box.sh > /workspace/logs/devlook.log 2>&1 < /dev/null &)" 2>/dev/null
log "started"
STALL_MIN=${STALL_MIN:-60} FAIL_REGEX='DEVLOOK_FAIL' exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/devlook.log DEVLOOK_DONE /workspace/devlook
