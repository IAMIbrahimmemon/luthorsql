#!/bin/bash
# Mac side of v4guard (runs/phase1/v4guard/PREREG.md): stage data/v4guard, wait for a box, push code + the v4desc
# generator checkpoint + its own validator + VALHARD-v4desc + held-out data (push6 runs leakcheck on every path), start
# v4guard_box.sh detached, then watch (watch_box.sh pulls /workspace/v4guard + logs, then deletes ONLY instance <id>).
#   DRY=1 bash scripts/chains/run_v4guard_box.sh                 # stage + leakcheck + print the push plan; no ssh, no spend
#   CAP=6 bash scripts/chains/run_v4guard_box.sh <id>            # results -> runs/phase1/v4guard/{v4guard,logs}/
# Box (one H100, Shadeform; created first BY HAND, by the owner):
#   python3 scripts/shade/sf.py offers H100 --n 5
#   python3 scripts/shade/sf.py create <cloud> <region> <type> --max-spend 6 --name luthor-v4guard
# Generator: phase-3 v4desc (runs/phase1/v4desc/r1: best_lora + best_heads.pt + args.json + vocab.json, grammar v4,
#   --bird-desc; on the Mac, no retrain). Copied into data/v4guard (leakcheck's allowlist covers data/v4guard only):
#   gen/        <- runs/phase1/v4desc/r1/{best_lora,best_heads.pt,args.json,vocab.json}
#   rr_own/     <- runs/phase1/v4desc/r1/rr                  (v4desc's own validator: DECISION 52.51 on own k=8)
#   rr_valhard/ <- runs/phase1/valhard_v4desc/valhard/rr     (VALHARD-v4desc, best epoch; 58.17 on v4desc's k=32)
#   k8_off.jsonl <- runs/phase1/v4desc/r1/rr/cands_dev.jsonl (v4desc's own unguarded k=8 held-out list, no scores)
# Nothing from BIRD dev, sealed or gate is staged or pushed (no data/sys1/bird_dev, no dev_databases, no dev.json).
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=${1:-}; DRY=${DRY:-0}; NAME=v4guard
[ -n "$ID" ] || [ "$DRY" = 1 ] || { echo "usage: CAP=6 bash scripts/chains/run_v4guard_box.sh <id>   (or DRY=1)"; exit 2; }
mkdir -p runs/phase1/$NAME
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a runs/phase1/$NAME/watch.log; }
[ "$DRY" = 1 ] && log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME DRY] $*"; }
[ -f runs/phase1/$NAME/PREREG.md ] || { log "no runs/phase1/$NAME/PREREG.md: not starting"; exit 1; }
GEN=runs/phase1/v4desc/r1; OWN=$GEN/rr; VH=runs/phase1/valhard_v4desc/valhard/rr; K8=$GEN/rr/cands_dev.jsonl
GD=data/v4guard
for f in $GEN/best_lora/adapter_model.safetensors $GEN/best_lora/adapter_config.json $GEN/best_heads.pt $GEN/args.json \
         $GEN/vocab.json $OWN/rerank_head.pt $OWN/rerank_lora/rerank/adapter_model.safetensors \
         $VH/rerank_head.pt $VH/rerank_lora/rerank/adapter_model.safetensors "$K8" \
         data/plans_v4/train/bird_train/cards data/plans_v4/train/bird_train/plans.jsonl data/raw/bird_train_desc \
         data/sys1/dbs data/sys1/dev_dbs_bird.json; do
  [ -e "$f" ] || { log "missing $f: not starting"; exit 1; }
done
python3 -c "import json,sys; a=json.load(open('$GEN/args.json')); sys.exit(0 if a.get('grammar')=='v4' and a.get('bird_desc') else 1)" \
  || { log "$GEN/args.json is not a grammar-v4 --bird-desc run: not starting"; exit 1; }
[ "$ID" ] && [ "$DRY" != 1 ] && [ -z "${CAP:-}" ] && { echo "CAP= dollars required (watch_box hard stop)"; exit 2; }

# ---- stage data/v4guard (copies; only the files the box reads) --------------------------------------------------
stage_rr() {  # stage_rr SRC DST: adapter + head + settings only
  mkdir -p "$2/rerank_lora/rerank"
  cp -f "$1/rerank_head.pt" "$2/"
  cp -f "$1/rerank_lora/rerank/adapter_config.json" "$1/rerank_lora/rerank/adapter_model.safetensors" "$2/rerank_lora/rerank/"
  [ -f "$1/settings.json" ] && cp -f "$1/settings.json" "$2/"
  return 0
}
mkdir -p $GD/gen/best_lora
cp -f $GEN/best_lora/adapter_config.json $GEN/best_lora/adapter_model.safetensors $GD/gen/best_lora/
cp -f $GEN/best_heads.pt $GEN/args.json $GEN/vocab.json $GD/gen/
stage_rr $OWN $GD/rr_own
stage_rr $VH $GD/rr_valhard
cmp -s "$K8" $GD/k8_off.jsonl || cp -f "$K8" $GD/k8_off.jsonl
for f in $GD/gen/best_heads.pt $GD/gen/best_lora/adapter_model.safetensors $GD/rr_own/rerank_head.pt $GD/rr_valhard/rerank_head.pt; do
  case "$f" in
    $GD/gen/best_lora/*) src=$GEN/best_lora/$(basename "$f");; $GD/gen/*) src=$GEN/$(basename "$f");;
    $GD/rr_own/*) src=$OWN/$(basename "$f");; *) src=$VH/$(basename "$f");;
  esac
  cmp -s "$f" "$src" || { log "staged copy $f differs from $src: not starting"; exit 1; }
done
log "staged $GD: $(du -sh $GD | cut -f1) ($(wc -l < $GD/k8_off.jsonl | tr -d ' ') k=8 questions)"

# ---- what goes where ("REMOTE_D|paths") ----------------------------------------------------------------------------
pushes=("/root/d/sys1|data/sys1/dbs data/sys1/dev_dbs_bird.json"
        "/root/d/plans_v4|data/plans_v4/train/bird_train"
        "/root/d|data/raw/bird_train_desc $GD")
all=(src scripts pyproject.toml)
for p in "${pushes[@]}"; do read -r -a ps <<< "${p#*|}"; all+=("${ps[@]}"); done
bash scripts/vast/leakcheck.sh "${all[@]}" > /dev/null || { bash scripts/vast/leakcheck.sh "${all[@]}" | tail -n 20; log "leakcheck failed: nothing sent"; exit 1; }
log "leakcheck ok: ${#all[@]} paths, $(du -shc "${all[@]}" 2>/dev/null | tail -1 | cut -f1) on disk"
START="mkdir -p /workspace/logs && (REPRO_GEN='${REPRO_GEN:-1}' REPRO_SCORE='${REPRO_SCORE:-1}' setsid nohup bash /workspace/luthor/scripts/chains/v4guard_box.sh > /workspace/logs/v4guard.log 2>&1 < /dev/null &)"
if [ "$DRY" = 1 ]; then
  for p in "${pushes[@]}"; do
    read -r -a ps <<< "${p#*|}"
    log "would push -> ${p%%|*}: ${ps[*]} ($(du -shc "${ps[@]}" | tail -1 | cut -f1))"
  done
  log "would start: $START"
  log "would watch: STALL_MIN=${STALL_MIN:-45} FAIL_REGEX=V4GUARD_FAIL CAP=\$CAP watch_box.sh <id> $NAME /workspace/logs/v4guard.log V4GUARD_DONE /workspace/v4guard"
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
    read -r -a ps <<< "${p#*|}"
    REMOTE_D=${p%%|*} bash scripts/vast/push6.sh "$ID" "${ps[@]}" >> runs/phase1/$NAME/push.log 2>&1 || { good=0; break; }
  done
  [ $good = 1 ] && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting $ID"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
bash scripts/vast/ssh.sh "$ID" "$START" 2>/dev/null
log "started"
STALL_MIN=${STALL_MIN:-45} FAIL_REGEX='V4GUARD_FAIL' exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/v4guard.log V4GUARD_DONE /workspace/v4guard
