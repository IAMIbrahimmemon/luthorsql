#!/bin/bash
# Mac side of VAL3 data v3 list generation: check the staged sets, push code + the R1/R2 generator checkpoints + the
# BIRD-train rrdbs + data/val3/{q,dbs,big_dbs.json} (push6 runs leakcheck on every path), start val3_lists_box.sh
# detached, then watch (watch_box.sh pulls /workspace/val3 + logs, then deletes the box, verified).
#   .venv/bin/python scripts/val3/sources.py                        # first, on the Mac (CPU, ~1 min)
#   DRY=1 bash scripts/chains/run_val3_lists_box.sh                  # leakcheck + print the push plan; no ssh, no spend
#   CAP=8 bash scripts/chains/run_val3_lists_box.sh <id>             # results -> runs/phase1/val3_lists/{val3,logs}/
#   then: .venv/bin/python scripts/val3/build.py                     # Mac: lists (see scripts/val3/build.py)
# Box: one H100 (80 GB), >= 100 GB disk (the big-DB archive needs ~45 GB at its peak; BIG=0 skips it).
# Env passed to the box: BIG (1), BIG_WAIT_MIN (90), BIG_MAX_MIN (100), R2_GREEDY (1), K (32).
# Nothing held-out, BIRD dev or sealed is staged or pushed (no data/sys1/dbs, no dev_databases, no data/eval).
set -uo pipefail
cd "$(dirname "$0")/../.."
ID=${1:-}; DRY=${DRY:-0}; NAME=val3_lists
[ -n "$ID" ] || [ "$DRY" = 1 ] || { echo "usage: CAP=8 bash scripts/chains/run_val3_lists_box.sh <id>   (or DRY=1)"; exit 2; }
mkdir -p runs/phase1/$NAME
log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME] $*" | tee -a runs/phase1/$NAME/watch.log; }
[ "$DRY" = 1 ] && log() { echo "$(date -u '+%m-%d %H:%M:%S') [$NAME DRY] $*"; }
R1=runs/r1r2/r1; R2G=runs/r1r2/r2; VQ=data/val3
for f in $R1/best_lora/adapter_model.safetensors $R1/best_heads.pt $R1/args.json $R1/vocab.json \
         $R2G/best_lora/adapter_model.safetensors $R2G/best_heads.pt $R2G/args.json $R2G/vocab.json \
         $VQ/q/bird_rr/plans.jsonl $VQ/q/synsql/plans.jsonl $VQ/q/bird_big/plans.jsonl $VQ/q/meta.jsonl \
         $VQ/big_dbs.json $VQ/dbs/synsql data/sys1/rrdbs; do
  [ -e "$f" ] || { log "missing $f (run scripts/val3/sources.py first): not starting"; exit 1; }
done
[ "$ID" ] && [ "$DRY" != 1 ] && [ -z "${CAP:-}" ] && { echo "CAP= dollars required (watch_box hard stop)"; exit 2; }
# every DB the sets name must be in what is pushed (the box checks again before setup)
python3 - <<'PY' || { log "staged DBs incomplete: not starting"; exit 1; }
import json, pathlib, sys
roots = {"bird_rr": "data/sys1/rrdbs", "synsql": "data/val3/dbs/synsql"}
bad = 0
for s, root in roots.items():
    have = {p.stem for p in pathlib.Path(root).rglob("*.sqlite")}
    miss = set(json.load(open(f"data/val3/q/{s}/dbs.json"))) - have
    bad += len(miss)
    print(f"{s}: {len(miss)} sqlite missing")
sys.exit(1 if bad else 0)
PY

# ---- what goes where ("REMOTE_D|SQLITE_ONLY|paths") ----------------------------------------------------------------
pushes=("/root/d/sys1|1|data/sys1/rrdbs"
        "/root/d/val3|0|$VQ/q $VQ/dbs $VQ/big_dbs.json"
        "/workspace/luthor/$R1|0|$R1/best_lora $R1/best_heads.pt $R1/args.json $R1/vocab.json"
        "/workspace/luthor/$R2G|0|$R2G/best_lora $R2G/best_heads.pt $R2G/args.json $R2G/vocab.json")
all=(src scripts pyproject.toml)
for p in "${pushes[@]}"; do r=${p#*|}; read -r -a ps <<< "${r#*|}"; all+=("${ps[@]}"); done
bash scripts/vast/leakcheck.sh "${all[@]}" > /dev/null || { bash scripts/vast/leakcheck.sh "${all[@]}" | tail -n 20; log "leakcheck failed: nothing sent"; exit 1; }
log "leakcheck ok: ${#all[@]} paths, $(du -shc "${all[@]}" 2>/dev/null | tail -1 | cut -f1) on disk (rrdbs sqlite only)"
BOXENV="BIG='${BIG:-1}' BIG_WAIT_MIN='${BIG_WAIT_MIN:-90}' BIG_MAX_MIN='${BIG_MAX_MIN:-100}' R2_GREEDY='${R2_GREEDY:-1}' K='${K:-32}'"
if [ "$DRY" = 1 ]; then
  for p in "${pushes[@]}"; do
    r=${p#*|}; read -r -a ps <<< "${r#*|}"
    log "would push -> ${p%%|*} (SQLITE_ONLY=${r%%|*}): ${ps[*]} ($(du -shc "${ps[@]}" | tail -1 | cut -f1))"
  done
  log "would start: $BOXENV setsid nohup bash /workspace/luthor/scripts/chains/val3_lists_box.sh > /workspace/logs/val3_lists.log"
  log "would watch: STALL_MIN=${STALL_MIN:-60} FAIL_REGEX=VAL3_LISTS_FAIL CAP=\$CAP watch_box.sh <id> $NAME /workspace/logs/val3_lists.log VAL3_LISTS_DONE /workspace/val3"
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
    r=${p#*|}; read -r -a ps <<< "${r#*|}"
    sq=""; [ "${r%%|*}" = 1 ] && sq=1
    SQLITE_ONLY=$sq REMOTE_D=${p%%|*} bash scripts/vast/push6.sh "$ID" "${ps[@]}" >> runs/phase1/$NAME/push.log 2>&1 || { good=0; break; }
  done
  [ $good = 1 ] && { ok=1; break; }
  log "push attempt $t failed"; sleep 30
done
[ $ok = 1 ] || { log "push failed: deleting"; CONFIRM=yes bash scripts/vast/destroy.sh "$ID"; exit 1; }
bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && ($BOXENV setsid nohup bash /workspace/luthor/scripts/chains/val3_lists_box.sh > /workspace/logs/val3_lists.log 2>&1 < /dev/null &)" 2>/dev/null
log "started"
STALL_MIN=${STALL_MIN:-60} FAIL_REGEX='VAL3_LISTS_FAIL' exec bash scripts/chains/watch_box.sh "$ID" "$NAME" /workspace/logs/val3_lists.log VAL3_LISTS_DONE /workspace/val3
