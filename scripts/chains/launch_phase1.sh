#!/bin/bash
# Mac side, Phase 1: push code + data + the R1/R2 checkpoints to ONE existing GPU box, for the head-dump stage
# (scripts/chains/dump_box.sh) and, with RL=1, the RL timing test / held-out eval (scripts/rl/). Never creates,
# stops or deletes an instance: HOST is the id of a box that is already up (Vast numeric id or Shadeform UUID, as
# scripts/vast/ssh.sh takes it).
#   bash scripts/chains/launch_phase1.sh <HOST>              push only
#   DRY=1 bash scripts/chains/launch_phase1.sh <HOST>        leakcheck every path + print sizes and the push commands
#   START=1 bash scripts/chains/launch_phase1.sh <HOST>      push, then start dump_box.sh detached on the box
# Every push goes through scripts/vast/push6.sh (allowlist + content scan in leakcheck.sh, size dry run, > 20 GB
# needs CONFIRM=yes). Box layout (what dump_box.sh reads):
#   code                         -> /workspace/luthor/{src,scripts,pyproject.toml}
#   runs/r1r2/<role>/{best_lora,best_heads.pt,args.json,vocab.json}  -> /workspace/luthor/runs/r1r2/<role>/
#   runs/r1r2/<role>/rr/cands_{dev,bird_dev[,train]}.jsonl           -> /workspace/luthor/runs/r1r2/<role>/rr/
#   data/sys1/{bird_train,bird_dev,dbs,dev_dbs_bird.json,bird_dev_dbs.json,contaminated_dbs.json} -> /root/d/sys1/
#   data/bench/dev_20240627/{dev_databases,dev.json} -> /root/d/  (original BIRD dev: evaluation only)
# Env: ROLES ("r1 r2"), TRAIN_Q (0; > 0 also pushes data/sys1/rrdbs and rr/cands_train.jsonl, as dump_box.sh needs),
#   EXTRA=1 (also data/plans_v4 -> /root/d/plans_v4, data/teacher/export -> /root/d/teacher/export,
#   data/sys1/heldout_v2 -> /root/d/sys1/heldout_v2, each when present), RL=1 (also data/sys1/rrdbs and R2's frozen
#   validator rr/{rerank_head.pt,rerank_lora,dev_metrics.json}, for scripts/rl/heldout_eval.sh), START=1, DRY=1,
#   and dump_box.sh's env when START=1 (ROLES TRAIN_Q KBEST FLIPS_K STATE_DTYPE LUTHOR_GRAMMAR SKIP_SETUP).
# Sealed sets, docs/, reports/ and every other runs/ file are never pushed (leakcheck refuses them anyway).
set -euo pipefail
cd "$(dirname "$0")/../.."
ID=${1:?usage: launch_phase1.sh <HOST: instance id of a running box>}
ROLES=${ROLES:-r1 r2}; TRAIN_Q=${TRAIN_Q:-0}; DRY=${DRY:-0}
BOX_REPO=${REMOTE_ROOT:-/workspace}/luthor
log() { echo "$(date -u +%H:%M:%S) [phase1 $ID] $*"; }

# ---- what goes where: "REMOTE_D|local path..." per push6 call -------------------------------------------------------
pushes=()
SYS1=(data/sys1/bird_train data/sys1/bird_dev data/sys1/dbs data/sys1/dev_dbs_bird.json data/sys1/bird_dev_dbs.json
      data/sys1/contaminated_dbs.json)
{ [ "$TRAIN_Q" -gt 0 ] || [ "${RL:-0}" = 1 ]; } && SYS1+=(data/sys1/rrdbs)
[ "${EXTRA:-0}" = 1 ] && [ -e data/sys1/heldout_v2 ] && SYS1+=(data/sys1/heldout_v2)
pushes+=("/root/d/sys1|${SYS1[*]}")
pushes+=("/root/d|data/bench/dev_20240627/dev_databases data/bench/dev_20240627/dev.json")
for role in $ROLES; do
  case "$role" in r1|r2) ;; *) echo "ROLES: r1 and/or r2 only (the leakcheck allowlist), not $role"; exit 2;; esac
  run=runs/r1r2/$role
  for f in best_lora/adapter_model.safetensors best_heads.pt args.json vocab.json rr/cands_dev.jsonl rr/cands_bird_dev.jsonl; do
    [ -e "$run/$f" ] || { echo "missing local checkpoint file $run/$f"; exit 1; }
  done
  pushes+=("$BOX_REPO/$run|$run/best_lora $run/best_heads.pt $run/args.json $run/vocab.json")
  rr="$run/rr/cands_dev.jsonl $run/rr/cands_bird_dev.jsonl"
  if [ "$TRAIN_Q" -gt 0 ]; then
    [ -e "$run/rr/cands_train.jsonl" ] || { echo "TRAIN_Q > 0 needs $run/rr/cands_train.jsonl"; exit 1; }
    rr+=" $run/rr/cands_train.jsonl"
  fi
  if [ "${RL:-0}" = 1 ] && [ "$role" = r2 ]; then rr+=" $run/rr/rerank_head.pt $run/rr/rerank_lora $run/rr/dev_metrics.json"; fi
  pushes+=("$BOX_REPO/$run/rr|$rr")
done
if [ "${EXTRA:-0}" = 1 ]; then
  [ -e data/plans_v4 ] && pushes+=("/root/d|data/plans_v4")
  [ -e data/teacher/export ] && pushes+=("/root/d/teacher|data/teacher/export")
fi

# ---- one leakcheck over everything first (push6 checks each call again) ----------------------------------------------
all=(src scripts pyproject.toml)
for p in "${pushes[@]}"; do read -r -a ps <<< "${p#*|}"; all+=("${ps[@]}"); done
for p in "${all[@]}"; do [ -e "$p" ] || { echo "missing local path: $p"; exit 1; }; done
bash scripts/vast/leakcheck.sh "${all[@]}" > /dev/null || { bash scripts/vast/leakcheck.sh "${all[@]}" | tail -n 20; exit 1; }
log "leakcheck ok: ${#all[@]} paths, $(du -shc "${all[@]}" 2>/dev/null | tail -1 | cut -f1) on disk"

for p in "${pushes[@]}"; do
  dst=${p%%|*}; read -r -a ps <<< "${p#*|}"
  if [ "$DRY" = 1 ]; then
    log "DRY: REMOTE_D=$dst bash scripts/vast/push6.sh $ID ${ps[*]}   ($(du -shc "${ps[@]}" | tail -1 | cut -f1))"
  else
    log "push -> $dst: ${ps[*]}"
    REMOTE_D=$dst bash scripts/vast/push6.sh "$ID" "${ps[@]}"
  fi
done
[ "$DRY" = 1 ] && { log "DRY: nothing sent"; exit 0; }
log "PUSHED_PHASE1"

if [ "${START:-0}" = 1 ]; then
  envs=""
  for v in ROLES TRAIN_Q KBEST FLIPS_K STATE_DTYPE LUTHOR_GRAMMAR SKIP_SETUP; do
    if [ -n "${!v+x}" ]; then envs+="$v=$(printf %q "${!v}") "; fi
  done
  if bash scripts/vast/ssh.sh "$ID" 'pgrep -f "[d]ump_box.sh" >/dev/null && echo RUNNING' 2>/dev/null | grep -q RUNNING; then
    log "dump_box.sh is already running on this box: not starting another"; exit 1
  fi
  log "starting dump_box.sh: $envs"
  bash scripts/vast/ssh.sh "$ID" "mkdir -p /workspace/logs && (${envs} setsid nohup bash $BOX_REPO/scripts/chains/dump_box.sh \
    >> /workspace/logs/dump.log 2>&1 < /dev/null &)" 2>/dev/null
  sleep 10
  bash scripts/vast/ssh.sh "$ID" 'tail -n 5 /workspace/logs/dump.log' 2>/dev/null | grep -av Welcome || true
  log "STARTED dump_box.sh"
fi
