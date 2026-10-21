#!/bin/bash
#SBATCH --job-name=luthor-sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/hpc/%x-%j.out
# System-1 SFT: scripts/train_sq2.py with the R1/R2 chain's recipe (scripts/chains/r1r2_chain.sh stage 2), on one
# GPU or on GPUS>1 GPUs of one node (torchrun data parallelism: --dbs-per-step stays the GLOBAL batch, split across
# the GPUs, so the step count, LR schedule and every update are the single-GPU run's). Submit with submit.sh.
#
# Resumable across queue time limits: --auto-resume keeps runs/hpc/$RUN/resume (every eval and every SAVE_EVERY
# steps); --job-hours stops the job JOB_MARGIN_H before the wall-time limit with the state saved, then this script
# re-submits itself (up to MAX_RESUBMITS times) and the next job continues the same step schedule.
#
# Env: RUN (required: output runs/hpc/$RUN), GPUS (1), STEPS (6714 = R1; R2 = 13428), RARE (a rare-skeleton plan dir
#   to add to --plans, e.g. $LUTHOR_DATA/rare1), EXTRA_TRAIN ("PATH[:W] ..." -> --extra-train each), SEED (--seed),
#   GRAMMAR (--grammar v3|v4; v4 also needs TRAIN_ARGS with data/plans_v4 plans and --vocab .../templates_union.json),
#   DROP_DBS (--drop-dbs, e.g. $LUTHOR_DATA/sys1/contaminated_dbs.json),
#   TRAIN_EXTRA (more train_sq2.py flags), TRAIN_ARGS (replaces the whole recipe), SAVE_EVERY (250),
#   JOB_MARGIN_H (0.5), RESUBMIT_N (set by the resubmission).
source scripts/hpc/common.sh
RUN=${RUN:?RUN=<name>}; O=$REPO/runs/hpc/$RUN; GPUS=${GPUS:-1}; STEPS=${STEPS:-6714}
mkdir -p "$O"
need "$S/bird_train" "$S/spider_train" "$S/synsql_train" "$S/dbs" "$S/bird_dev" "$S/templates.json" \
     "$S/dev_dbs_bird.json" "$D/dev_databases"
# hours left in this job (squeue %L: [D-]HH:MM:SS or MM:SS) minus the margin for the state save
left=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null || true)
JOB_H=$(python3 - "${left:-}" "${JOB_MARGIN_H:-0.5}" <<'PY'
import sys
s, margin = sys.argv[1].strip(), float(sys.argv[2])
d, _, hms = s.rpartition("-")
p = [int(x) for x in hms.split(":")] if hms and hms[0].isdigit() else []
p = [0] * (3 - len(p)) + p if p else []
h = (int(d or 0) * 24 + p[0] + p[1] / 60 + p[2] / 3600) if p else 0
print(f"{max(h - margin, 0.25):.3f}" if h else "0")
PY
)
extra=()
for e in ${EXTRA_TRAIN:-}; do extra+=(--extra-train "$e"); done
if [ -n "${TRAIN_ARGS:-}" ]; then read -r -a recipe <<< "$TRAIN_ARGS"
else
  recipe=(--plans "$S/bird_train" "$S/spider_train" "$S/synsql_train" ${RARE:+"$RARE"} --db-roots "$S/dbs" "$D/dev_databases"
    --base seeklhy/OmniSQL-7B --subfolder "" --vocab "$S/templates.json" --dev-dbs "$S/dev_dbs_bird.json"
    --eval-every 1000 --budget-chars 9000 --gate "$S/bird_dev" --gate-total 1534 --col-mentions --repeat bird_train=3
    --max-steps "$STEPS" --cand-order v2 --value-side)
fi
# shellcheck disable=SC2206
flags=("${recipe[@]}" "${extra[@]}" ${SEED:+--seed "$SEED"} ${GRAMMAR:+--grammar "$GRAMMAR"} ${DROP_DBS:+--drop-dbs "$DROP_DBS"} --out "$O" --auto-resume --save-every "${SAVE_EVERY:-250}"
       ${TRAIN_EXTRA:-})
[ "$JOB_H" != 0 ] && flags+=(--job-hours "$JOB_H")
if [ "$GPUS" -gt 1 ]; then
  cmd=(torchrun --standalone --nproc_per_node="$GPUS" scripts/train_sq2.py "${flags[@]}")
else
  cmd=(python scripts/train_sq2.py "${flags[@]}")
fi
log "job hours $JOB_H (left: ${left:-?}); run: ${cmd[*]}"
rc=0; "${cmd[@]}" || rc=$?
if [ "$rc" = 0 ] && [ -e "$O/job_stopped" ]; then
  n=${RESUBMIT_N:-0}
  if [ "$n" -ge "${MAX_RESUBMITS:-12}" ]; then log "job stopped but MAX_RESUBMITS ($n) reached: resubmit by hand"; exit 1; fi
  log "job stopped at $(cat "$O/job_stopped"): resubmitting ($((n + 1)))"
  RESUBMIT_N=$((n + 1)) bash scripts/hpc/submit.sh sft.sbatch
  exit 0
fi
[ "$rc" = 0 ] && [ -d "$O/best_lora" ] && touch "$O/train.ok" && log "DONE_TRAIN: $O" || log "train_sq2.py exited $rc"
exit "$rc"
