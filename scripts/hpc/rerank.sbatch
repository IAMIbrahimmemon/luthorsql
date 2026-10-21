#!/bin/bash
#SBATCH --job-name=luthor-rerank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/hpc/%x-%j.out
# K-best candidates + validator for a trained run (the R1/R2 chain's stage 3): scripts/rerank_sq2.py gen on BIRD-train
# (rrdbs, 9,000 q), the held-out training DBs and BIRD dev; validator training (1 h, 1 epoch); scored BIRD dev in
# runs/hpc/$RUN/rr/bird_dev_scored.jsonl. One GPU. Each step is skipped when its output exists, so resubmitting
# after a time limit continues with the next step (a step cut mid-way starts over).
# BIRD dev is generated and scored only: the validator trains on BIRD-train candidates, never on BIRD dev.
# Env: RUN (required: the SFT run, runs/hpc/$RUN), STEPS_TO_RUN ("gen_train gen_dev gen_bird_dev train score"),
#   DEVALL_FLAGS (default "--all-questions --questions $LUTHOR_DATA/dev.json"), K (8), GEN_EXTRA / RR_TRAIN_EXTRA.
source scripts/hpc/common.sh
RUN=${RUN:?RUN=<sft run name>}; O=$REPO/runs/hpc/$RUN; RD=$O/rr
need "$O/best_lora" "$O/args.json" "$S/bird_train" "$S/bird_dev" "$S/rrdbs" "$S/dbs" "$S/dev_dbs_bird.json" \
     "$S/bird_dev_dbs.json" "$D/dev_databases"
mkdir -p "$RD"
DEVALL_FLAGS=${DEVALL_FLAGS---all-questions --questions $D/dev.json}
TODO=" ${STEPS_TO_RUN:-gen_train gen_dev gen_bird_dev train score} "
G=(python scripts/rerank_sq2.py gen --run "$O" --k "${K:-8}" --workers "$W")
want() { [[ "$TODO" == *" $1 "* ]] && [ ! -s "$2" ]; }
run() { log "run: $*"; "$@"; }
# shellcheck disable=SC2086
want gen_train "$RD/cands_train.jsonl" && run "${G[@]}" --plans "$S/bird_train" --db-roots "$S/rrdbs" \
  --dev-dbs "$S/dev_dbs_bird.json" --split train --max-q 9000 --out "$RD/cands_train.jsonl" ${GEN_EXTRA:-}
# shellcheck disable=SC2086
want gen_dev "$RD/cands_dev.jsonl" && run "${G[@]}" --plans "$S/bird_train" --db-roots "$S/dbs" \
  --dev-dbs "$S/dev_dbs_bird.json" --split dev --out "$RD/cands_dev.jsonl" ${GEN_EXTRA:-}
# shellcheck disable=SC2086
want gen_bird_dev "$RD/cands_bird_dev.jsonl" && run "${G[@]}" --plans "$S/bird_dev" --db-roots "$D/dev_databases" \
  --dev-dbs "$S/bird_dev_dbs.json" --split dev $DEVALL_FLAGS --out "$RD/cands_bird_dev.jsonl" ${GEN_EXTRA:-}
# shellcheck disable=SC2086
want train "$RD/rerank_head.pt" && run python scripts/rerank_sq2.py train --cards "$S/bird_train" "$S/bird_dev" \
  --train-cands "$RD/cands_train.jsonl" --dev-cands "$RD/cands_dev.jsonl" --out "$RD" --base seeklhy/OmniSQL-7B \
  --subfolder "" --hours 1.0 --epochs 1 ${RR_TRAIN_EXTRA:-}
want score "$RD/bird_dev_scored.jsonl" && run python scripts/rerank_sq2.py score --rr "$RD" --cards "$S/bird_dev" \
  --dev-cands "$RD/cands_bird_dev.jsonl" --out "$RD/bird_dev_scored.jsonl" --base seeklhy/OmniSQL-7B --subfolder ""
log "DONE_RERANK: $RD"
