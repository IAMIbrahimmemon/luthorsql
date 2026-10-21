#!/bin/bash
# Runs ON the Vast instance (round 6). Stages write /workspace/logs/<stage>.log and end with DONE_<STAGE> or
# <STAGE>_FAIL.   STAGE=s1 nohup bash /workspace/luthor/scripts/vast/run6.sh > /workspace/logs/s1.log 2>&1 &
# Data pushed by push6.sh into /root/d: dev6c_bird.jsonl, cards/ (BIRD dev schema cards), dev_databases/,
# prompts.jsonl (training questions for few-shot examples), sqlw5/ (round-5 writer v2 LoRA).
set -x -o pipefail   # pipefail: a crashing python piped into grep/sed must not read as success
R=/workspace; D=/root/d; O=$R/luthor/runs/r6
cd $R/luthor && export HF_HOME=$R/hf PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_ENABLE_HF_TRANSFER=1 VLLM_TEST_FORCE_FP8_MARLIN=1 \
  PATH=$R/venv6/bin:$PATH TOKENIZERS_PARALLELISM=false
V=$R/venv6/bin/python
mkdir -p $O
W=$(( $(nproc) > 4 ? $(nproc) - 2 : 2 ))
# show TAG PATTERN cmd...: run cmd with its FULL output kept in this stage log (tracebacks stay visible), then echo
# the lines matching PATTERN prefixed with TAG. Returns cmd's exit status, or 1 if no line matched.
show() {
  local tag=$1 pat=$2 t rc=0; shift 2; t=$(mktemp)
  "$@" > "$t" 2>&1 || rc=$?
  cat "$t"; grep -a "$pat" "$t" | sed "s/^/$tag/" || { [ $rc != 0 ] || rc=1; }
  rm -f "$t"; [ $rc = 0 ] || echo "show: '$*' failed (exit $rc)"; return $rc
}
Q="--questions $D/dev6c_bird.jsonl --cards $D/cards --db-roots $D/dev_databases --workers $W"
case "${STAGE:?}" in
  s1)  # step 1: OmniSQL-32B off the shelf, three prompt styles + fix; then round-5 v2 baselines on corrected dev
    $V scripts/gen6.py --model seeklhy/OmniSQL-32B --quant ${QUANT:-fp8} ${EAGER---eager} $Q --shots $D/prompts.jsonl --hints --fix \
      --styles think:8 sql:8 fewshot:8 --out $O/s1_omni.jsonl || { echo S1_FAIL; exit 1; }
    echo DONE_S1OMNI
    $V scripts/gen6.py --model Qwen/Qwen2.5-Coder-7B-Instruct --lora $D/sqlw5/lora $Q --hints \
      --styles sql:16 --gpu-mem 0.85 --out $O/s1_v2full.jsonl || { echo S1_FAIL; exit 1; }
    echo DONE_S1V2FULL
    # the round-5 system exactly (9,000-character schema cut), for the round-5 -> round-6 comparison
    mkdir -p $D/bird_dev_cards && ln -sfn $D/cards $D/bird_dev_cards/cards
    $V scripts/sqlw_gen.py --lora $D/sqlw5/lora --hints --questions $D/dev6c_bird.jsonl --cards $D/bird_dev_cards \
      --db-roots $D/dev_databases --workers $W --out $O/s1_v2r5.jsonl || { echo S1_FAIL; exit 1; }
    echo DONE_S1 ;;
  arctic)  # Arctic-Text2SQL-R1-7B off the shelf (its own prompt): greedy + 8 samples on prompts built on the Mac
           # ($D/arc/<set>_prompts.jsonl: BIRD dev corrected + sealed sets). Execution and scoring on the Mac.
    for p in $D/arc/*_prompts.jsonl; do
      n=$(basename $p _prompts.jsonl); [ -s $O/arc_$n.jsonl ] && continue
      $V scripts/gen6.py --model Snowflake/Arctic-Text2SQL-R1-7B --from-prompts $p --styles arctic:${ARCN:-8} \
        --temp 0.8 --gpu-mem 0.90 --out $O/arc_$n.jsonl || { echo ARCTIC_FAIL; exit 1; }
      echo DONE_ARCTIC_$n
    done
    echo DONE_ARCTIC ;;
  selgen)  # selector training candidates: OmniSQL-32B (fp8, CUDA graphs on) on prompts built on the Mac
           # ($D/sel/<part>_prompts.jsonl, BIRD-Verified questions); execution on the Mac.
    for p in $D/sel/*_prompts.jsonl; do
      n=$(basename $p _prompts.jsonl); [ -s $O/sel_$n.jsonl ] && continue
      VLLM_TEST_FORCE_FP8_MARLIN=0 $V scripts/gen6.py --model seeklhy/OmniSQL-32B --quant fp8 --from-prompts $p --styles ${SELSTYLES:-think:8 sql:8} \
        --gpu-mem 0.92 --out $O/sel_$n.jsonl || { echo SELGEN_FAIL; exit 1; }
      echo DONE_SELGEN_$n
    done
    echo DONE_SELGEN ;;
  seltrain)  # selector over pooled AR + slot candidates: train on $D/sel/train_cands.jsonl (BIRD-Verified train),
             # choices on val_cands.jsonl, then score every $D/sel/score_<set>.jsonl. SELFLAGS e.g. --result-preview
    RD=$O/${SELD:-sel}; mkdir -p $RD   # card dirs pushed by push6.sh straight under $D
    CARDS="$D/bird_train $D/bird_dev $D/spider_dk $D/spider_realistic $D/kaggledbqa"
    $V scripts/rerank_sq2.py train ${SELFLAGS:-} --cards $CARDS --train-cands $D/sel/train_cands.jsonl \
      --dev-cands $D/sel/val_cands.jsonl --out $RD --base seeklhy/OmniSQL-7B --subfolder "" --train-k ${TRAINK:-8} \
      --hours ${SELH:-0.75} --epochs ${SELEP:-2} || { echo SELTRAIN_FAIL; exit 1; }
    for f in $D/sel/score_*.jsonl; do
      n=$(basename $f .jsonl | sed 's/^score_//')
      show "[$n] " "^SCORE" $V scripts/rerank_sq2.py score --rr $RD --cards $CARDS --dev-cands $f --out $RD/${n}_scored.jsonl \
        --base seeklhy/OmniSQL-7B --subfolder "" || { echo SELTRAIN_FAIL; exit 1; }
    done
    echo DONE_SELTRAIN ;;
  s2b)  # step 2b: OmniSQL-32B samples the training questions (prompts built on the Mac, so no train DBs here);
        # one chunk at a time so each finished chunk can be copied back. Execution happens on the Mac.
    for p in $D/s2b/part_*.jsonl; do
      n=$(basename $p .jsonl); [ -s $O/s2b_$n.jsonl ] && continue
      $V scripts/gen6.py --model seeklhy/OmniSQL-32B --quant ${QUANT:-fp8} ${EAGER---eager} --from-prompts $p \
        --styles sql:4 --keep-text --out $O/s2b_$n.jsonl || { echo S2B_FAIL; exit 1; }
      echo DONE_S2B_$n
    done
    echo DONE_S2B ;;
  sys1)  # System-1 pilot: grammar v3 slot model on an OmniSQL-7B backbone (docs/SYSTEM1_PLAN.md).
         # Data from push6.sh: sys1/ (v3 plans + cards, vocab, dev_dbs.json, dbs/ for held-out dev), dev_databases/
    S=$D/sys1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $V scripts/train_sq2.py --plans $S/bird_train $S/spider_train $S/synsql_train --db-roots $S/dbs $D/dev_databases \
      --base ${BASE:-seeklhy/OmniSQL-7B} --subfolder "" --vocab $S/templates.json --dev-dbs $S/${DEVDBS:-dev_dbs.json} \
      --out $O/${OUTN:-sys1_pilot} --hours ${HOURS:-2} --eval-every ${EVAL_EVERY:-500} --budget-chars ${BUDGET:-12000} \
      --gate $S/bird_dev --gate-total 1534 ${EXTRA:-} || { echo SYS1_FAIL; exit 1; }
    echo DONE_SYS1 ;;
  sys1k)  # top-K headroom of the trained slot model on BIRD dev: greedy, execution-guided pick, oracle@K
    S=$D/sys1
    $V scripts/eval_sq2.py --run $O/${OUTN:-sys1_pilot} --plans $S/bird_dev --sealed --db-roots $D/dev_databases \
      --k ${K:-32} --device cuda --out $O/${OUTN:-sys1_pilot}/topk_bird_dev.jsonl || { echo SYS1K_FAIL; exit 1; }
    echo DONE_SYS1K ;;
  rr)  # validator (reranker) for the slot model OUTN: candidates on BIRD train DBs (+ Spider and SynSQL train with RRWIDE=1),
       # held-out training DBs and BIRD dev; train a rerank adapter on the same backbone; score BIRD dev and the sealed sets
    S=$D/sys1; R=$O/${OUTN:?}; G="$V scripts/rerank_sq2.py gen --run $R --k ${K:-8} --workers $W ${DISTINCT:+--distinct}"; RD=${RRD:-rr}
    DV=$S/${DEVDBS:-dev_dbs_bird.json}; CARDS="$S/bird_train $S/bird_dev $S/spider_train $S/synsql_train $S/sealed/spider_dk $S/sealed/spider_realistic $S/sealed/kaggledbqa"
    mkdir -p $R/$RD
    [ -s $R/$RD/cands_train_bird.jsonl ] || $G --plans $S/bird_train --db-roots $S/rrdbs --dev-dbs $DV --split train \
      --max-q ${MAXQ:-9000} --out $R/$RD/cands_train_bird.jsonl || { echo RR_FAIL; exit 1; }
    if [ "${RRWIDE:-0}" = 1 ]; then
      for set in spider_train synsql_train; do
        [ -s $R/$RD/cands_train_$set.jsonl ] || $G --plans $S/$set --db-roots $S/rrdbs2 --dev-dbs $DV --split train \
          --max-q ${MAXQW:-3000} --out $R/$RD/cands_train_$set.jsonl || { echo RR_FAIL; exit 1; }
      done
    fi
    cat $R/$RD/cands_train_*.jsonl > $R/$RD/cands_train.jsonl
    [ -s $R/$RD/cands_dev.jsonl ] || $G --plans $S/bird_train $S/spider_train --db-roots $S/dbs --dev-dbs $DV --split dev \
      --out $R/$RD/cands_dev.jsonl || { echo RR_FAIL; exit 1; }
    [ -s $R/$RD/cands_bird_dev.jsonl ] || $G --plans $S/bird_dev --db-roots $D/dev_databases --dev-dbs $S/bird_dev_dbs.json --split dev \
      --out $R/$RD/cands_bird_dev.jsonl || { echo RR_FAIL; exit 1; }
    for set in spider_dk spider_realistic kaggledbqa; do
      [ -s $R/$RD/cands_$set.jsonl ] || $G --plans $S/sealed/$set --db-roots $S/sealed/dbs --dev-dbs $S/sealed/$set.dbs.json --split dev \
        --out $R/$RD/cands_$set.jsonl || { echo RR_FAIL; exit 1; }
    done
    echo DONE_RRGEN
    $V scripts/rerank_sq2.py train --cards $CARDS --train-cands $R/$RD/cands_train.jsonl \
      --dev-cands $R/$RD/cands_dev.jsonl --out $R/$RD --base seeklhy/OmniSQL-7B --subfolder "" --hours ${RRH:-1.0} --epochs ${RREP:-1} || { echo RR_FAIL; exit 1; }
    for set in bird_dev spider_dk spider_realistic kaggledbqa; do
      show "[$set] " "^SCORE" $V scripts/rerank_sq2.py score --rr $R/$RD --cards $CARDS --dev-cands $R/$RD/cands_$set.jsonl \
        --out $R/$RD/${set}_scored.jsonl --base seeklhy/OmniSQL-7B --subfolder "" || { echo RR_FAIL; exit 1; }
    done
    echo DONE_RR ;;
  rrtrain)  # a second validator on the candidates already generated in RRSRC (e.g. RRFLAGS=--result-preview),
             # trained and scored exactly like the rr stage, into RRD
    S=$D/sys1; R=$O/${OUTN:?}; RD=${RRD:?}; SRC=$R/${RRSRC:?}
    CARDS="$S/bird_train $S/bird_dev $S/spider_train $S/synsql_train $S/sealed/spider_dk $S/sealed/spider_realistic $S/sealed/kaggledbqa"
    mkdir -p $R/$RD
    $V scripts/rerank_sq2.py train --cards $CARDS --train-cands $SRC/cands_train.jsonl --dev-cands $SRC/cands_dev.jsonl \
      --out $R/$RD --base seeklhy/OmniSQL-7B --subfolder "" --hours ${RRH:-1.0} --epochs ${RREP:-1} ${RRFLAGS:-} || { echo RR_FAIL; exit 1; }
    for set in bird_dev spider_dk spider_realistic kaggledbqa; do
      show "[$set] " "^SCORE" $V scripts/rerank_sq2.py score --rr $R/$RD --cards $CARDS --dev-cands $SRC/cands_$set.jsonl \
        --out $R/$RD/${set}_scored.jsonl --base seeklhy/OmniSQL-7B --subfolder "" || { echo RR_FAIL; exit 1; }
    done
    echo DONE_RR ;;
  pool)  # pool distinct top-K candidates from older checkpoints (POOLRUNS, dirs under $O) with OUTN's, then score
          # the pooled lists with OUTN's validator (RRD): BIRD dev and the sealed sets
    S=$D/sys1; R=$O/${OUTN:?}; RD=${RRD:?}; CARDS="$S/bird_dev $S/sealed/spider_dk $S/sealed/spider_realistic $S/sealed/kaggledbqa"
    for set in bird_dev spider_dk spider_realistic kaggledbqa; do
      if [ $set = bird_dev ]; then P="--plans $S/bird_dev --db-roots $D/dev_databases --dev-dbs $S/bird_dev_dbs.json"
      else P="--plans $S/sealed/$set --db-roots $S/sealed/dbs --dev-dbs $S/sealed/$set.dbs.json"; fi
      ins=$R/${RRSRC:-$RD}/cands_$set.jsonl
      for pr in ${POOLRUNS:?}; do
        mkdir -p $R/pool; f=$R/pool/${pr}_$set.jsonl
        [ -s $f ] || $V scripts/rerank_sq2.py gen --run $O/$pr --k ${K:-32} --workers $W --distinct $P --split dev --out $f \
          || { echo POOL_FAIL; exit 1; }
        ins="$ins $f"
      done
      show "[$set] " "" $V scripts/merge_cands.py --inputs $ins --k ${POOLK:-48} --out $R/$RD/cands_${set}_pool.jsonl \
        || { echo POOL_FAIL; exit 1; }
      show "[$set pool] " "^SCORE" $V scripts/rerank_sq2.py score --rr $R/$RD --cards $CARDS --dev-cands $R/$RD/cands_${set}_pool.jsonl \
        --out $R/$RD/${set}_pool_scored.jsonl --base seeklhy/OmniSQL-7B --subfolder "" || { echo POOL_FAIL; exit 1; }
    done
    echo DONE_POOL ;;
  evalset)  # robustness of slot model OUTN (top-K, no validator): BIRD dev with descriptions removed / auto-written,
            # and the sealed sets (never trained or tuned on)
    S=$D/sys1; R=$O/${OUTN:?}
    for set in bird_dev_nodesc bird_dev_autodesc; do
      show "[$set] " "^FINAL\|Error" $V scripts/eval_sq2.py --run $R --plans $S/$set --sealed --db-roots $D/dev_databases \
        --k ${K:-8} --device cuda --out $R/eval_$set.jsonl || { echo EVALSET_FAIL; exit 1; }
    done
    for set in spider_dk spider_realistic kaggledbqa; do
      show "[$set] " "^FINAL\|Error" $V scripts/eval_sq2.py --run $R --plans $S/sealed/$set --sealed --db-roots $S/sealed/dbs \
        --k ${K:-8} --device cuda --out $R/eval_$set.jsonl || { echo EVALSET_FAIL; exit 1; }
    done
    echo DONE_EVALSET ;;
  f1diag)  # does the column-mention feature cause the formula_1 collapse? same checkpoint, mentions on / off / capped
    for mode in normal off cap2 inv; do
      ( [ $mode = off ] && export LUTHOR_COLMENT_OFF=1; [ $mode = cap2 ] && export LUTHOR_COLMENT_MAXHIT=2
        [ $mode = inv ] && export LUTHOR_MENTION_WEIGHT=inv
        show "$mode " "^FINAL\|Error" $V scripts/eval_sq2.py --run $O/${OUTN:?} --plans $D/sys1/bird_dev --dev-dbs $D/sys1/f1_only.json \
          --db-roots $D/dev_databases --k 1 --device cuda --out $O/${OUTN}/f1_$mode.jsonl ) || { echo F1DIAG_FAIL; exit 1; }
    done
    echo DONE_F1DIAG ;;
  speed)  # step time on this host: plain, joint head, column mentions (50 steps each, small data load)
    S=$D/sys1
    for cfg in plain "--joint-layers 2" "--col-mentions"; do
      x=$cfg; [ "$cfg" = plain ] && x=""
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $V scripts/train_sq2.py --plans $S/bird_train $S/spider_train $S/synsql_train \
        --db-roots $S/dbs $D/dev_databases --base seeklhy/OmniSQL-7B --subfolder "" --vocab $S/templates.json \
        --dev-dbs $S/dev_dbs.json --limit 3000 --out /tmp/speed --max-steps 50 --eval-every 100000 --budget-chars 9000 $x \
        > /tmp/speed.log 2>&1 || { cat /tmp/speed.log; echo SPEED_FAIL; exit 1; }
      cat /tmp/speed.log; grep -a "^step 50/" /tmp/speed.log | sed "s/^/[$cfg] /" || { echo SPEED_FAIL; exit 1; }
    done
    echo DONE_SPEED ;;
esac
