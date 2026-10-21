#!/bin/bash
# Two H100s (one Vast, one Shadeform): OmniSQL-32B samples half the BIRD-Verified questions each; the Mac executes
# them and builds the selector files; then each machine trains one selector (A plain, B --result-preview),
# scores the pools, is pulled and DELETED.     bash selchain.sh <vast_id> <shadeform_id>
cd ~/Documents/GitHub/luthor-general
SP=${SP:-/tmp/luthor_chain}; mkdir -p $SP
F=$SP/selflags; mkdir -p $F runs/sel
log() { echo "$(date -u +%H:%M) $*"; }
S() { bash scripts/vast/ssh.sh $1 "${@:2}" 2>/dev/null | grep -av "Welcome\|Have fun"; }
pull() { bash -c 'source scripts/vast/ssh.sh '$1'; mkdir -p runs/sel/'$2'; "${RS[@]}" --exclude rerank_lora "$H:/workspace/luthor/runs/r6/" runs/sel/'$2'/; "${RS[@]}" "$H:/workspace/logs/" runs/sel/'$2'/logs/' >/dev/null 2>&1; }
kill_m() { pull $1 $2; CONFIRM=yes bash scripts/vast/destroy.sh $1 2>&1 | grep -v Welcome | tail -2; log "$2 DELETED ($3)"; touch $F/$2.$3; }
waitlog() {  # id log done fail
  while true; do
    st=$(S $1 "grep -aE '^($3|$4)\$|Traceback' /workspace/logs/$2.log | tail -1")
    echo "$st" | grep -q "^$3$" && return 0
    echo "$st" | grep -qE "^$4$|Traceback" && { S $1 "tail -20 /workspace/logs/$2.log"; return 1; }
    sleep 60
  done
}
machine() {  # id part selflags
  ID=$1; P=$2
  until S $ID 'command -v rsync && command -v python3' | grep -q python3; do REFRESH=1 bash scripts/vast/ssh.sh $ID true >/dev/null 2>&1; sleep 15; done
  log "$P booted: $(S $ID 'nvidia-smi --query-gpu=name,memory.used --format=csv,noheader')"
  bash scripts/vast/push6.sh $ID data/eval/sel_parts/$P/sel data/sys1/bird_train data/sys1/bird_dev data/sys1/sealed/spider_dk \
    data/sys1/sealed/spider_realistic data/sys1/sealed/kaggledbqa 2>&1 | grep -q "^PUSHED6" || { kill_m $ID $P PUSH_FAILED; return; }
  S $ID '(MODELS="seeklhy/OmniSQL-32B seeklhy/OmniSQL-7B" setsid nohup bash /workspace/luthor/scripts/vast/setup6.sh > /workspace/logs/setup.log 2>&1 < /dev/null &)'
  waitlog $ID setup DONE_SETUP6 SETUP6_FAIL || { kill_m $ID $P SETUP_FAILED; return; }
  log "$P setup done"
  S $ID "(STAGE=selgen setsid nohup bash /workspace/luthor/scripts/vast/run6.sh > /workspace/logs/selgen.log 2>&1 < /dev/null &)"
  waitlog $ID selgen DONE_SELGEN SELGEN_FAIL || { kill_m $ID $P SELGEN_FAILED; return; }
  pull $ID $P; log "$P samples pulled"; touch $F/$P.gen
  until [ -e $F/cands_ready ] || [ -e $F/cands_failed ]; do sleep 20; done
  [ -e $F/cands_failed ] && { kill_m $ID $P NO_CANDS; return; }
  bash scripts/vast/push6.sh $ID data/eval/sel_cands/sel 2>&1 | grep -q "^PUSHED6" || { kill_m $ID $P PUSH2_FAILED; return; }
  S $ID "(STAGE=seltrain SELD=sel_$P SELFLAGS='$3' SELH=${SELH:-0.6} setsid nohup bash /workspace/luthor/scripts/vast/run6.sh > /workspace/logs/seltrain.log 2>&1 < /dev/null &)"
  waitlog $ID seltrain DONE_SELTRAIN SELTRAIN_FAIL || { kill_m $ID $P SELTRAIN_FAILED; return; }
  S $ID "grep -aE '^\[|RERANK DEV' /workspace/logs/seltrain.log" | sed "s/^/$P /"
  kill_m $ID $P ALL_DONE
}
machine $1 a "" > $SP/sel_a.log 2>&1 &
machine $2 b "--result-preview" > $SP/sel_b.log 2>&1 &
# Mac: once both halves are back, execute and build the selector files
until { [ -e $F/a.gen ] || ls $F/a.*FAIL* >/dev/null 2>&1; } && { [ -e $F/b.gen ] || ls $F/b.*FAIL* >/dev/null 2>&1; }; do sleep 30; done
G=$(ls runs/sel/a/sel_*.jsonl runs/sel/b/sel_*.jsonl 2>/dev/null)
log "building selector files from: $G"
mkdir -p data/eval/sel_cands/sel
B=".venv/bin/python scripts/build_sel.py --gen $G --db-roots data/raw/bird_train/train_databases"
$B --questions data/eval/sel/verified_train.jsonl --out data/eval/sel_cands/sel/train_cands.jsonl && \
$B --questions data/eval/sel/verified_val.jsonl --out data/eval/sel_cands/sel/val_cands.jsonl && \
cp data/eval/sel/score_*.jsonl data/eval/sel_cands/sel/ && touch $F/cands_ready || touch $F/cands_failed
wait
log "ALL FINISHED"; ls $F
