#!/bin/bash
# Free-form pilot ff1 (docs/paper_notes/research/freeform_plan_2026-10-10.md, steps S1 + S2), ON the box:
#   setsid nohup bash /workspace/luthor/scripts/chains/ff1_box.sh > /workspace/logs/ff1.log 2>&1 < /dev/null &
# DECISION (796 held-out BIRD-train questions) + the BIRD-train bug set (training DBs) only: no RESERVED, no BIRD dev.
#   S1a  OmniSQL-7B on DECISION: greedy + 16 samples at T=0.8 (official prompt, scripts/freeform/gen.py)
#   S1b  probes: Qwen2.5-Coder-7B-Instruct, same settings, on DECISION; OmniSQL n=8 (+greedy) without evidence on
#        DECISION (and Qwen the same, QWEN_NOEV=1 default, for the P3 comparison); both models greedy + 16 on the bug set
#   label every sample exactly as at grammar gen time (merge.py --samples), then merge into the DECISION grammar pool:
#        m_omni, m_qwen, m_both (= grammar + omni + qwen) and, when /root/d/ff1/extra_cands.jsonl exists (arithmetic
#        mutants), m_mut (grammar + mut) and m_both_mut; uncovered DECISION questions become ff-only records
#   S2   score ONE superset of those pools (scoreset.py) with the R1 and R2 validators (rerank_sq2.py score), attach the
#        scores to each pool -> /workspace/ff1/scored/<pool>_by_r{1,2}.jsonl. The pick rules are fitted on the Mac.
# Inputs (pushed by scripts/chains/run_ff1_box.sh): /root/d/ff1/{decision_ids.json, decision_pool.jsonl,
#   bug_train_ids.json, train.json[, extra_cands.jsonl]}, /root/d/ffdbs/<db>/<db>.sqlite,
#   /root/d/ens3/{v_r1, v_r2, sys1/bird_train/cards}.
# Outputs: /workspace/ff1/{gen,lab,pools,scored}, FF1_DONE at the end (FF1_FAIL on errors). Every step is skipped
# when its output exists, so a re-run resumes.
# DRY=1: no setup, no GPU (fake generator + fake validator scores), for path checks on the Mac with FF1_R / FF1_D /
# FF1_CODE pointing at local dirs and MAX_Q limiting the questions.
set -uo pipefail
R=${FF1_R:-/workspace}; D=${FF1_D:-/root/d}; F=$D/ff1; DB=$D/ffdbs; E3=$D/ens3
O=$R/ff1; L=$R/logs; CODE=${FF1_CODE:-$R/luthor}; DRY=${DRY:-0}; MAX_Q=${MAX_Q:-0}
GV=$R/venvvl/bin/python; V=$R/venv/bin/python
[ "$DRY" = 1 ] && { GV=${DRY_PY:-python3}; V=${DRY_PY:-python3}; }
OMNI=seeklhy/OmniSQL-7B; QWEN=Qwen/Qwen2.5-Coder-7B-Instruct
N=${N:-16}; NNOEV=${NNOEV:-8}; TEMP=${TEMP:-0.8}; QWEN_NOEV=${QWEN_NOEV:-1}; MAX_MUT=${MAX_MUT:-0}
mkdir -p "$L" $O/gen $O/lab $O/pools $O/scored; cd "$CODE" || exit 1
# the box image has no nvcc: FlashInfer's sampler JIT-compiles on first use and fails; use vLLM's torch sampler
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME=$R/hf UV_CACHE_DIR=$R/uvcache PATH=$HOME/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1 \
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
log() { echo "$(date -u '+%m-%d %H:%M:%S') [ff1] $*"; }
die() { log "FF1_FAIL $*"; exit 1; }
x() { local n=$1; shift; log "run $n"; "$@" > "$L/ff1_$n.log" 2>&1; local rc=$?
  [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 25 "$L/ff1_$n.log" | sed 's/Traceback/TB/'; }; return $rc; }
for f in $F/decision_ids.json $F/decision_pool.jsonl $F/bug_train_ids.json $F/train.json $E3/v_r1/rerank_head.pt \
         $E3/v_r2/rerank_head.pt; do [ -s $f ] || die "missing $f"; done
[ -d $E3/sys1/bird_train/cards ] || die "missing $E3/sys1/bird_train/cards"
[ -d $DB ] || die "missing $DB"
log "inputs ok: $(wc -l < $F/decision_pool.jsonl) pool records, $(ls $DB | wc -l) DBs, extra=$([ -s $F/extra_cands.jsonl ] && echo yes || echo no)"

# ---- setup: vLLM venv (generation) + validator venv (as ens3_box.sh) + both checkpoints ---------------------------------
if [ "$DRY" != 1 ] && [ ! -e "$L/ff1_setup.ok" ]; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  DRV=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1)
  # current vLLM wheels are built for CUDA 13 (setup6.sh); a 12.8/12.9 driver gets a cu128 vLLM (VLLM_SPEC_CU128)
  if awk -v v="${DRV:-0}" 'BEGIN { split(v, a, "."); exit !(a[1] + 0 >= 13) }'; then TB=cu130; VS=${VLLM_SPEC:-vllm}
  elif awk -v v="${DRV:-0}" 'BEGIN { split(v, a, "."); exit !(a[1] + 0 == 12 && a[2] + 0 >= 8) }'; then
    TB=cu128; VS=${VLLM_SPEC_CU128:-vllm==0.11.0}
  else die "driver CUDA ${DRV:-unknown} < 12.8"; fi
  log "driver CUDA $DRV: torch backend $TB, $VS"
  df -h $R | tail -1
  command -v uv >/dev/null || pip install -q --break-system-packages uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  ( set -e
    uv venv -q --python 3.12 $R/venvvl
    VIRTUAL_ENV=$R/venvvl uv pip install -q --torch-backend=$TB "$VS" sqlglot huggingface_hub hf_transfer
    uv venv -q --python 3.12 $R/venv
    VIRTUAL_ENV=$R/venv uv pip install -q --torch-backend=auto torch "transformers>=5.0" peft sqlglot numpy safetensors \
      accelerate huggingface_hub hf_transfer
    VIRTUAL_ENV=$R/venv uv pip install -q -e $CODE --no-deps ) > "$L/ff1_setup.log" 2>&1 || { tail -n 30 "$L/ff1_setup.log"; die setup; }
  $GV -c "import torch, vllm; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda); assert torch.cuda.is_available()" \
    || die "vllm import / CUDA"
  $V -c "import torch, transformers, peft; print('torch', torch.__version__, torch.version.cuda, 'tf', transformers.__version__, 'peft', peft.__version__); assert torch.cuda.is_available(); torch.zeros(1).cuda()" \
    || die "validator venv"
  for m in $OMNI $QWEN; do $V -c "from huggingface_hub import snapshot_download as s; s('$m')" > /dev/null || die "download $m"; done
  touch "$L/ff1_setup.ok"
fi
log DONE_SETUP

# ---- S1: generation -------------------------------------------------------------------------------------------------
# gen <name> <model> <ids manifest> <n> [extra gen.py args]  -> $O/gen/<name>.jsonl
gen() {
  local name=$1 model=$2 ids=$3 n=$4; shift 4; local out=$O/gen/$name.jsonl
  [ -s $out ] && { log "skip gen $name (exists)"; return 0; }
  if [ "$DRY" = 1 ]; then
    x gen_$name $GV - "$F/train.json" "$ids" "$out.tmp" "$n" "$model" "$MAX_Q" <<'PY' || return 1
import json, sys
tr, ids, out, n, model, mq = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
raw = json.load(open(tr)); keep = json.load(open(ids)); keep = keep["ids"] if isinstance(keep, dict) else keep
qs = [(f"bird_train_train_{i}", r) for i, r in enumerate(raw) if f"bird_train_train_{i}" in set(keep)][: mq or None]
with open(out, "w") as f:
    for j, (qid, r) in enumerate(qs):  # fake: greedy = gold, then gold / a constant / a broken query (/ no SQL block)
        sq = [r["SQL"], r["SQL"], "SELECT 1", "SELECT nope FROM nowhere"] * 5
        sq[-1 if j % 4 else 3] = None
        cl = [{"rank": k, "sql": s, "src": "ff", "gen": model.split("/")[-1], "temp": 0.0 if k == 0 else 0.8,
               "logp": None, "ntok": 1, "finish": "stop"} for k, s in enumerate(sq[: n + 1])]
        f.write(json.dumps({"id": qid, "db_id": r["db_id"], "question": r["question"], "context": r.get("evidence", ""),
                            "gold_sql": r["SQL"], "in_slots": None, "covered": None, "cands": cl, "ff_model": model,
                            "ff_skipped": False, "ff_no_evidence": False}) + "\n")
print(f"GEN {len(qs)} questions (DRY fake)")
PY
  else
    x gen_$name $GV scripts/freeform/gen.py --questions $F/train.json --id-prefix bird_train_train_ --ids $ids \
      --db-roots $DB --model $model --n $n --temp $TEMP --greedy --max-q $MAX_Q --out $out.tmp "$@" || return 1
  fi
  mv $out.tmp $out; grep -a '^GEN' "$L/ff1_gen_$name.log" | tail -1
}
DEC=$F/decision_ids.json; BUG=$F/bug_train_ids.json
gen omni_dec $OMNI $DEC $N || die gen_omni_dec
# kill gate (plan §5 S1): >= 5% of samples without a SQL block means a prompt / parse bug: stop before spending more
$V - $O/gen/omni_dec.jsonl <<'PY' || die "omni_dec: >= 5% of samples have no SQL block"
import json, sys
c = [x for l in open(sys.argv[1]) for x in json.loads(l)["cands"]]
bad = sum(x["sql"] is None for x in c)
print(f"omni_dec: {len(c)} samples, {bad} without a SQL block ({100 * bad / max(len(c), 1):.2f}%)")
sys.exit(1 if bad >= 0.05 * len(c) else 0)
PY
gen qwen_dec $QWEN $DEC $N || die gen_qwen_dec
gen omni_noev $OMNI $DEC $NNOEV --no-evidence || die gen_omni_noev
gen omni_bug $OMNI $BUG $N || die gen_omni_bug
gen qwen_bug $QWEN $BUG $N || die gen_qwen_bug
[ "$QWEN_NOEV" = 1 ] && { gen qwen_noev $QWEN $DEC $NNOEV --no-evidence || die gen_qwen_noev; }
log DONE_GEN

# ---- label every sample (gen-time rule) ------------------------------------------------------------------------------
M=(scripts/freeform/merge.py --db-roots $DB --workers ${WORKERS:-$(( $(nproc 2>/dev/null || echo 8) > 24 ? 24 : $(nproc 2>/dev/null || echo 8) ))})
labf() {  # labf <name> <ids>
  local out=$O/lab/$1.jsonl
  [ -s $O/gen/$1.jsonl ] || return 0
  [ -s $out ] || { x lab_$1 $V "${M[@]}" --ff $O/gen/$1.jsonl --ids $2 --samples --out $out.tmp && mv $out.tmp $out; } || return 1
  grep -a '^SUMMARY' "$L/ff1_lab_$1.log" | tail -1
}
for s in omni_dec qwen_dec omni_noev qwen_noev; do labf $s $DEC || die lab_$s; done
for s in omni_bug qwen_bug; do labf $s $BUG || die lab_$s; done
for s in omni_dec qwen_dec omni_noev qwen_noev omni_bug qwen_bug; do   # P1 gold-text recall per sample (Mac repeats)
  [ -s $O/lab/$s.jsonl ] && $V scripts/freeform/probe.py text --gen $O/lab/$s.jsonl --src ff 2>&1 | tail -1
done

# ---- merge into the DECISION grammar pool ----------------------------------------------------------------------------
P=$F/decision_pool.jsonl; X=$F/extra_cands.jsonl
mp() {  # mp <pool name> [merge args]
  local name=$1; shift; local out=$O/pools/$name.jsonl
  [ -s $out ] || { x merge_$name $V "${M[@]}" --pool $P --ids $DEC --prelabelled --out $out.tmp "$@" && mv $out.tmp $out; } \
    || return 1
  grep -aE '^(MERGED|SUMMARY|EXTRA)' "$L/ff1_merge_$name.log" | tail -3
}
mp m_omni --ff $O/lab/omni_dec.jsonl || die merge_m_omni
mp m_qwen --ff $O/lab/qwen_dec.jsonl || die merge_m_qwen
mp m_both --ff $O/lab/omni_dec.jsonl $O/lab/qwen_dec.jsonl || die merge_m_both
POOLS=($O/pools/m_both.jsonl $O/pools/m_omni.jsonl $O/pools/m_qwen.jsonl)
if [ -s $X ]; then
  mp m_both_mut --ff $O/lab/omni_dec.jsonl $O/lab/qwen_dec.jsonl --extra $X --max-extra $MAX_MUT || die merge_m_both_mut
  mp m_mut --extra $X --max-extra $MAX_MUT || die merge_m_mut
  POOLS=($O/pools/m_both_mut.jsonl "${POOLS[@]}" $O/pools/m_mut.jsonl)
fi
log DONE_MERGE

# ---- S2: one superset, scored by R1 and R2, scores attached to every pool ---------------------------------------------
SUP=$O/pools/superset.jsonl
[ -s $SUP ] || $V scripts/freeform/scoreset.py build --pools "${POOLS[@]}" --out $SUP || die superset
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for v in r1 r2; do
  out=$O/scored/superset_by_$v.jsonl
  if [ ! -s $out ]; then
    if [ "$DRY" = 1 ]; then
      x score_$v $V - $SUP $out.tmp <<'PY' || die score_$v
import json, random, sys
rnd = random.Random(0); lines = []
for l in open(sys.argv[1]):
    r = json.loads(l)
    for c in r["cands"]:
        if c.get("sql"):
            c["rr"] = rnd.gauss(0, 1) + (1.0 if c.get("ok") else 0.0)
    lines.append(json.dumps(r) + "\n")
open(sys.argv[2], "w").write("".join(lines)); print("SCORE (DRY fake)")
PY
    else
      x score_$v $V scripts/rerank_sq2.py score --rr $E3/v_$v --cards $E3/sys1/bird_train --dev-cands $SUP \
        --out $out.tmp --base $OMNI --subfolder "" || die score_$v
    fi
    mv $out.tmp $out
  fi
  grep -a '^SCORE' "$L/ff1_score_$v.log" | head -1 | cut -c1-400
  $V scripts/freeform/scoreset.py attach --scored $out --pools "${POOLS[@]}" --suffix _by_$v --out-dir $O/scored \
    || die attach_$v
done
ls -la $O/scored
log FF1_DONE
