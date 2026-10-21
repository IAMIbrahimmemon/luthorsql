#!/bin/bash
# Free-form phase 2 (ff2; docs/paper_notes/research/freeform_plan_2026-10-10.md §2, §3.3-2, S3), ON the box:
#   setsid nohup bash /workspace/luthor/scripts/chains/ff2_box.sh > /workspace/logs/ff2.log 2>&1 < /dev/null &
# Retrain the candidate validator on MIXED grammar + free-form (ff) lists, then score the phase-1 DECISION pools with it.
# BIRD-train training DBs only for generation / labelling / training: no DECISION or RESERVED question is generated
# for, labelled or trained on, no BIRD dev, no sealed set.
#   a  gen   ff samples on the staged BIRD-train subset (/root/d/ff2/gen_questions.jsonl, scripts/freeform/ff2_stage.py):
#            Qwen2.5-Coder-7B-Instruct n=8 T=1.0 and OmniSQL-7B n=8 T=1.0 (official prompt, no greedy; plan §2)
#   b  lab   every sample labelled with the gen-time rule against the list gold (merge.py --samples: sqlglot
#            normalisation kept only when the sig is unchanged)
#   c  lists build_lists.py --ff-extend: mixed lists (<= 4 ff classes per grammar list, ff-only lists for
#            grammar-uncovered questions, <= 1 arithmetic-mutant class; gold copies dropped; OmniSQL's right rate
#            capped at Qwen's); val split = the val2 one (4 pinned DBs)
#   d  train one validator, val2 recipe: rerank_sq2.py train, OmniSQL-7B + LoRA r16, 2 epochs cosine, multipos, label
#            smoothing 0.05, token budget TOK, --max-hours MAX_HOURS (the cosine is shortened to fit)
#   e  score the phase-1 DECISION superset (grammar + Omni ff + Qwen ff + arithmetic mutants; scoreset.py) and attach
#            the scores to each pool (scored/<pool>_by_new.jsonl), then the R1 u R2 held-out union val2 scored
#            (scored/heldout_by_new.jsonl; it holds RESERVED rows: no SCORE line is kept for it, RESERVED stays blind;
#            the Mac evaluates DECISION only)
# Inputs (pushed by scripts/chains/run_ff2_box.sh): /root/d/ff2/{gen_questions.jsonl, grammar_lists.jsonl,
#   arith_lab.jsonl, dbs.txt, dec/{superset,m_*}.jsonl, heldout_union.jsonl}, /root/d/ff2dbs/<db>/<db>.sqlite,
#   /root/d/sys1/bird_train/cards.
# Outputs: /workspace/ff2/{gen,lab,lists,rr,scored}; FF2_DONE at the end (FF2_FAIL on errors). Every step is skipped
# when its output exists, so a re-run resumes.
# DRY=1: no setup, no GPU (fake generator: gold / constant / broken SQL; fake validator: a stub checkpoint and random
# scores), for path checks on the Mac with FF2_R / FF2_D / FF2_CODE pointing at local dirs and MAX_Q limiting questions.
set -uo pipefail
R=${FF2_R:-/workspace}; D=${FF2_D:-/root/d}; F=$D/ff2; DB=$D/ff2dbs; S=$D/sys1
O=$R/ff2; L=$R/logs; CODE=${FF2_CODE:-$R/luthor}; DRY=${DRY:-0}; MAX_Q=${MAX_Q:-0}
GV=$R/venvvl/bin/python; V=$R/venv/bin/python
[ "$DRY" = 1 ] && { GV=${DRY_PY:-python3}; V=${DRY_PY:-python3}; }
OMNI=seeklhy/OmniSQL-7B; QWEN=Qwen/Qwen2.5-Coder-7B-Instruct
N=${N:-8}; TEMP=${TEMP:-1.0}; TOK=${TOK:-120000}; MAX_HOURS=${MAX_HOURS:-2.0}
mkdir -p "$L" $O/gen $O/lab $O/lists $O/scored; cd "$CODE" || exit 1
# the box image has no nvcc: FlashInfer's sampler JIT-compiles on first use and fails; use vLLM's torch sampler
export VLLM_USE_FLASHINFER_SAMPLER=0
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf UV_CACHE_DIR=$R/uvcache PATH=$HOME/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1 \
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
log() { echo "$(date -u '+%m-%d %H:%M:%S') [ff2] $*"; }
die() { log "FF2_FAIL $*"; exit 1; }
x() { local n=$1; shift; log "run $n"; "$@" > "$L/ff2_$n.log" 2>&1; local rc=$?
  [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 25 "$L/ff2_$n.log" | grep -av '^SCORE' | sed 's/Traceback/TB/'; }; return $rc; }
for f in $F/gen_questions.jsonl $F/grammar_lists.jsonl $F/dbs.txt $F/dec/superset.jsonl $F/dec/m_both_mut.jsonl \
         $F/heldout_union.jsonl; do [ -s $f ] || die "missing $f"; done
[ -d $S/bird_train/cards ] || die "missing $S/bird_train/cards"
[ -d $DB ] || die "missing $DB"
while read -r d; do [ -z "$d" ] || [ -s $DB/$d/$d.sqlite ] || die "missing DB $d"; done < $F/dbs.txt
log "inputs ok: $(wc -l < $F/gen_questions.jsonl) questions, $(wc -l < $F/grammar_lists.jsonl) grammar lists, $(ls $DB | wc -l) DBs, arith=$([ -s $F/arith_lab.jsonl ] && echo yes || echo no), N=$N T=$TEMP TOK=$TOK MAX_HOURS=$MAX_HOURS"

# ---- setup: vLLM venv (generation) + validator venv + both checkpoints (as ff1_box.sh) ---------------------------------
if [ "$DRY" != 1 ] && [ ! -e "$L/ff2_setup.ok" ]; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  DRV=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1)
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
    VIRTUAL_ENV=$R/venv uv pip install -q -e $CODE --no-deps ) > "$L/ff2_setup.log" 2>&1 || { tail -n 30 "$L/ff2_setup.log"; die setup; }
  $GV -c "import torch, vllm; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda); assert torch.cuda.is_available()" \
    || die "vllm import / CUDA"
  $V -c "import torch, transformers, peft; print('torch', torch.__version__, torch.version.cuda, 'tf', transformers.__version__, 'peft', peft.__version__); assert torch.cuda.is_available(); torch.zeros(1).cuda()" \
    || die "validator venv"
  for m in $OMNI $QWEN; do $V -c "from huggingface_hub import snapshot_download as s; s('$m')" > /dev/null || die "download $m"; done
  touch "$L/ff2_setup.ok"
fi
log DONE_SETUP

# ---- a: generation on the BIRD-train subset ----------------------------------------------------------------------------
gen() {  # gen <name> <model>  -> $O/gen/<name>.jsonl
  local name=$1 model=$2; local out=$O/gen/$name.jsonl
  [ -s $out ] && { log "skip gen $name (exists)"; return 0; }
  if [ "$DRY" = 1 ]; then
    x gen_$name $GV - "$F/gen_questions.jsonl" "$out.tmp" "$N" "$model" "$MAX_Q" <<'PY' || return 1
import json, sys
src, out, n, model, mq = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
qs = [json.loads(l) for l in open(src)]
qs = qs[:: max(1, len(qs) // mq)][:mq] if mq else qs  # spread over the file: val DBs included
with open(out, "w") as f:
    for j, q in enumerate(qs):  # fake: gold (a copy: dropped by hygiene), a reordered gold, constants, a broken query
        g = q["gold_sql"]
        sq = [g, "SELECT * FROM (" + g + ")", "SELECT 1", "SELECT 2", "SELECT nope FROM nowhere",
              None if j % 3 == 0 else "SELECT 3"] * 3
        cl = [{"rank": k, "sql": s, "src": "ff", "gen": model.split("/")[-1], "temp": 1.0, "logp": None, "ntok": 1,
               "finish": "stop"} for k, s in enumerate(sq[:n])]
        f.write(json.dumps({k: q.get(k) for k in ("id", "db_id", "question", "context", "gold_sql", "in_slots", "covered")}
                           | {"cands": cl, "ff_model": model, "ff_skipped": False, "ff_no_evidence": False}) + "\n")
print(f"GEN {len(qs)} questions (DRY fake)")
PY
  else
    x gen_$name $GV scripts/freeform/gen.py --pool $F/gen_questions.jsonl --db-roots $DB --model $model --n $N \
      --temp $TEMP --max-q $MAX_Q --out $out.tmp || return 1
  fi
  mv $out.tmp $out; grep -a '^GEN' "$L/ff2_gen_$name.log" | tail -1
}
nosql_gate() {  # plan §5 S1 kill gate: >= 5% of samples without a SQL block = a prompt / parse bug
  $V - $1 <<'PY'
import json, sys
c = [x for l in open(sys.argv[1]) for x in json.loads(l)["cands"]]
bad = sum(x["sql"] is None for x in c)
print(f"{sys.argv[1].rsplit('/', 1)[-1]}: {len(c)} samples, {bad} without a SQL block ({100 * bad / max(len(c), 1):.2f}%)")
sys.exit(1 if bad >= 0.05 * len(c) else 0)
PY
}
gen qwen_tr $QWEN || die gen_qwen_tr
nosql_gate $O/gen/qwen_tr.jsonl || die "qwen_tr: >= 5% of samples have no SQL block"
gen omni_tr $OMNI || die gen_omni_tr
nosql_gate $O/gen/omni_tr.jsonl || die "omni_tr: >= 5% of samples have no SQL block"
log DONE_GEN

# ---- b: label every sample (gen-time rule, list gold) -----------------------------------------------------------------
NP=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
M=(scripts/freeform/merge.py --db-roots $DB --workers ${WORKERS:-$(( NP > 24 ? 24 : NP ))})
for s in qwen_tr omni_tr; do
  out=$O/lab/$s.jsonl
  [ -s $out ] || { x lab_$s $V "${M[@]}" --ff $O/gen/$s.jsonl --samples --out $out.tmp && mv $out.tmp $out; } || die lab_$s
  grep -a '^SUMMARY' "$L/ff2_lab_$s.log" | tail -1
  $V scripts/freeform/probe.py text --gen $out --src ff 2>&1 | tail -1   # P1 gold-text recall per sample
done
log DONE_LABEL

# ---- c: mixed lists ------------------------------------------------------------------------------------------------
if [ ! -s $O/lists/train_lists.jsonl ]; then
  AR=(); [ -s $F/arith_lab.jsonl ] && AR=(--arith $F/arith_lab.jsonl)
  x lists $V scripts/val2/build_lists.py --ff-extend --grammar-lists $F/grammar_lists.jsonl \
    --questions $F/gen_questions.jsonl --ff-qwen $O/lab/qwen_tr.jsonl --ff-omni $O/lab/omni_tr.jsonl ${AR[@]+"${AR[@]}"} \
    --out $O/lists.tmp || die lists
  rm -rf $O/lists && mv $O/lists.tmp $O/lists
fi
grep -aE '^(CAP|FF2LISTS)' "$L/ff2_lists.log" | cut -c1-1500
[ -s $O/lists/train_lists.jsonl ] && [ -s $O/lists/val_lists.jsonl ] || die "empty lists"
wc -l $O/lists/train_lists.jsonl $O/lists/val_lists.jsonl
log DONE_LISTS

# ---- d: train one validator (val2 recipe) ------------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ ! -s $O/rr/rerank_head.pt ]; then
  if [ "$DRY" = 1 ]; then
    mkdir -p $O/rr/rerank_lora/rerank && echo stub > $O/rr/rerank_head.pt && echo '{"trainer": "DRY"}' > $O/rr/settings.json \
      && echo "trained (DRY stub)" > "$L/ff2_train.log"
  else
    x train $V scripts/rerank_sq2.py train --cards $S/bird_train --train-file $O/lists/train_lists.jsonl \
      --val-file $O/lists/val_lists.jsonl --out $O/rr --base $OMNI --subfolder "" --lora-r 16 --epochs 2 \
      --sched cosine --loss multipos --label-smooth 0.05 --tok-budget $TOK --max-hours $MAX_HOURS || die train
  fi
fi
grep -aE '^(train lists|[0-9]+ steps per epoch|time limit|VAL |SKIPPED_FOR_MEMORY|trained )' "$L/ff2_train.log" | tail -12
[ -s $O/rr/rerank_head.pt ] || die "no checkpoint"
cp -f "$L/ff2_train.log" $O/ 2>/dev/null
log DONE_TRAIN

# ---- e: score DECISION pools, then the held-out union (RESERVED blind) --------------------------------------------------
score() {  # score <name> <cands file> <out>
  local name=$1 in=$2 out=$3
  [ -s $out ] && return 0
  if [ "$DRY" = 1 ]; then
    x score_$name $V - $in $out.tmp <<'PY' || return 1
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
    x score_$name $V scripts/rerank_sq2.py score --rr $O/rr --cards $S/bird_train --dev-cands $in --out $out.tmp \
      --base $OMNI --subfolder "" --lora-r 16 || return 1
  fi
  mv $out.tmp $out
}
score dec $F/dec/superset.jsonl $O/scored/superset_by_new.jsonl || die score_dec
grep -a '^SCORE' "$L/ff2_score_dec.log" | head -1 | cut -c1-400   # DECISION only
POOLS=(); for p in m_both_mut m_both m_omni m_qwen m_mut; do [ -s $F/dec/$p.jsonl ] && POOLS+=($F/dec/$p.jsonl); done
$V scripts/freeform/scoreset.py attach --scored $O/scored/superset_by_new.jsonl --pools "${POOLS[@]}" --suffix _by_new \
  --out-dir $O/scored || die attach
log DONE_SCORE_DEC
score heldout $F/heldout_union.jsonl $O/scored/heldout_by_new.jsonl || die score_heldout
sed -i.bak '/^SCORE/d' "$L/ff2_score_heldout.log" 2>/dev/null; rm -f "$L/ff2_score_heldout.log.bak"  # RESERVED stays blind
ls -la $O/scored
log FF2_DONE
