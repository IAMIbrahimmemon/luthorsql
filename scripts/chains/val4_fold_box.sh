#!/bin/bash
# VAL4 fold generator F (runs/phase1/val4/PREREG.md), ON the box (one H100 80 GB), started by run_val4_fold_box.sh:
#   FOLD=1 setsid nohup bash /workspace/luthor/scripts/chains/val4_fold_box.sh > /workspace/logs/val4fold.log 2>&1 < /dev/null &
# A teacher-role generator that never saw fold F's databases: R1's EXACT training recipe (runs/r1r2/r1/args.json and
# r1r2_chain.sh stage 2: same plan sets incl. rare1 (the same 5,483 rows, sha256 c9ac3783cfb92e0a), --repeat
# bird_train=3, --max-steps 6714, --cand-order v2 --value-side --col-mentions, budget 9000, eval every 1000, seed 0 =
# no --seed flag, lr / LoRA r / batch at their defaults as in R1) EXCEPT
#   --drop-dbs $VF/fold<F>_drop.json   fold F + the 9 held-out DBs (DECISION / RESERVED / retail_world) leave training
#   --dev-dbs  $VF/fold<F>_dev.json    checkpoint selection on the left-out fold (its DBs with an SQLite here), as R1
#                                      selected on its held-out DBs
#   --db-roots $S/rrdbs                the dev DBs' SQLite files (only the dev evaluation reads --db-roots)
#   $D/rare1_fold<F>                   R1's rare1 rows minus those on fold F's DBs (2,941 / 2,542 of 5,483; cards = rare1's).
#                                      Training rows are identical (rows on --dev-dbs never train, the rest of fold F is
#                                      dropped), but the dev set is then BIRD-train rows only, as R1's (its held-out DBs
#                                      had no rare1 rows); unfiltered, ~43% of the fold dev rows would be synthetic.
#   --max-oom-skip 0.05                as runs/phase1/seed: R1 skipped its OOM'd works_cycles batches and went on (the
#                                      0.2% default guard is newer than R1)
#   no --gate / --gate-total, no BIRD dev anything (no dev.json, no dev_databases, no bird_dev plans on this box).
# Then labelled k=32 --distinct candidate lists on fold F's questions with the best checkpoint, exactly as box W made
# runs/phase1/p1w/p1w/r1/cands_train.jsonl (rerank_sq2.py gen --split train: strict plan rows, exact-question repeats
# dropped, gen-time execution labels ok/runs/nonnull/sig/res), with --dev-dbs = the OTHER fold's drop list (so the
# "train" split = exactly fold F's DBs) and no --max-q (every fold-F question the gen script accepts).
# Output $O = /workspace/val4/fold<F>: cands_fold.jsonl, args.json, best.json, train_mix.json, oom.json, train_fold.log,
# gen_fold.log, projection.txt. The checkpoint stays in $R/val4_run/fold<F> (not pulled: teacher only).
# Ends VAL4FOLD_DONE (VAL4FOLD_FAIL <why> on any error). No BIRD dev, no sealed set, no RESERVED label is printed
# (fold F is BIRD-train only; the held-out DBs are not on this box).
# Preflight BEFORE setup uses only bash, coreutils, awk and the SYSTEM python3 (never $R/venv6, which setup creates).
# Env: FOLD (1|2), DRY=1 (echo every GPU/python command; with REMOTE_ROOT / REMOTE_D / PY / REPO pointing at local
#      paths it runs anywhere), PRICE_H (2.73, projection only), STEPS (6714; do not change for a real run).
set -uo pipefail
R=${REMOTE_ROOT:-/workspace}; D=${REMOTE_D:-/root/d}; S=$D/sys1; VF=$D/val4; L=$R/logs
V=${PY:-$R/venv6/bin/python}; REPO=${REPO:-$R/luthor}; DRY=${DRY:-0}
FOLD=${FOLD:-}; STEPS=${STEPS:-6714}; PRICE_H=${PRICE_H:-2.73}
SYSPY=$(PATH=/usr/local/bin:/usr/bin:/bin command -v python3 || true)  # system python only (not the venv)
log() { echo "$(date -u '+%m-%d %H:%M:%S') [val4 fold${FOLD}] $*"; }
die() { log "VAL4FOLD_FAIL $*"; exit 1; }
case "$FOLD" in 1) OTHER=2;; 2) OTHER=1;; *) echo "VAL4FOLD_FAIL FOLD must be 1 or 2 (got '$FOLD')"; exit 1;; esac
[[ "$STEPS" =~ ^[0-9]+$ ]] || die "bad STEPS '$STEPS'"
O=$R/val4/fold$FOLD; RUN=$R/val4_run/fold$FOLD; RARE=$D/rare1_fold$FOLD
DROP=$VF/fold${FOLD}_drop.json; DEV=$VF/fold${FOLD}_dev.json; GENX=$VF/fold${OTHER}_drop.json
mkdir -p "$L" "$O" "$RUN"
cd "$REPO" || die "no repo $REPO"
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 \
  HF_HUB_ENABLE_HF_TRANSFER=1
NP=$(nproc 2>/dev/null || sysctl -n hw.ncpu); W=$(( NP > 4 ? NP - 2 : 2 ))
x() {  # x NAME cmd...: all output in $L/NAME.log; DRY=1 only echoes; GATE lines are never echoed
  local n=$1; shift
  if [ "$DRY" = 1 ]; then log "DRY: $* > $L/$n.log"; return 0; fi
  log "run $n: $*"; "$@" > "$L/$n.log" 2>&1; local rc=$?
  [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 25 "$L/$n.log" | grep -av GATE | sed 's/Traceback/TB/'; }
  return $rc
}
# R1's command (r1r2_chain.sh stage 2, runs/r1r2/r1/args.json) with the VAL4 changes listed in the header
TRAIN=("$V" scripts/train_sq2.py --plans "$S/bird_train" "$S/spider_train" "$S/synsql_train" "$RARE"
       --db-roots "$S/rrdbs" --base seeklhy/OmniSQL-7B --subfolder "" --vocab "$S/templates.json" --dev-dbs "$DEV"
       --out "$RUN" --eval-every 1000 --budget-chars 9000 --col-mentions --repeat bird_train=3 --max-steps "$STEPS"
       --cand-order v2 --value-side --drop-dbs "$DROP" --max-oom-skip 0.05)
GEN=("$V" scripts/rerank_sq2.py gen --run "$RUN" --k 32 --distinct --workers "$W" --plans "$S/bird_train"
     --db-roots "$S/rrdbs" --dev-dbs "$GENX" --split train --out "$O/cands_fold.jsonl")
log "fold $FOLD (other $OTHER): drop $DROP, dev $DEV, gen excludes $GENX; steps $STEPS; workers $W"
log "train: ${TRAIN[*]}"
log "gen: ${GEN[*]}"
for a in "${TRAIN[@]}" "${GEN[@]}"; do  # nothing BIRD dev / sealed / gate may be on the command lines
  case "$a" in *bird_dev*|*dev_databases*|*sealed*|*gate*|*dev.json) [ "$a" = "$DEV" ] || die "forbidden path in a command: $a";; esac
done

# ---- boot + preflight (system tools only; seconds, before setup) ----------------------------------------------------
if [ "$DRY" != 1 ]; then
  g=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits) || die "nvidia-smi failed"
  log "GPUs: $g"
  echo "$g" | awk -F', *' '$3 + 0 > 512 { bad = 1 } END { exit !bad }' && die "GPU memory already in use at boot (> 512 MiB)"
  echo "$g" | head -1 | awk -F', *' '$4 + 0 < 79000 { bad = 1 } END { exit !bad }' && die "GPU 0 has < 80 GB (R1's recipe needs an 80 GB card)"
fi
for f in "$DROP" "$DEV" "$GENX" "$S/templates.json" "$S/templates_nums.json" "$D/rare1/plans.jsonl" \
         scripts/train_sq2.py scripts/rerank_sq2.py; do
  [ "$DRY" = 1 ] || [ -s "$f" ] || die "missing $f"
done
for p in "$S/bird_train" "$S/spider_train" "$S/synsql_train" "$D/rare1"; do
  [ "$DRY" = 1 ] || { [ -s "$p/plans.jsonl" ] && [ -d "$p/cards" ]; } || die "missing plan set $p (plans.jsonl + cards/)"
done
[ "$DRY" = 1 ] || [ -d "$S/rrdbs" ] || die "missing $S/rrdbs"
for p in "$D/dev_databases" "$D/dev.json" "$S/bird_dev" "$S/sealed" "$S/dbs"; do  # held-out / BIRD dev data never come here
  [ -e "$p" ] && die "$p is on this box: VAL4 fold boxes carry no held-out, BIRD dev or sealed data"
done
# the trainer / generator must know every flag this chain passes (grep the source: no venv before setup)
for f in --drop-dbs --dev-dbs --max-steps --cand-order --value-side --col-mentions --repeat --budget-chars --eval-every \
         --max-oom-skip --db-roots --vocab; do
  grep -q -- "\"$f\"" scripts/train_sq2.py || die "train_sq2.py has no $f (code not pushed?)"
done
for f in --distinct --split --dev-dbs --k --run --workers; do
  grep -q -- "\"$f\"" scripts/rerank_sq2.py || die "rerank_sq2.py has no $f (code not pushed?)"
done
if [ -n "$SYSPY" ] && [ -d "$S/rrdbs" ] && [ -s "$D/rare1/plans.jsonl" ] && [ -s "$DROP" ]; then  # (DRY: when staged)
  "$SYSPY" - "$DROP" "$DEV" "$GENX" "$S/rrdbs" "$D/rare1/plans.jsonl" "$FOLD" <<'PY' || die "fold list check failed"
import hashlib, json, os, sys
drop, dev, genx, rr, rare, fold = sys.argv[1:7]
L = lambda p: json.load(open(p))
d, v, x = L(drop), L(dev), L(genx)
assert all(isinstance(z, list) and all(isinstance(s, str) for s in z) for z in (d, v, x)), "lists must be plain JSON lists"
d, v, x = set(d), set(v), set(x)
assert v and v <= d, "dev DBs must all be dropped from training"
assert not (v & x), "a dev DB is excluded from generation"
held = d & x                       # the 9 held-out DBs are in both drop lists
assert len(held) == 9, f"expected the 9 held-out DBs in both drop lists, got {len(held)}"
assert not (held & v), "a held-out DB is a dev DB"
have = {n for n in os.listdir(rr) if os.path.exists(os.path.join(rr, n, n + ".sqlite"))}
assert v <= have, f"dev DBs without an SQLite: {sorted(v - have)[:5]}"
assert not (held & have), "a held-out DB has an SQLite in rrdbs (it must never be listed)"
gen_dbs = have - x
assert gen_dbs == v, f"gen would list {sorted(gen_dbs ^ v)[:5]} outside / missing from fold {fold}"
sha = hashlib.sha256(open(rare, "rb").read()).hexdigest()[:16]
assert sha == "c9ac3783cfb92e0a", f"rare1 plans.jsonl sha256 {sha} is not R1's c9ac3783cfb92e0a"
print(f"fold {fold}: {len(v)} dev/list DBs, {len(d)} dropped DBs ({len(d) - 9} fold + 9 held-out), rare1 = R1's rows")
PY
  "$SYSPY" - "$D/rare1" "$RARE" "$DROP" <<'PY' || die "rare1 fold filter failed"
import json, os, sys
src, dst, drop = sys.argv[1], sys.argv[2], set(json.load(open(sys.argv[3])))
os.makedirs(dst, exist_ok=True)
keep = gone = 0
with open(os.path.join(dst, "plans.jsonl.tmp"), "w") as out:
    for line in open(os.path.join(src, "plans.jsonl")):
        if json.loads(line)["db_id"] in drop:
            gone += 1
        else:
            keep += 1
            out.write(line)
os.replace(os.path.join(dst, "plans.jsonl.tmp"), os.path.join(dst, "plans.jsonl"))
cards = os.path.join(dst, "cards")
if not os.path.islink(cards):
    os.symlink(os.path.join(src, "cards"), cards)
print(f"rare1 for fold training: kept {keep} rows, removed {gone} on fold DBs -> {dst}")
PY
elif [ "$DRY" != 1 ]; then die "no system python3 for the preflight"
fi
# projection from earlier logs (H100 PCIe, R1 recipe: runs/phase1/seed/logs/train_r1.log 3.08 s/step, 6714 steps 5.75 h
# incl. dev evals; gen k=32 distinct: runs/phase1/p1w gen_train_r1 4545 questions in 60.4 min)
static_projection() {
  awk -v st="$STEPS" -v p="$PRICE_H" 'BEGIN { tr = st * 3.08 / 3600 * 1.0; gen = 0.6; setup = 0.35;
    printf "PROJECTION fold: setup %.2f h + train %.2f h (%d steps at 3.08 s/step, H100 PCIe) + gen %.2f h = %.1f h, ~$%.0f at $%.2f/h\n",
      setup, tr, st, gen, setup + tr + gen, (setup + tr + gen) * p, p }'
}
static_projection | tee -a "$O/projection.txt"
log "preflight ok"

# ---- setup (venv + OmniSQL-7B) ----------------------------------------------------------------------------------------
if [ "$DRY" != 1 ] && [ ! -e "$L/setup.ok" ]; then
  EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/setup6.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/'; die setup; }
  grep -qa '^SETUP6_FAIL' "$L/setup.log" && die "SETUP6_FAIL in setup.log"
  grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
  touch "$L/setup.ok"
fi
log DONE_SETUP
export PATH=$(dirname "$V"):$PATH
if [ "$DRY" != 1 ]; then
  [ -x "$V" ] || die "no venv python at $V after setup"
  h=$("$V" scripts/train_sq2.py --help 2>&1) || die "train_sq2.py --help failed: $(echo "$h" | tail -2)"
  for f in --drop-dbs --max-steps --cand-order --value-side --col-mentions --max-oom-skip; do
    echo "$h" | grep -q -- "$f" || die "train_sq2.py --help has no $f"
  done
  h=$("$V" scripts/rerank_sq2.py gen --help 2>&1) || die "rerank_sq2.py --help failed: $(echo "$h" | tail -2)"
  echo "$h" | grep -q -- --distinct || die "rerank_sq2.py gen --help has no --distinct"
fi

# ---- train (R1's recipe on the other fold + the non-BIRD sets) ----------------------------------------------------------
project() {  # PROJECTION lines after ~50 and ~500 steps from train_sq2's "step i/T ... X.XXs/step" lines
  local f=$L/train_fold.log seen=0 line st tot sps
  for mark in 50 500; do
    while :; do
      [ -e "$RUN/train.ok" ] && return 0
      line=$(grep -aE '^step [0-9]+/[0-9]+ .*s/step' "$f" 2>/dev/null | tail -1 || true)
      st=$(echo "$line" | sed -nE 's#^step ([0-9]+)/.*#\1#p'); tot=$(echo "$line" | sed -nE 's#^step [0-9]+/([0-9]+).*#\1#p')
      sps=$(echo "$line" | sed -nE 's#.* ([0-9.]+)s/step.*#\1#p')
      if [ -n "$st" ] && [ "$st" -ge "$mark" ] && [ "$st" -gt "$seen" ] && [ -n "$sps" ]; then seen=$st; break; fi
      sleep 60
    done
    awk -v st="$st" -v tot="$tot" -v sps="$sps" -v p="$PRICE_H" 'BEGIN { h = (tot - st) * sps / 3600 * 1.04 + 0.6;
      printf "PROJECTION fold: step %d/%d at %.2f s/step -> %.1f h left incl. gen, ~$%.0f\n", st, tot, sps, h, h * p }' \
      | tee -a "$O/projection.txt"
  done
}
if [ -e "$RUN/train.ok" ]; then log "training already done"
else
  PROJ_PID=""
  [ "$DRY" = 1 ] || { project & PROJ_PID=$!; }
  x train_fold env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${TRAIN[@]}" \
    || { [ -n "$PROJ_PID" ] && kill "$PROJ_PID" 2>/dev/null; die "train_sq2.py failed (train_fold.log)"; }
  [ -n "$PROJ_PID" ] && { kill "$PROJ_PID" 2>/dev/null || true; }
  if [ "$DRY" != 1 ]; then
    [ -d "$RUN/best_lora" ] && [ -s "$RUN/best_heads.pt" ] || die "no best_lora / best_heads.pt after training"
    grep -q '"gate": "None"' "$RUN/args.json" || die "args.json records a gate"
    touch "$RUN/train.ok"
  fi
fi
if [ "$DRY" != 1 ]; then
  grep -aE '^(drop dbs|WARNING --drop-dbs|repeat |training rows by source|train [0-9]+ /|schedule|eval step|done in|OOM:)' \
    "$L/train_fold.log" | grep -av GATE | cut -c1-400
  for f in args.json best.json train_mix.json oom.json vocab.json; do cp -f "$RUN/$f" "$O/" 2>/dev/null; done
  grep -av GATE "$L/train_fold.log" > "$O/train_fold.log"
fi
log DONE_TRAIN

# ---- labelled k=32 distinct lists on fold F's questions --------------------------------------------------------------
if [ -s "$O/cands_fold.jsonl" ]; then log "lists already written"
else
  x gen_fold "${GEN[@]}" || die "gen failed (gen_fold.log)"
fi
if [ "$DRY" != 1 ]; then
  [ -s "$O/cands_fold.jsonl" ] || die "no $O/cands_fold.jsonl"
  "$SYSPY" - "$O/cands_fold.jsonl" "$DEV" <<'PY' || die "list check failed"
import json, sys
dev = set(json.load(open(sys.argv[2])))
n = g = 0
dbs = set()
for line in open(sys.argv[1]):
    r = json.loads(line)
    assert r["db_id"] in dev, f"list on {r['db_id']}, not a fold DB"
    assert all("ok" in c for c in r["cands"]), "unlabelled candidate"
    n += 1; dbs.add(r["db_id"])
    cs = sorted(r["cands"], key=lambda c: c["rank"])
    g += bool(cs and cs[0].get("ok"))
print(f"LISTS {n} questions on {len(dbs)} fold DBs; greedy right {100 * g / max(n, 1):.1f}% (BIRD-train, fold DBs unseen)")
PY
  grep -aE '^(gen train|GEN )' "$L/gen_fold.log" | cut -c1-400
  cp -f "$L/gen_fold.log" "$O/" 2>/dev/null
fi
log VAL4FOLD_DONE
