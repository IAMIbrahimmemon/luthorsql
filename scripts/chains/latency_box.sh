#!/bin/bash
# System-1 latency (frozen System 1 = R1 generator + VALHARD validator), ON the box (one H100, ~1 h):
#   PRICE_H=2.50 setsid nohup bash /workspace/luthor/scripts/chains/latency_box.sh > /workspace/logs/latency.log 2>&1 < /dev/null &
# (started by scripts/chains/run_latency_box.sh on the Mac, which stages + pushes everything below first.)
#
# One in-process run of scripts/analysis/latency/sys1_latency.py run on a seeded sample of 200 DECISION questions
# (+ 3 warm-ups): single-query latency (batch 1) for greedy k=1, k=32 + VALHARD, k=64 + repair + VALHARD with a
# per-stage breakdown; throughput of the chains' batched paths (k=32, k=64+repair) and $ per 1,000 questions at PRICE_H;
# GPU / versions / peak memory -> /workspace/latency/{latency.json,latency.txt,rows.jsonl}. Ends LATENCY_DONE
# (LATENCY_FAIL <why> on errors). Nothing is trained.
# LATENCY ONLY: the box holds no label (question rows are id / db_id / question / context; no gold SQL, plan or ok), so
# no accuracy can be computed or printed. No BIRD dev, no sealed set, no RESERVED row.
# Preflight BEFORE setup uses only bash, coreutils, awk and the SYSTEM python3 (never $R/venv6, which setup creates).
#
# Box layout (run_latency_box.sh):
#   /workspace/luthor/{src,scripts,pyproject.toml}  /workspace/luthor/runs/r1r2/r1/{best_lora,best_heads.pt,args.json,vocab.json}
#   /root/d/latency/{qs/{plans.jsonl,cards/},sample.json,rr_valhard/,MANIFEST.sha256}
#   /root/d/sys1/dbs/<8 DECISION dbs>/<db>.sqlite
# The run step logs to $L/sys1lat.log (naming it latency.log overwrote this script's own log and lost LATENCY_DONE).
# Env: SETUP (setup6 | setup_bird: the BIRD A100 / CUDA 12.2 env check), PRICE_H ($/h of this box, 2.50), N (0 = all 200), SQ_MAX_MIN (single-query time cap, 35), DRY=1 (echo the commands
#   only; with REMOTE_ROOT / REMOTE_D / PY / REPO pointing at local paths it runs anywhere).
set -uo pipefail
R=${REMOTE_ROOT:-/workspace}; D=${REMOTE_D:-/root/d}; LD=$D/latency; S=$D/sys1; L=$R/logs; O=$R/latency
V=${PY:-$R/venv6/bin/python}; REPO=${REPO:-$R/luthor}; RUN=runs/r1r2/r1; DRY=${DRY:-0}
PRICE_H=${PRICE_H:-2.50}; N=${N:-0}; SQ_MAX_MIN=${SQ_MAX_MIN:-35}
SYSPY=$(PATH=/usr/local/bin:/usr/bin:/bin command -v python3 || true)  # system python only (not the venv)
if command -v sha256sum >/dev/null; then SHA=(sha256sum); else SHA=(shasum -a 256); fi  # (shasum: DRY runs on a Mac)
W=$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8) - 2 )); [ "$W" -lt 2 ] && W=2
mkdir -p "$L" "$O"
log() { echo "$(date -u '+%m-%d %H:%M:%S') [latency] $*"; }
die() { log "LATENCY_FAIL $*"; exit 1; }
cd "$REPO" || die "no repo $REPO"
export LUTHOR_GRAMMAR=v3 HF_HOME=$R/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
x() {  # x NAME cmd...: output in $L/NAME.log; DRY=1 only echoes. No label exists here; label-looking lines are dropped anyway.
  local n=$1; shift
  if [ "$DRY" = 1 ]; then log "DRY: $* > $L/$n.log"; return 0; fi
  log "run $n"; "$@" > "$L/$n.log" 2>&1; local rc=$?
  sed -i.bak -E '/^(GEN |SCORE|RERANK DEV |greedy|oracle)/d' "$L/$n.log" 2>/dev/null; rm -f "$L/$n.log.bak"
  [ $rc = 0 ] && log "ok $n" || { log "$n exit $rc"; tail -n 20 "$L/$n.log" | grep -avE '^(GEN|SCORE)' | sed 's/Traceback/TB/'; }
  return $rc
}
[[ "$PRICE_H" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "bad PRICE_H '$PRICE_H'"
[[ "$N" =~ ^[0-9]+$ ]] || die "bad N '$N'"
[[ "$SQ_MAX_MIN" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "bad SQ_MAX_MIN '$SQ_MAX_MIN'"
log "frozen System 1: generator R1 ($RUN, grammar v3), validator VALHARD ($LD/rr_valhard); price \$$PRICE_H/h, workers $W -> $O"

# ---- preflight (system tools only; seconds, before setup) ------------------------------------------------------------
if [ "$DRY" != 1 ]; then
  g=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits) || die "nvidia-smi failed"
  log "GPUs: $g"
  echo "$g" | awk -F', *' '$3 + 0 > 512 { bad = 1 } END { exit !bad }' && die "GPU memory already in use at boot (> 512 MiB)"
fi
for f in $RUN/best_lora/adapter_model.safetensors $RUN/best_heads.pt $RUN/args.json $RUN/vocab.json \
         $LD/qs/plans.jsonl $LD/sample.json $LD/MANIFEST.sha256 $LD/rr_valhard/rerank_head.pt \
         $LD/rr_valhard/rerank_lora/rerank/adapter_model.safetensors scripts/analysis/latency/sys1_latency.py \
         scripts/rerank_sq2.py scripts/devlook/blind_gen.py scripts/repair/repair_gen.py; do
  [ -s "$f" ] || die "missing $f"
done
[ "$(find "$LD/qs/cards" -name '*.json' | wc -l | tr -d ' ')" -eq 8 ] || die "expected 8 DECISION cards under $LD/qs/cards"
[ "$(find "$S/dbs" -name '*.sqlite' | wc -l | tr -d ' ')" -eq 8 ] || die "expected the 8 DECISION sqlite files under $S/dbs"
( cd "$LD" && "${SHA[@]}" --status -c MANIFEST.sha256 ) || die "staged files do not match MANIFEST.sha256 (partial push?)"
# nothing gold-like / BIRD dev / sealed / RESERVED on this box
[ -z "$(find "$D" \( -name 'dev.json' -o -name 'dev_*.json' -o -name 'reserved*.json' -o -name '*.sql' -o -name '*gold*' \
        -o -name 'dev_databases' -o -name 'sealed*' \) 2>/dev/null | head -1)" ] || die "a gold / dev / sealed / reserved file is on the box under $D: refusing"
grep -qE '"(ok|gold_sql|SQL|plan|compiled_sql|covered|in_slots|value_labels)": ' "$LD/qs/plans.jsonl" \
  && die "$LD/qs/plans.jsonl carries a label field: refusing"
if [ -n "$SYSPY" ]; then
  "$SYSPY" - "$LD" "$S/dbs" "$RUN/args.json" <<'PY' || die "staged data check failed"
import json, sys
from pathlib import Path
ld, dbs, args = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
s = json.loads((ld / "sample.json").read_text())
assert s["split"] == "decision" and len(s["warmup"]) == 3 and len(s["measured"]) == 200, "sample.json"
rows = [json.loads(l) for l in open(ld / "qs/plans.jsonl")]
assert [r["id"] for r in rows] == s["warmup"] + s["measured"], "plans.jsonl rows != sample ids"
for r in rows:
    assert set(r) <= {"id", "db_id", "question", "context", "outcome"} and r["outcome"] == "missing", r["id"]
have = {p.stem for p in dbs.rglob("*.sqlite")}
assert {r["db_id"] for r in rows} <= have == set(s["databases"]), "DBs"
assert {p.stem for p in (ld / "qs/cards").glob("*.json")} == set(s["databases"]), "cards"
g = json.load(open(args)).get("grammar")
assert g in (None, "", "None", "v3"), "R1 must be a grammar-v3 run"
print(f"staged: {len(rows)} label-free DECISION rows on {len(have)} DBs; R1 grammar v3")
PY
elif [ "$DRY" != 1 ]; then die "no system python3 for the preflight"
fi
log "preflight ok"

# ---- setup (venv + OmniSQL-7B) ----------------------------------------------------------------------------------------
if [ "$DRY" != 1 ] && [ ! -e "$L/setup.ok" ]; then
  EXPECT_GPUS=1 MODELS="seeklhy/OmniSQL-7B" bash scripts/vast/${SETUP:-setup6}.sh > "$L/setup.log" 2>&1 || { tail -n 30 "$L/setup.log" | sed 's/Traceback/TB/'; die setup; }
  grep -qa '^DONE_SETUP6' "$L/setup.log" || die "setup (no DONE_SETUP6)"
  touch "$L/setup.ok"
fi
log DONE_SETUP
export PATH=$(dirname "$V"):$PATH
if [ "$DRY" != 1 ]; then
  [ -x "$V" ] || die "no venv python at $V after setup"
  h=$("$V" scripts/analysis/latency/sys1_latency.py run --help 2>&1) || die "sys1_latency.py --help failed: $(echo "$h" | tail -2)"
  echo "$h" | grep -q -- "--price-h" || die "sys1_latency.py run has no --price-h (code not pushed?)"
  grep -q "def load_validator" scripts/rerank_sq2.py || die "rerank_sq2.py has no load_validator (code not pushed?)"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- measure ----------------------------------------------------------------------------------------------------------
NARG=(); [ "$N" != 0 ] && NARG=(--n "$N")
[ -s $O/latency.json ] || x sys1lat "$V" scripts/analysis/latency/sys1_latency.py run --run $RUN --data "$LD" \
  --db-roots "$S/dbs" --out "$O" --price-h "$PRICE_H" --workers "$W" --sq-max-min "$SQ_MAX_MIN" ${NARG[@]+"${NARG[@]}"} \
  || die latency
if [ "$DRY" != 1 ]; then
  [ -s $O/latency.json ] && [ -s $O/latency.txt ] || die "no latency.json / latency.txt in $O"
  cp -f "$L/sys1lat.log" "$O/" 2>/dev/null
  cp -f "$L/bird_env.txt" "$L/requirement.txt" "$O/" 2>/dev/null
  cat $O/latency.txt
fi
log LATENCY_DONE
