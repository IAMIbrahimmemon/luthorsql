#!/bin/bash
# Build the BIRD submission package (Mac, no GPU):
#   bash scripts/bird_submit/build_package.sh
# -> data/bird_submit/luthorsql_submission/  {README.md, run_bird.sh, requirement.txt, luthor/<code the pipeline imports>}
#    data/bird_submit/luthorsql_submission.zip
#    data/bird_submit/weights/  {r1/{best_lora,best_heads.pt,args.json,vocab.json}, valhard_rr/{rerank_head.pt,
#                                settings.json,rerank_lora/rerank/}}   (uploaded separately, privately)
#    data/bird_submit/dev_blind/{questions.json,dev_tables.json}  BIRD dev questions with the SQL field removed (box check)
# Code = the import closure of predict.py (checked by importing it from the package copy). Every file is scanned with
# every leak pattern (all keys of ~/.config/luthor/leak_patterns + secrets); any hit stops the build.
set -euo pipefail
cd "$(dirname "$0")/../.."
B=data/bird_submit; P=$B/luthorsql_submission; WT=$B/weights
CODE=(scripts/bird_submit/predict.py scripts/devlook/blind_gen.py scripts/eval_sq2.py scripts/prepare_plans.py
      scripts/rerank_sq2.py scripts/train_sq2.py
      src/luthor/__init__.py src/luthor/grammar_cli.py
      src/luthor/sq/__init__.py src/luthor/sq/bird_desc.py src/luthor/sq/candidates.py src/luthor/sq/convert.py
      src/luthor/sq/model.py src/luthor/sq/plan.py src/luthor/sq/schema_card.py
      src/luthor/sq2/model.py src/luthor/sq2/prefix_cache.py src/luthor/sq2/rerank.py src/luthor/sq2/serialize.py
      src/luthor/sq2/targets.py src/luthor/sq2/topk.py src/luthor/sq2/train.py)
REQ=runs/phase1/bird_a100/latency/requirement.txt
rm -rf "$P" "$WT" "$B/dev_blind" "$B/luthorsql_submission.zip"
mkdir -p "$P/luthor" "$WT/r1" "$WT/valhard_rr/rerank_lora/rerank" "$B/dev_blind"
for f in "${CODE[@]}"; do mkdir -p "$P/luthor/$(dirname "$f")"; cp "$f" "$P/luthor/$f"; done
cp scripts/bird_submit/run_bird.sh "$P/"
if [ -n "${SINGLE:-}" ]; then cp scripts/bird_submit/README_single.md "$P/README.md"; else cp scripts/bird_submit/README.md "$P/"; fi
# pip freeze of the A100 / CUDA 12.2 run, minus the editable repo line; torch without the +cu128 local tag (the index
# is chosen by --torch-backend / --extra-index-url)
grep -v '^-e ' "$REQ" | sed -E 's/^torch==([0-9.]+)\+cu[0-9]+$/torch==\1/' > "$P/requirement.txt"
# weights: SINGLE=<single_lora.py out dir> packs the one-adapter model as weights/single/; otherwise R1 + VALHARD
if [ -n "${SINGLE:-}" ]; then
  rm -rf "$WT/r1" "$WT/valhard_rr"; mkdir -p "$WT/single"
  cp -R "$SINGLE/best_lora" "$WT/single/"
  cp "$SINGLE"/{best_heads.pt,args.json,vocab.json,rerank_head.pt,settings.json} "$WT/single/"
  python3 -c "import json,sys; assert json.load(open('$WT/single/settings.json'))['shared_adapter']" || { echo "BUILD_FAIL $SINGLE is not a shared-adapter dir"; exit 1; }
else
  cp -R runs/r1r2/r1/best_lora "$WT/r1/"; cp runs/r1r2/r1/{best_heads.pt,args.json,vocab.json} "$WT/r1/"
  V=runs/phase1/valhard/valhard/rr
  cp $V/rerank_head.pt $V/settings.json "$WT/valhard_rr/"; cp $V/rerank_lora/rerank/* "$WT/valhard_rr/rerank_lora/rerank/"
fi
# blind dev questions for the box check (no SQL field)
python3 - <<'EOF'
import json
q = json.load(open("data/bench/dev_20240627/dev.json"))
out = [{k: r[k] for k in ("question_id", "db_id", "question", "evidence", "difficulty") if k in r} for r in q]
assert not any("SQL" in r for r in out)
json.dump(out, open("data/bird_submit/dev_blind/questions.json", "w"), indent=1)
print(f"dev_blind: {len(out)} questions, no SQL")
EOF
cp data/bench/dev_20240627/dev_tables.json "$B/dev_blind/"
# the package imports cleanly from its own copy (closure check; GPU modules are imported, not run)
( cd "$P/luthor" && LUTHOR_GRAMMAR=v3 "$OLDPWD/.venv/bin/python" -I - <<'EOF'
import sys
sys.path[:0] = ["scripts/bird_submit", "scripts/devlook", "scripts", "src"]
import predict  # noqa
from luthor.grammar_cli import preset; preset()
import prepare_plans, blind_gen, rerank_sq2, train_sq2, eval_sq2  # noqa
import luthor.sq.plan, luthor.sq2.model, luthor.sq2.topk, luthor.sq2.train, luthor.sq2.rerank  # noqa
import luthor.sq2.prefix_cache, luthor.sq.schema_card, luthor.sq.candidates  # noqa
bad = [m.__file__ for m in list(sys.modules.values()) if getattr(m, "__file__", None) and "luthor-general" in m.__file__
       and "/luthorsql_submission/" not in m.__file__ and "/.venv/" not in m.__file__]
assert not bad, f"imported from outside the package: {bad}"
print("import closure ok")
EOF
)
# CASCADE=1: add stage 2 (Claude Opus 5.5 on System 1's unsure questions) and use the cascade README
if [ -n "${CASCADE:-}" ]; then
  mkdir -p "$P/cascade"
  cp scripts/bird_submit/cascade/{cascade.py,PROMPT_routed.md,sql_tool.py,requirement_api.txt} "$P/cascade/"
  cp scripts/bird_submit/cascade/run_cascade.sh "$P/"
  cp scripts/bird_submit/cascade/README_cascade.md "$P/README.md"
  cmp -s scripts/bird_submit/cascade/PROMPT_routed.md runs/analysis/hybrid/opus_s1_raw/PROMPT_s1.md || { echo "BUILD_FAIL routed prompt differs from the paper's"; exit 1; }
  cmp -s scripts/bird_submit/cascade/sql_tool.py scripts/system2/sql_tool.py || { echo "BUILD_FAIL SQL tool differs from the paper's"; exit 1; }
fi
# leak scan: every pattern key on every file (package + weights text files)
LP=${LEAK_PATTERNS:-$HOME/.config/luthor/leak_patterns}
SECRET='\bhf_[A-Za-z0-9]{30,}|\bsk-(ant-|proj-)?[A-Za-z0-9_-]{30,}|\bAKIA[0-9A-Z]{16}\b|\bghp_[A-Za-z0-9]{30,}|BEGIN [A-Z ]*PRIVATE KEY'
hits=0
while IFS= read -r line; do
  [[ "$line" == *=* ]] || continue
  pat=${line#*=}; [ -n "$pat" ] || continue
  n=$(grep -rIEil --exclude='*.safetensors' --exclude='*.pt' -- "$pat" "$P" "$WT" "$B/dev_blind" 2>/dev/null | wc -l | tr -d ' ' || true)
  [ "$n" = 0 ] || { echo "LEAK: pattern key ${line%%=*} in $n file(s)"; hits=$((hits + n)); }
done < "$LP"
n=$(grep -rIEl --exclude='*.safetensors' --exclude='*.pt' -- "$SECRET" "$P" "$WT" 2>/dev/null | wc -l | tr -d ' ' || true)
[ "$n" = 0 ] || { echo "LEAK: secret pattern in $n file(s)"; hits=$((hits + n)); }
[ "$hits" = 0 ] || { echo "BUILD_FAIL leak scan"; exit 1; }
echo "leak scan ok"
find "$P" -name __pycache__ -prune -exec rm -rf {} +
ZIP=luthorsql_submission.zip; [ -n "${CASCADE:-}" ] && ZIP=luthorsql_cascade_submission.zip
( cd "$B" && rm -f "$ZIP" && zip -qr -X "$ZIP" luthorsql_submission -x '*.DS_Store' -x '*/venv_api/*' )
echo "BUILD_OK $(du -sh "$B/$ZIP" | cut -f1) zip, $(du -sh "$WT" | cut -f1) weights"
