#!/bin/bash
# Round-3 question generation, unattended: for BIRD then Spider train databases,
# sample plans (400/db) -> drop plans already used in round 2 -> batches -> DeepSeek writes ->
# top-up -> blind checks. Status lines go to reports/qgen3_status.log. Public data only.
cd "$(dirname "$0")/.."
V3=/Volumes/MacSSD/AI/datasets/text-to-sql/plans_v3
V2=/Volumes/MacSSD/AI/datasets/text-to-sql/plans_v2
PY=.venv/bin/python
log() { echo "$(date '+%F %T') $*" >> reports/qgen3_status.log; }

new_plans() {  # $1 sampled plans, $2 round-2 plans_with_ids, $3 output, $4 cards dir
  $PY - "$1" "$2" "$3" "$4" <<'PYEOF'
import json, sys, collections
sys.path.insert(0, "scripts")
from quality import fix_fan_out, limit_without_order
from luthor.sq.plan import Plan, compile_plan
from luthor.sq.schema_card import SchemaCard
old = {json.loads(l)["sql"] for l in open(sys.argv[2])}
cards, st = {}, collections.Counter()
with open(sys.argv[3], "w") as out:
    for l in open(sys.argv[1]):
        r = json.loads(l); st["sampled"] += 1
        if r["sql"] in old: st["seen_in_round2"] += 1; continue
        if limit_without_order(r["plan"]): st["limit_without_order"] += 1; continue
        card = cards.setdefault(r["db_id"], SchemaCard.load(f"{sys.argv[4]}/{r['db_id']}.json"))
        p, s = fix_fan_out(r["plan"], card)
        if p is None: st["fan_out_dropped"] += 1; continue
        if s == "rewritten":  # the plan now counts distinct keys; its SQL and English must follow
            st["fan_out_rewritten"] += 1
            r["plan"] = p; r["sql"] = compile_plan(Plan.from_dict(p), card.graph())
            r["english"] = r["english"].replace("the number of rows of", "the number of distinct")
        out.write(json.dumps(r) + "\n"); st["kept"] += 1
print(dict(st))
PYEOF
}

cap_per_db() {  # $1 plans file, $2 max new plans per database (in place)
  $PY - "$1" "$2" <<'PYEOF'
import json, sys, random, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
random.Random(0).shuffle(rows)
n = collections.Counter(); keep = []
for r in rows:
    if n[r["db_id"]] < int(sys.argv[2]):
        n[r["db_id"]] += 1; keep.append(r)
open(sys.argv[1], "w").write("".join(json.dumps(r) + "\n" for r in keep))
print(f"capped to {len(keep)} plans")
PYEOF
}

run_src() {  # $1 name, $2 cards dir, $3 db root, $4 round-2 qgen dir, $5 max new plans per db
  local D=data/qgen3/$1
  mkdir -p $D
  if [ ! -s $D/plans_all.jsonl ] || pgrep -f "sample_plans.py.*$D/plans_all" >/dev/null; then
    while pgrep -f "sample_plans.py.*$D/plans_all" >/dev/null; do sleep 60; done
  fi
  if [ ! -s $D/plans_all.jsonl ]; then
    log "$1: sampling"
    $PY scripts/sample_plans.py --cards $2 --db-root $3 --shapes $V2/bird_train/plans.jsonl $V2/spider_train/plans.jsonl \
      --per-db 400 --out $D/plans_all.jsonl > reports/qgen3_$1_sample.log 2>&1
  fi
  log "$1: $(new_plans $D/plans_all.jsonl data/qgen/$4/ds/plans_with_ids.jsonl $D/plans.jsonl $2)"
  log "$1: $(cap_per_db $D/plans.jsonl $5)"
  if [ ! -d $D/ds ]; then
    $PY scripts/qgen.py batches --plans $D/plans.jsonl --cards $2 --out $D/ds --size 25 >> reports/qgen3_$1_sample.log 2>&1
  fi
  log "$1: writing $(ls $D/ds/writer_*.json | wc -l) batches"
  $PY -u scripts/qgen_ollama.py write --batches $D/ds --workers 2 > reports/qgen3_$1_write.log 2>&1
  log "$1: top-up"
  $PY -u scripts/qgen_ollama.py topup --batches $D/ds --workers 2 > reports/qgen3_$1_topup.log 2>&1
  $PY scripts/qgen.py checks --batches $D/ds > /dev/null
  log "$1: checking"
  $PY -u scripts/qgen_ollama.py check --batches $D/ds --workers 2 > reports/qgen3_$1_check.log 2>&1
  $PY -u scripts/qgen_ollama.py check --batches $D/ds --workers 2 >> reports/qgen3_$1_check.log 2>&1  # retry gaps
  log "$1: $($PY scripts/qgen.py verifies --batches $D/ds --cards $2)"
  $PY -u scripts/qgen_ollama.py verify --batches $D/ds --workers 2 > reports/qgen3_$1_verify.log 2>&1
  $PY -u scripts/qgen_ollama.py verify --batches $D/ds --workers 2 >> reports/qgen3_$1_verify.log 2>&1  # retry gaps
  log "$1: done"
}

run_src bird $V2/bird_train/cards data/raw/bird_train/train_databases bird 400
run_src spider $V3/spider_train/cards data/raw/spider_data/database spider 100
log "ALL_DONE"
