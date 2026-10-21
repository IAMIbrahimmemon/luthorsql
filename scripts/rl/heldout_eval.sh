#!/bin/bash
# Held-out evaluation of an RL run (GPU box, from the repo root), exactly as R2's chain scored itself
# (scripts/chains/r1r2_chain.sh stage 3), so the numbers compare with runs/r1r2/r2 directly:
#   1. k=8 candidate lists on the 1,000 held-out BIRD-train questions (the 9 databases in dev_dbs_bird.json, never
#      trained on): GEN line = first pick ("greedy"), fallback, oracle@8 ("oracle").
#   2. the FROZEN R2 validator (runs/r1r2/r2/rr) scores those lists: SCORE line = validator pick ("rerank").
#   3. side by side with R2's own numbers (runs/r1r2/r2/rr/dev_metrics.json: greedy 0.606, oracle 0.712, rerank 0.674).
# Pass rule (docs/SYSTEM1_FANOUT_R3.md rung 1b): >= +1.5 on the pick with oracle@8 down no more than 1 point.
#
#   bash scripts/rl/heldout_eval.sh runs/rl1/grpo            # rl_train.py --heldout-eval calls this
# Env: S (data dir, default data/sys1), R2 (default runs/r1r2/r2), V (python, default python), W (exec workers),
#   BASELINE=1 also re-scores R2's own cands_dev.jsonl with the same command (same-code baseline).
# BIRD dev and the sealed sets are not touched.
set -euo pipefail
O=${1:?usage: heldout_eval.sh RL_RUN_DIR}
S=${S:-data/sys1}; R2=${R2:-runs/r1r2/r2}; V=${V:-python}
NP=$(nproc 2>/dev/null || sysctl -n hw.ncpu); W=${W:-$(( NP > 4 ? NP - 2 : 2 ))}
for f in "$O/best_lora/adapter_model.safetensors" "$O/best_heads.pt" "$O/args.json" "$O/vocab.json" \
         "$R2/rr/rerank_head.pt" "$S/dev_dbs_bird.json"; do
  [ -e "$f" ] || { echo "missing $f"; exit 2; }
done
mkdir -p "$O/rr"
[ -s "$O/rr/cands_dev.jsonl" ] || "$V" scripts/rerank_sq2.py gen --run "$O" --k 8 --workers "$W" \
  --plans "$S/bird_train" --db-roots "$S/dbs" --dev-dbs "$S/dev_dbs_bird.json" --split dev \
  --out "$O/rr/cands_dev.jsonl" 2>&1 | tee "$O/rr/gen_dev.log"
"$V" scripts/rerank_sq2.py score --rr "$R2/rr" --cards "$S/bird_train" --dev-cands "$O/rr/cands_dev.jsonl" \
  --out "$O/rr/dev_scored.jsonl" --base seeklhy/OmniSQL-7B --subfolder "" 2>&1 | tee "$O/rr/score_dev.log"
if [ "${BASELINE:-0}" = 1 ]; then
  "$V" scripts/rerank_sq2.py score --rr "$R2/rr" --cards "$S/bird_train" --dev-cands "$R2/rr/cands_dev.jsonl" \
    --out "$O/rr/r2_dev_scored.jsonl" --base seeklhy/OmniSQL-7B --subfolder "" 2>&1 | tee "$O/rr/score_r2_dev.log"
fi
python3 - "$O" "$R2" <<'EOF'
import json, re, sys
o, r2 = sys.argv[1:3]
def line(p, tag):
    try:
        ls = [l for l in open(p, errors="replace") if re.match(rf"^{tag} (dev )?\{{", l)]
    except OSError:
        return None
    return json.loads(re.sub(rf"^{tag} (dev )?", "", ls[-1]).split(" in ")[0]) if ls else None
rl = {**(line(f"{o}/rr/gen_dev.log", "GEN") or {}), **(line(f"{o}/rr/score_dev.log", "SCORE") or {})}
base = json.load(open(f"{r2}/rr/dev_metrics.json"))
b2 = line(f"{o}/rr/score_r2_dev.log", "SCORE")
if b2:
    base = {**base, "rerank_same_code": b2.get("rerank")}
out = {"rl": rl, "r2": base, "delta_points": {k: round(100 * (rl[k] - base[k]), 2) for k in ("greedy", "oracle", "rerank")
                                              if k in rl and k in base}}
open(f"{o}/rr/heldout_compare.json", "w").write(json.dumps(out, indent=1))
print("HELDOUT", json.dumps(out))
EOF
