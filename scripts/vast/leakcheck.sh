#!/bin/bash
# Gate for anything leaving the Mac for a Vast.ai (community) host. Exit 0 only if every path is inside the
# public allowlist and no file name or text file content matches a work/secret pattern.
#   bash scripts/vast/leakcheck.sh <path>...
# Code is scanned for work terms and personal names too; data (public datasets) for work terms and secrets.
set -euo pipefail
cd "$(dirname "$0")/../.."
# Public sources only. Anything else (reports/, docs/, runs/ other than these, .env, ~/...) is refused. The R1/R2
# checkpoints (LoRA + heads trained on public data only) go by exact file / dir name; their text files are scanned too.
# data/raw/bird_train_desc: BIRD-train database_description CSVs from the public BIRD release (phase 3 v4desc).
# data/valhard: VALHARD validator lists (BIRD-train lists + hard negatives), the held-out k=32 pool (as data/val2) and a
#   copy of a generator's own validator adapter (base_rr, trained on public data only).
# data/repair: System-1 repair stage inputs (scripts/chains/run_repair_box.sh): R1's own held-out k=32 pool (a copy of
#   data/valhard/r1/heldout_pool.jsonl) and copies of two R1-family validator adapters (trained on public BIRD-train only).
# data/ff1: the free-form pilot inputs (DECISION grammar pool + BIRD-train ids, mutants of grammar SQL): BIRD-train derived.
# data/devlook: BIRD dev look #1 box inputs (scripts/devlook/prep.py): R1's BIRD dev k=32 pool reduced to SQL + execution
#   facts (no label field: prep.py audit), the public BIRD dev questions/evidence + schema cards (no plan or SQL), and a copy
#   of the VALHARD validator adapter (trained on public BIRD-train only).
# data/v4guard: v4guard re-decode inputs (scripts/chains/run_v4guard_box.sh): copies of the phase-3 v4desc generator
#   checkpoint (LoRA + heads + args/vocab), its own validator and VALHARD-v4desc adapters (all trained on public BIRD-train
#   / Spider / SynSQL only) and v4desc's own held-out k=8 list.
# data/val3: VAL3 data-v3 box inputs (scripts/chains/run_val3_lists_box.sh): new BIRD-train / SynSQL-2.5M question
#   sets as plan rows + their schema cards (scripts/val3/sources.py), hard links to public SynSQL SQLite files, and the
#   built validator lists (BIRD-train + SynSQL derived). No held-out, BIRD dev or sealed data.
# data/ff2: free-form phase 2 inputs (BIRD-train subset questions, grammar lists, relabelled arithmetic mutants, phase-1
#   DECISION pools): BIRD-train derived (scripts/freeform/ff2_stage.py).
# data/val4: VAL4 inputs (runs/phase1/val4/PREREG.md): the BIRD-train fold split + per-fold DB lists (scripts/val4/folds.py;
#   db_id names only) and the cross-fitted validator lists built from the fold generators' BIRD-train k=32 lists
#   (scripts/val4/build_lists.sh) plus label-free copies of R1's held-out pools (scripts/val4/stage.py, as data/valhard/val3).
# data/latency: System-1 latency box inputs (scripts/analysis/latency/sys1_latency.py stage): a seeded sample of DECISION
#   question rows reduced to id / db_id / question / context (no SQL, plan or label), the 8 DECISION DBs' schema cards and
#   a copy of the VALHARD validator adapter (trained on public BIRD-train only). No RESERVED, BIRD dev or sealed data.
# data/single_soup: weight average of single-LoRA runs (scripts/soup_lora.py; R1 + VALHARD-list training, public data only).
# data/bird_submit: the BIRD submission package (scripts/bird_submit/build_package.sh): a copy of src/scripts files, the
#   R1 + VALHARD weights (public data only) and BIRD dev questions with the SQL field removed (dev_blind/).
ALLOW='^(src|scripts|pyproject\.toml|data/bird_submit|data/raw/(bird_train|bird_train_desc|bird_filtered|spider_data|spider_syn|sparc|cosql|sparc_cosql|synsql|wikisql|kaggledbqa|spider_dk|spider_realistic)|data/bench|data/eval|data/rl6|data/sys1|data/plans|data/eval_plans_v2|data/sqlw5|data/curr|data/sel|data/rare1|data/ens1|data/plans_v4|data/decider|data/val2|data/val3|data/val4|data/latency|data/valhard|data/repair|data/v4guard|data/ens3|data/ff1|data/ff2|data/devlook|data/single_soup|data/teacher/export|runs/r1r2/rare_merge|runs/r1r2/(r1|r2)/(best_lora|best_heads\.pt|last_lora|last_heads\.pt|args\.json|vocab\.json|rr)|runs/phase1/p1w/p1w/(r1/|r2/)?(cands_train|cands_dev|heldout_union)\.jsonl|runs/phase1/rl_[a-z0-9]+/out/[a-z0-9]+/(best_lora|best_heads\.pt|args\.json|vocab\.json|rl_args\.json)|runs/sqlw5|runs/sqlw_sft|runs/sqlw_rl)(/|$)'
# Whole words only. Some private terms are ordinary words in public datasets (names, astronomy),
# so CODE_WORK terms are checked in code only; AWS's documented ...EXAMPLE keys and truncated key placeholders don't match.
# Private terms (employer, internal tools, owner name) live outside the repo, one ERE per line:
#   ~/.config/luthor/leak_patterns  (lines "WORK=<regex>", "CODE_WORK=<regex>", "PERSON=<regex>")
# Without that file, only secrets and forbidden file names are checked.
LP=${LEAK_PATTERNS:-$HOME/.config/luthor/leak_patterns}
lp() { [ -r "$LP" ] && sed -n "s/^$1=//p" "$LP" | head -1; }
WORK=$(lp WORK); WORK=${WORK:-__no_private_terms_configured__}
CODE_WORK=$(lp CODE_WORK); CODE_WORK=${CODE_WORK:-__no_private_terms_configured__}
SECRET='BEGIN [A-Z ]*PRIVATE KEY-----(\\\\n|\\n|[[:space:]])*[A-Za-z0-9+/]{40}|\bhf_[A-Za-z0-9]{30,}|\bsk-(ant-|proj-)?[A-Za-z0-9_-]{30,}|tskey-[a-z]+-[A-Za-z0-9]{10,}|\bAKIA[0-9A-Z]{16}\b|\bghp_[A-Za-z0-9]{30,}|\bxox[abp]-[0-9A-Za-z-]{10,}'
PERSON=$(lp PERSON); PERSON=${PERSON:-__no_private_terms_configured__}
BADNAME='(^|/)(\.env[^/]*|id_rsa[^/]*|id_ed25519[^/]*|[^/]*\.pem|[^/]*\.key|credentials[^/]*|\.netrc|\.Rhistory)$'  # .DS_Store is never sent (push6 excludes it)
fail=0
for p in "$@"; do
  rel=${p#./}; rel=${rel%/}
  [[ "$rel" =~ $ALLOW ]] || { echo "LEAKCHECK: path not in public allowlist: $p"; fail=1; continue; }
  [ -e "$p" ] || { echo "LEAKCHECK: missing: $p"; fail=1; continue; }
  # names
  if bad=$(find -L "$p" -type f ! -name .DS_Store 2>/dev/null | grep -E "$BADNAME" | head -5) && [ -n "$bad" ]; then
    echo "LEAKCHECK: forbidden file names under $p:"; echo "$bad"; fail=1
  fi
  # contents of text files (sqlite and weights are binary and skipped by -I)
  pat="$WORK|$SECRET"; [[ "$rel" =~ ^(src|scripts|pyproject) ]] && pat="$pat|$CODE_WORK|$PERSON"
  # (AWS's documented example keys, AKIA...EXAMPLE, are public placeholders and are ignored)
  if hits=$(grep -rIoiE --exclude='*.sqlite' --exclude='*.safetensors' --exclude='leakcheck.sh' "$pat" "$p" 2>/dev/null \
            | grep -viE ':AKIA[0-9A-Z]{9}EXAMPLE$' | cut -d: -f1 | sort -u | head -10) && [ -n "$hits" ]; then
    echo "LEAKCHECK: work/secret pattern in:"; echo "$hits"; fail=1
  fi
done
[ "$fail" = 0 ] && echo "LEAKCHECK_OK $*" || { echo "LEAKCHECK_FAIL: nothing sent"; exit 1; }
