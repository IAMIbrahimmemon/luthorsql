#!/bin/bash
# Round 6: rent ONE on-demand (not interruptible) Vast.ai instance from an offer id found with offers.sh.
#   CONFIRM=yes bash scripts/vast/create.sh <offer_id>
# Without CONFIRM=yes it only prints the offer's price and exits. Refuses if any instance already exists
# (one GPU at a time). Env: IMAGE, DISK (GB, default 300), LABEL (default luthor-r6), MIN_CUDA (default 13.0; 12.2 for
# the BIRD-environment check, with a CUDA 12.x IMAGE), PKGS (apt packages installed at start).
# No --bid_price is passed, so the CLI rents on-demand. Prints the new instance id last.
set -euo pipefail
command -v vastai >/dev/null || { echo "vastai CLI not found (pip install vastai)"; exit 1; }
OFFER=${1:?usage: CONFIRM=yes create.sh <offer_id>}
IMAGE=${IMAGE:-nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04}   # CUDA 13 devel; torch comes from setup6.sh's venv
DISK=${DISK:-300}; LABEL=${LABEL:-luthor-r6}
# Offer price (with DISK GB of storage), GPU and CUDA, from the same search the CLI uses.
# The search query can't filter by id, and ids depend on --storage, so list offers at this DISK and pick the id.
read -r DPH GPU NG CUDA < <(vastai search offers "rentable=true num_gpus=1${GPU:+ gpu_name=$GPU}" -n -t on-demand --storage "$DISK" --limit 5000 --raw | OFFER=$OFFER python3 -c '
import json, sys
r = json.load(sys.stdin); r = r.get("offers", r) if isinstance(r, dict) else r
import os; o = next((x for x in r if str(x.get("id")) == os.environ["OFFER"]), {})
print(o.get("dph_total", "NONE"), str(o.get("gpu_name", "?")).replace(" ", "_"), o.get("num_gpus", "?"), o.get("cuda_max_good", "?"))') || { echo "offer search failed"; exit 1; }
[ "$DPH" != NONE ] || { echo "offer $OFFER not found or no longer rentable"; exit 1; }
echo "offer $OFFER: $NG x $GPU  cuda $CUDA  \$$DPH/h on-demand (incl. ${DISK} GB disk)  image $IMAGE  label $LABEL"
[ "$NG" = 1 ] || { echo "REFUSED: offer has $NG GPUs, round 6 uses one"; exit 1; }
MIN_CUDA=${MIN_CUDA:-13.0}
python3 -c "import sys; sys.exit(0 if float('$CUDA') >= float('$MIN_CUDA') else 1)" 2>/dev/null || { echo "REFUSED: host CUDA $CUDA < $MIN_CUDA"; exit 1; }
NI=$(vastai show instances --raw | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d.get("instances",d) if isinstance(d,dict) else d; print(len(d))')
# ALLOW_MULTI=N: up to N instances at once (parallel arms the owner approved); default one GPU at a time
[ "$NI" -lt "${ALLOW_MULTI:-1}" ] || { echo "REFUSED: $NI instance(s) already exist (vastai show instances); destroy first or set ALLOW_MULTI"; exit 1; }
[ "${CONFIRM:-}" = yes ] || { echo "not created: re-run with CONFIRM=yes to rent at \$$DPH/h"; exit 1; }
# rsync/python3/curl for push6.sh and setup6.sh; the base CUDA image has none of them.
OUT=$(vastai create instance "$OFFER" --image "$IMAGE" --disk "$DISK" --ssh --direct --label "$LABEL" \
  --onstart-cmd "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ${PKGS:-rsync python3 python3-venv python3.12-dev build-essential curl tmux} >/dev/null 2>&1; mkdir -p /workspace/logs" \
  --raw)
ID=$(printf '%s' "$OUT" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("new_contract","") if d.get("success") else "")')
[ -n "$ID" ] || { echo "CREATE_FAIL: $OUT"; exit 1; }
echo "created instance $ID at \$$DPH/h; next: bash scripts/vast/ssh.sh $ID nvidia-smi (once it is running)"
echo "$ID"
