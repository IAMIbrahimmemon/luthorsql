#!/bin/bash
# Round 6: DESTROY (not stop: a stopped Vast instance keeps billing storage) a Vast.ai instance.
#   CONFIRM=yes bash scripts/vast/destroy.sh <id>
# Copies nothing back: pull results first. Prints the instance's price and cost so far, then destroys.
set -euo pipefail
ID=${1:?usage: CONFIRM=yes destroy.sh <instance_id>}
if [[ "$ID" == *-* ]]; then   # Shadeform (UUID id)
  # Deletion is VERIFIED, not assumed: a DNS blip once made the delete call fail while the chain logged "DELETED".
  # The delete request is retried, then the instance is polled until the API reports it deleting/deleted/gone.
  # Prints VERIFIED_DELETED <id> and exits 0 only then; otherwise exits 1 (check: python3 scripts/shade/sf.py list).
  SF=(python3 "$(dirname "$0")/../shade/sf.py")
  "${SF[@]}" info "$ID" | grep -E '"(status|shade_instance_type|hourly_price|cost_estimate)"' || true
  [ "${CONFIRM:-}" = yes ] || { echo "not destroyed: re-run with CONFIRM=yes"; exit 1; }
  gone() { case "$1" in gone|deleted|deleting) return 0;; *) return 1;; esac; }
  for t in 1 2 3 4 5 6; do
    st=$("${SF[@]}" status "$ID" 2>/dev/null) || st="unreachable"
    gone "$st" && break
    "${SF[@]}" delete "$ID" && break
    echo "delete attempt $t failed (status=$st); retrying in 20 s"; sleep 20
  done
  for t in $(seq 1 ${POLLS:-40}); do   # up to ~10 minutes
    st=$("${SF[@]}" status "$ID" 2>/dev/null) || st="unreachable"
    if gone "$st"; then
      # second opinion: the instance list must not show it as anything but deleting/deleted
      if lst=$("${SF[@]}" list 2>/dev/null); then
        row=$(echo "$lst" | awk -v id="$ID" '$1 == id' || true)
        if [ -z "$row" ] || echo "$row" | grep -qE " (deleting|deleted) "; then
          rm -f "$HOME/.cache/luthor_vast_$ID.env"; echo "VERIFIED_DELETED $ID (status=$st)"; exit 0
        fi
        echo "status says $st but list shows: $row"
      fi
    fi
    echo "$(date -u +%H:%M:%S) waiting for deletion of $ID: status=$st"
    # still active after a while (the delete may have been lost): ask again
    if [ "$st" = active ] && [ $((t % 6)) = 0 ]; then "${SF[@]}" delete "$ID" || true; fi
    sleep 15
  done
  echo "NOT_VERIFIED: $ID may still be running and billing; run: python3 scripts/shade/sf.py list"; exit 1
fi
command -v vastai >/dev/null || { echo "vastai CLI not found (pip install vastai)"; exit 1; }
# Cost so far: a cost field if the API returns one, else dph_total x hours since start_date (estimate).
vastai show instance "$ID" --raw | python3 -c '
import json, sys, time
d = json.load(sys.stdin)
dph, t0 = d.get("dph_total") or 0, d.get("start_date")
h = (time.time() - t0) / 3600 if t0 else None
print("instance %s label=%s status=%s gpu=%s $%.3f/h" % (d.get("id"), d.get("label"), d.get("actual_status"), d.get("gpu_name"), dph))
for k in ("total_cost", "cost", "spent"):
    if d.get(k) is not None: print("  %s: $%s" % (k, d[k]))
if h is not None: print("  up %.2f h -> about $%.2f (estimate; vastai show invoices for billed amounts)" % (h, dph * h))
' || echo "could not read instance $ID (already gone?)"
[ "${CONFIRM:-}" = yes ] || { echo "not destroyed: re-run with CONFIRM=yes"; exit 1; }
vastai destroy instance "$ID" -y </dev/null
sleep 5
if vastai show instances --raw | python3 -c 'import json,sys
d=json.load(sys.stdin); d=d.get("instances",d) if isinstance(d,dict) else d
sys.exit(0 if any(str(i.get("id"))==sys.argv[1] for i in d) else 1)' "$ID"; then
  echo "WARNING: $ID still listed; check vastai show instances"; exit 1
fi
echo "VERIFIED_DELETED $ID"
rm -f "$HOME/.cache/luthor_vast_$ID.env"
