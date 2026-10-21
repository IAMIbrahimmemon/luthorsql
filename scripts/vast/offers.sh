#!/bin/bash
# Round 6: list the cheapest on-demand 1-GPU Vast.ai offers that meet the operating rules (docs/ROUND6_PLAN.md).
# Free (search only, rents nothing):  bash scripts/vast/offers.sh [H100_SXM|H100_NVL|H200|A100_SXM4|A100_80GB]
# Env: DISK (GB, default 300), N (rows, default 10), NO_DC=1 drops the datacenter filter, EXTRA_Q adds query terms.
# Field names follow `vastai search offers --help`; written without the CLI installed, so check them once:
#   cuda_vers (alias of cuda_max_good), datacenter, -t on-demand, --storage. Raw costs inet_*_cost are $/GB.
set -euo pipefail
command -v vastai >/dev/null || { echo "vastai CLI not found (pip install vastai)"; exit 1; }
GPU=${1:-H100_SXM}; DISK=${DISK:-300}; N=${N:-10}
case "$GPU" in
  A100_80GB) Q="gpu_name in [A100_SXM4,A100_PCIE] gpu_ram>=79" ;;   # 80 GB A100s: both form factors
  *)         Q="gpu_name=$GPU" ;;                                      # Vast writes spaces as _ (H100 SXM -> H100_SXM)
esac
Q="$Q num_gpus=1 verified=true rentable=true reliability>0.98 cuda_vers>=13.0 disk_space>=$DISK inet_down>=500"
[ -n "${NO_DC:-}" ] || Q="$Q datacenter=true"
Q="$Q ${EXTRA_Q:-}"
echo "query: $Q" >&2
PY=$(cat <<'EOF2'
import json, os, sys
rows = json.load(sys.stdin)
rows = rows.get("offers", rows) if isinstance(rows, dict) else rows
rows.sort(key=lambda r: r.get("dph_total") or 1e9)
def f(v, p=2): return "-" if v is None else f"{v:.{p}f}"
def tb(v): return None if v is None else v * 1000                  # $/GB -> $/TB
print("%10s %-10s %6s %6s %6s %5s %6s %6s %7s %7s  %s" % ("id", "gpu", "$/h", "diskGB", "rel", "cuda",
      "downMb", "upMb", "up$/TB", "dn$/TB", "location"))
for r in rows[: int(os.environ["N"])]:
    print("%10s %-10s %6s %6s %6s %5s %6s %6s %7s %7s  %s" % (
        r.get("id"), str(r.get("gpu_name")).replace(" ", "_"), f(r.get("dph_total"), 3), f(r.get("disk_space"), 0),
        f(r.get("reliability2", r.get("reliability")), 4), f(r.get("cuda_max_good"), 1), f(r.get("inet_down"), 0),
        f(r.get("inet_up"), 0), f(tb(r.get("inet_up_cost"))), f(tb(r.get("inet_down_cost"))), r.get("geolocation", "-")))
print(f"{len(rows)} offers match", file=sys.stderr)
EOF2
)
vastai search offers "$Q" -t on-demand --storage "$DISK" -o dph_total --raw | N=$N python3 -c "$PY"
