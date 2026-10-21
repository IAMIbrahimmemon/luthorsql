#!/bin/bash
# Round 6: push code to $REMOTE_ROOT/luthor and each local data path to $REMOTE_D/ on a Vast.ai instance.
#   bash scripts/vast/push6.sh <id> [data paths...]       (dir without trailing / lands at $REMOTE_D/<name>)
# Env: REMOTE_ROOT (default /workspace), REMOTE_D (default /root/d, what the stage scripts read),
#      SQLITE_ONLY=1 sends only *.sqlite under the data paths, CODE_ONLY=1 skips data.
# Vast may bill bandwidth per GB: a dry run sizes the push first; over 20 GB needs CONFIRM=yes.
# Public data only. Code is sent by allowlist and every push is scanned first (leakcheck.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/vast/ssh.sh "${1:?usage: push6.sh <instance_id> [data paths...]}"; shift
REMOTE_ROOT=${REMOTE_ROOT:-/workspace}; REMOTE_D=${REMOTE_D:-/root/d}
# Code goes by allowlist (src, scripts, pyproject.toml only: no docs, reports, git history or .env). That covers
# every new script and module (e.g. scripts/qwrite.py, scripts/build_rare_mix.py, scripts/chains/*.sh, src/luthor/**);
# data goes only by the paths given (data/rare1 is in leakcheck's allowlist for the rare-skeleton inputs).
# Push only what a run needs: e.g. data/sys1 sub-paths with REMOTE_D=/root/d/sys1, never data/sys1/sealed unless a
# stage scores the sealed sets.
CODE=(src scripts pyproject.toml)
CODE_X=(--exclude __pycache__ --exclude .DS_Store --exclude '.env*')
DATA_X=(--exclude .DS_Store --exclude '.env*'); [ -n "${SQLITE_ONLY:-}" ] && DATA_X+=(--include '*/' --include '*.sqlite' --exclude '*')
DATA=(); [ -n "${CODE_ONLY:-}" ] || DATA=("$@")
for p in ${DATA[@]+"${DATA[@]}"}; do [ -e "$p" ] || { echo "missing local path: $p"; exit 1; }; done
bash scripts/vast/leakcheck.sh "${CODE[@]}" ${DATA[@]+"${DATA[@]}"} || exit 1
"${S[@]}" "$H" "mkdir -p $REMOTE_ROOT/luthor $REMOTE_ROOT/logs $REMOTE_D; command -v rsync >/dev/null || \
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync >/dev/null)"
# Dry run: bytes rsync would send (uncompressed upper bound; works with openrsync and GNU rsync).
bytes() { "${RS[@]}" -n --stats "$@" | sed -n 's/^Total transferred file size: *//p' | tr -dc '0-9'; }
TOTAL=$(bytes -R "${CODE_X[@]}" "${CODE[@]}" "$H:$REMOTE_ROOT/luthor/")
for p in ${DATA[@]+"${DATA[@]}"}; do TOTAL=$(( TOTAL + $(bytes "${DATA_X[@]}" "$p" "$H:$REMOTE_D/") )); done
GB=$(python3 -c "print(f'{$TOTAL/1e9:.2f}')")
echo "to send: $TOTAL bytes ($GB GB) to $H"
if [ "$TOTAL" -gt 20000000000 ] && [ "${CONFIRM:-}" != yes ]; then
  echo "over 20 GB: check the host's inet_up_cost/inet_down_cost (offers.sh), then re-run with CONFIRM=yes"; exit 1
fi
"${RS[@]}" -R "${CODE_X[@]}" "${CODE[@]}" "$H:$REMOTE_ROOT/luthor/"
[ -n "${CODE_ONLY:-}" ] && { echo PUSHED6_CODE; exit 0; }
for p in ${DATA[@]+"${DATA[@]}"}; do "${RS[@]}" "${DATA_X[@]}" "$p" "$H:$REMOTE_D/"; done
echo PUSHED6
