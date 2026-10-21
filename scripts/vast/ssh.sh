#!/bin/bash
# Round 6: ssh into a Vast.ai instance (numeric id) or a Shadeform VM (UUID id).  Run:    bash scripts/vast/ssh.sh <id> 'nvidia-smi'
#                                         Source: source scripts/vast/ssh.sh <id>   (bash only, not zsh)
# Sourcing defines VAST_ID HOST PORT H (root@host), E (ssh command string for rsync -e) and the arrays
# S=(ssh ...) and RS=(rsync ... -e "$E"), so callers use "${S[@]}" "$H" cmd / "${RS[@]}" src "$H:dst".
# Host/port are cached in ~/.cache/luthor_vast_<id>.env (HOST/PORT only); REFRESH=1 re-resolves them.
# Key: VAST_KEY (default ~/.ssh/runpod_luthor; its .pub must be added to the Vast account).
_vast_sourced=0; [[ "${BASH_SOURCE[0]}" != "$0" ]] && _vast_sourced=1
[ $_vast_sourced = 1 ] || set -euo pipefail
VAST_ID=${1:?usage: ssh.sh <instance_id> [cmd...]}
_vast_env=$HOME/.cache/luthor_vast_$VAST_ID.env
if [[ "$VAST_ID" == *-* ]] && { [ -n "${REFRESH:-}" ] || [ ! -s "$_vast_env" ]; }; then   # Shadeform (UUID id)
  read -r HOST PORT _user < <(python3 "$(dirname "${BASH_SOURCE[0]}")/../shade/sf.py" hostline "$VAST_ID" 2>/dev/null || true)
  [ -n "${HOST:-}" ] || { echo "no ssh host yet for $VAST_ID (still starting?)" >&2; return 1 2>/dev/null || exit 1; }
  _k=${VAST_KEY:-$HOME/.ssh/runpod_luthor}
  _o=(-o ForwardAgent=no -o IdentitiesOnly=yes -i "$_k" -p "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
  if [ "$_user" != root ] && ! ssh "${_o[@]}" -o BatchMode=yes "root@$HOST" true 2>/dev/null; then
    # VMs log in as a normal user: allow key-only root login once, so every stage script runs unchanged
    ssh "${_o[@]}" "$_user@$HOST" 'sudo mkdir -p /root/.ssh && sudo cp ~/.ssh/authorized_keys /root/.ssh/authorized_keys &&
      sudo chmod 700 /root/.ssh && sudo chmod 600 /root/.ssh/authorized_keys &&
      sudo sed -i -E "s/^#?PermitRootLogin.*/PermitRootLogin prohibit-password/" /etc/ssh/sshd_config &&
      (sudo systemctl reload ssh 2>/dev/null || sudo systemctl reload sshd)' >/dev/null 2>&1 || true
    ssh "${_o[@]}" -o BatchMode=yes "root@$HOST" true 2>/dev/null || { echo "root login not enabled on $HOST" >&2; return 1 2>/dev/null || exit 1; }
  fi
  mkdir -p "$HOME/.cache"; (umask 077; printf 'HOST=%s\nPORT=%s\n' "$HOST" "$PORT" > "$_vast_env"); unset _user _k _o
fi
if [[ "$VAST_ID" != *-* ]] && { [ -n "${REFRESH:-}" ] || [ ! -s "$_vast_env" ]; }; then   # Vast (numeric id)
  command -v vastai >/dev/null || { echo "vastai CLI not found (pip install vastai)" >&2; return 1 2>/dev/null || exit 1; }
  _u=$(vastai ssh-url "$VAST_ID" 2>/dev/null | grep -Eo 'ssh://[^ ]+' | head -1 || true)   # ssh://root@HOST:PORT
  HOST=${_u#ssh://*@}; PORT=${HOST##*:}; HOST=${HOST%:*}
  if [ -z "$_u" ] || [ -z "$HOST" ] || [ -z "$PORT" ]; then                                 # fallback: raw instance JSON
    read -r HOST PORT < <(vastai show instance "$VAST_ID" --raw | python3 -c 'import json,sys
d=json.load(sys.stdin); print(d.get("ssh_host",""), d.get("ssh_port",""))')
  fi
  [ -n "$HOST" ] && [ -n "$PORT" ] || { echo "no ssh host/port yet for $VAST_ID (still loading?)" >&2; return 1 2>/dev/null || exit 1; }
  mkdir -p "$HOME/.cache"; (umask 077; printf 'HOST=%s\nPORT=%s\n' "$HOST" "$PORT" > "$_vast_env")
fi
HOST=$(sed -n 's/^HOST=//p' "$_vast_env"); PORT=$(sed -n 's/^PORT=//p' "$_vast_env")   # parsed, never sourced
KEY=${VAST_KEY:-$HOME/.ssh/runpod_luthor}
H="root@$HOST"
SSH_OPTS=(-o ForwardAgent=no -o ForwardX11=no -o IdentitiesOnly=yes -o ClearAllForwardings=yes -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20)
S=(ssh "${SSH_OPTS[@]}")
E="ssh -o ForwardAgent=no -o ForwardX11=no -o IdentitiesOnly=yes -o ClearAllForwardings=yes -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20"
RS=(rsync -az --no-owner --no-group -e "$E")
unset _u _vast_env
if [ $_vast_sourced = 0 ]; then shift; exec "${S[@]}" "$H" "$@"; fi
unset _vast_sourced
