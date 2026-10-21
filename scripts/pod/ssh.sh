#!/bin/bash
# Run a command on the pod:  POD_HOST=1.2.3.4 POD_PORT=12345 scripts/pod/ssh.sh 'nvidia-smi'
exec ssh -i ~/.ssh/runpod_luthor -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20 \
  -p "${POD_PORT:?set POD_PORT}" "root@${POD_HOST:?set POD_HOST}" "$@"
