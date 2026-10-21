#!/bin/bash
# Push Luthor's code and data/mix (public-dataset derived) to the pod. Nothing else leaves the Mac.
set -e
cd "$(dirname "$0")/../.."
E="ssh -i $HOME/.ssh/runpod_luthor -o StrictHostKeyChecking=accept-new -p ${POD_PORT:?set POD_PORT}"
rsync -az --no-owner --no-group -e "$E" --exclude .venv --exclude .git --exclude __pycache__ --exclude 'data/' \
  --exclude models --exclude reports --exclude runs --exclude '.env*' ./ "root@${POD_HOST:?set POD_HOST}:/workspace/luthor/"
rsync -az --no-owner --no-group -e "$E" data/mix "root@$POD_HOST:/workspace/luthor/data/"
