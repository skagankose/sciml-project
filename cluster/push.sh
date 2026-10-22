#!/bin/bash
# Mac -> HPC3: copy the working tree (with .git, so the commit hash travels) via the
# transfer host. Prefer committed code: production jobs refuse a dirty tree.
#   HPC3_USER=<UCInetID> bash cluster/push.sh
# `--exclude` also protects HPC3-side data/outputs/logs from `--delete`.
set -euo pipefail

: "${HPC3_USER:?set HPC3_USER to your UCInetID}"
DEST=${DEST:-/pub/$HPC3_USER/projects/well-rb-four-families}
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain)" ]; then
    echo "warning: uncommitted changes; HPC3 production runs will refuse this tree" >&2
fi
# Files generated on HPC3 (task manifests need the real data; env snapshots) are protected
# from --delete until they have been pulled back and committed here.
PROTECT=()
[ -f tasks/split_manifest.json ] || PROTECT+=(--exclude 'tasks/*.json' --exclude 'tasks/*.jsonl')
[ -f env/hpc3-pip.txt ] || PROTECT+=(--exclude 'env/hpc3-*')
rsync -rlt --delete \
    --exclude data/ --exclude outputs/ --exclude logs/ --exclude .venv/ \
    --exclude __pycache__/ --exclude .pytest_cache/ --exclude .ruff_cache/ --exclude '*.egg-info/' \
    ${PROTECT[@]+"${PROTECT[@]}"} ./ "hpc3-xfer:$DEST/"
echo "pushed $(git rev-parse --short HEAD) to hpc3:$DEST"
