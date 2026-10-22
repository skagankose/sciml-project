#!/bin/bash
# HPC3 -> Mac: compact results back (never the HDF5 data; checkpoints only on request).
#   HPC3_USER=<UCInetID> bash cluster/pull.sh            # outputs without *.pt
#   HPC3_USER=<UCInetID> bash cluster/pull.sh --env      # + env snapshots
#   HPC3_USER=<UCInetID> bash cluster/pull.sh --tasks    # + generated task manifests
#   HPC3_USER=<UCInetID> bash cluster/pull.sh --checkpoints
set -euo pipefail

: "${HPC3_USER:?set HPC3_USER to your UCInetID}"
SRC=${SRC:-/pub/$HPC3_USER/projects/well-rb-four-families}
cd "$(dirname "$0")/.."

CKPT=(--exclude '*.pt' --exclude '*.pt.tmp')
for arg in "$@"; do
    case "$arg" in
        --env) rsync -rlt "hpc3-xfer:$SRC/env/" env/ ;;
        --tasks) rsync -rlt --include '*.json' --include '*.jsonl' --exclude '*' "hpc3-xfer:$SRC/tasks/" tasks/ ;;
        --checkpoints) CKPT=() ;;
        *) echo "unknown option $arg" >&2; exit 2 ;;
    esac
done
mkdir -p outputs
rsync -rlt ${CKPT[@]+"${CKPT[@]}"} "hpc3-xfer:$SRC/outputs/" outputs/
echo "pulled outputs from hpc3:$SRC"
