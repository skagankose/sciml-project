#!/bin/bash
# HPC3 -> Mac: compact results back (never the HDF5 data; checkpoints only on request).
#   HPC3_USER=<UCInetID> bash cluster/pull.sh                  # outputs without *.pt
#   HPC3_USER=<UCInetID> bash cluster/pull.sh --env            # env snapshots only
#   HPC3_USER=<UCInetID> bash cluster/pull.sh --tasks          # generated task manifests only
#   HPC3_USER=<UCInetID> bash cluster/pull.sh --checkpoints    # outputs including *.pt
# Options combine, e.g. `--env --tasks`. Outputs are pulled when no option is given or with
# --checkpoints, so `--env` works before HPC3 has an outputs/ folder.
set -euo pipefail

: "${HPC3_USER:?set HPC3_USER to your UCInetID}"
SRC=${SRC:-/pub/$HPC3_USER/projects/well-rb-four-families}
cd "$(dirname "$0")/.."

CKPT=(--exclude '*.pt' --exclude '*.pt.tmp')
# raw solver rollouts stay on HPC3: at 512 x 128 they are nearly copies of the test data (rule 2); their
# scored tables come back under outputs/baselines/solver_<res>/
SKIP=(--exclude '/solver/')
PULL_OUTPUTS=$(( $# == 0 ))
for arg in "$@"; do
    case "$arg" in
        --env) rsync -rlt "hpc3-xfer:$SRC/env/" env/ && echo "pulled env/ from hpc3:$SRC" ;;
        --tasks) rsync -rlt --include '*.json' --include '*.jsonl' --exclude '*' "hpc3-xfer:$SRC/tasks/" tasks/ \
                     && echo "pulled tasks/ from hpc3:$SRC" ;;
        --checkpoints) CKPT=(); PULL_OUTPUTS=1 ;;
        *) echo "unknown option $arg" >&2; exit 2 ;;
    esac
done
if [ "$PULL_OUTPUTS" -eq 1 ]; then
    mkdir -p outputs
    rsync -rlt "${SKIP[@]}" ${CKPT[@]+"${CKPT[@]}"} "hpc3-xfer:$SRC/outputs/" outputs/
    echo "pulled outputs from hpc3:$SRC"
fi
