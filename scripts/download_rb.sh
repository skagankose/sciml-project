#!/bin/bash
# Download The Well `rayleigh_benard` (NOT rayleigh_benard_uniform) one split at a time with
# the official CLI, then verify each split before marking it complete.
#
# Run on an HPC3 compute node (never the login or transfer host), inside the project env:
#   source cluster/env.sh && bash scripts/download_rb.sh [train valid test]
#
# Resuming: the official tool runs `curl --continue-at -` for every file, and the server
# accepts byte ranges, so rerunning continues partial files. A split counts as complete only
# when `rbbench verify-download` finds every registry file with the exact remote size,
# readable by h5py, and has written its SHA-256 manifest (rbbench_complete_<split>.json).
set -eo pipefail

: "${DATA_ROOT:?set DATA_ROOT (e.g. /pub/\$USER/data/the_well)}"
DATASET=rayleigh_benard
SPLITS=("$@")
[ ${#SPLITS[@]} -eq 0 ] && SPLITS=(train valid test)
MAX_TRIES=${MAX_TRIES:-5}
DEST="$DATA_ROOT/datasets/$DATASET"
mkdir -p "$DEST"

report_disk() {
    echo "--- disk: $(du -sh "$DEST" 2>/dev/null | cut -f1) in $DEST"
    df -h "$DATA_ROOT" | tail -1
}

for split in "${SPLITS[@]}"; do
    case "$split" in train|valid|test) ;; *) echo "unknown split '$split' (use train, valid, test)"; exit 2 ;; esac
    if [ -f "$DEST/rbbench_complete_${split}.json" ]; then
        echo "== $split already verified complete; skipping"
        continue
    fi
    for try in $(seq 1 "$MAX_TRIES"); do
        echo "== $split: download attempt $try/$MAX_TRIES ($(date -Is))"
        the-well-download --base-path "$DATA_ROOT" --dataset "$DATASET" --split "$split" || true
        report_disk
        if python -m rbbench verify-download --data-root "$DATA_ROOT" --split "$split"; then
            echo "== $split verified"
            break
        fi
        if [ "$try" -eq "$MAX_TRIES" ]; then
            echo "== $split still incomplete after $MAX_TRIES attempts; see $DEST/rbbench_verify_${split}.json"
            exit 1
        fi
        sleep 60
    done
done
report_disk
