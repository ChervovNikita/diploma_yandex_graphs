#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

DEVICE="cuda:0"

for SPLIT in 0 1 2 3 4; do
    echo "=== GPU0: Ensemble split=$SPLIT ==="
    bash run_ensemble.sh $SPLIT $DEVICE
done

echo "GPU0 ensemble done."
