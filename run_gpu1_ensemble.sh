#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

DEVICE="cuda:1"

for SPLIT in 5 6 7 8 9; do
    echo "=== GPU1: Ensemble split=$SPLIT ==="
    bash run_ensemble.sh $SPLIT $DEVICE
done

echo "GPU1 ensemble done."
