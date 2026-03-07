#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

DEVICE="cuda:1"

for SPLIT in 0 1 2 3 4 5 6 7 8 9; do
    echo "=== GPU1: TABM split=$SPLIT ==="
    bash run2.sh $SPLIT $DEVICE
done

echo "GPU1 TABM done."
