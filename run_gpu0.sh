#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

DEVICE="cuda:0"

for SPLIT in 0 1 2 3 4 5 6 7 8 9; do
    if (( SPLIT % 2 == 0 )); then
        echo "=== GPU0: Base split=$SPLIT ==="
        bash run.sh $SPLIT $DEVICE
    else
        echo "=== GPU0: TABM split=$SPLIT ==="
        bash run2.sh $SPLIT $DEVICE
    fi
done

echo "GPU0 done."
