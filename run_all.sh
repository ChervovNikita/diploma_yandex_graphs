#!/bin/bash
set -e

export PYTHONUNBUFFERED=1

for SPLIT in 0 1 2 3 4 5 6 7 8 9; do
    if (( SPLIT % 2 == 0 )); then
        BASE_DEV="cuda:0"
        TABM_DEV="cuda:1"
    else
        BASE_DEV="cuda:1"
        TABM_DEV="cuda:0"
    fi

    echo "========== SPLIT $SPLIT / 9  (base=$BASE_DEV, tabm=$TABM_DEV) =========="

    bash run.sh $SPLIT $BASE_DEV &
    PID_BASE=$!

    bash run2.sh $SPLIT $TABM_DEV &
    PID_TABM=$!

    wait $PID_BASE
    echo "--- Base done (split $SPLIT) ---"

    wait $PID_TABM
    echo "--- TABM done (split $SPLIT) ---"

    echo "========== SPLIT $SPLIT done =========="
done

echo "All 10 splits done for both base and TABM."
