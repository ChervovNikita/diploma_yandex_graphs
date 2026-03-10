#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

DEVICE=${1:-cuda:0}
DATASETS=(amazon-ratings minesweeper tolokers questions)

for DS in "${DATASETS[@]}"; do
    DS_DIR=$(echo "$DS" | tr '-' '_')
    mkdir -p "results/${DS_DIR}/logs"
    mkdir -p "all_checkpoints/${DS_DIR}/base"
    mkdir -p "all_checkpoints/${DS_DIR}/ensemble"
    mkdir -p "all_checkpoints/${DS_DIR}/tabm"
done

for DS in "${DATASETS[@]}"; do
    DS_DIR=$(echo "$DS" | tr '-' '_')
    RESULTS_DIR="results/${DS_DIR}"
    CKPT_DIR="all_checkpoints/${DS_DIR}"

    echo "=== $DS: BASE (all layers) ==="
    L="$RESULTS_DIR/logs/base.log"
    python run_base.py --dataset "$DS" --device "$DEVICE" \
        --save_dir "$CKPT_DIR/base" --log_path "$RESULTS_DIR/base.csv" \
        --layers 1 2 3 4 5 \
        --stdout_log "$L" >> "$L" 2>&1

    echo "=== $DS: ENSEMBLE ==="
    L="$RESULTS_DIR/logs/ensemble.log"
    python run_base_ensemble.py --dataset "$DS" --device "$DEVICE" \
        --save_dir "$CKPT_DIR/ensemble" --log_path "$RESULTS_DIR/ensemble.csv" \
        --stdout_log "$L" >> "$L" 2>&1

    echo "=== $DS: TABM ==="
    L="$RESULTS_DIR/logs/tabm.log"
    python run_tabm.py --dataset "$DS" --device "$DEVICE" \
        --save_dir "$CKPT_DIR/tabm" --log_path "$RESULTS_DIR/tabm.csv" \
        --stdout_log "$L" >> "$L" 2>&1
done

echo "All experiments done."
