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
    python run_base.py --dataset "$DS" --device "$DEVICE" \
        --save_dir "$CKPT_DIR/base" --log_path "$RESULTS_DIR/base.csv" \
        --layers 1 2 3 4 5 \
        >> "$RESULTS_DIR/logs/base.log" 2>&1

    echo "=== $DS: ENSEMBLE ==="
    python run_base_ensemble.py --dataset "$DS" --device "$DEVICE" \
        --save_dir "$CKPT_DIR/ensemble" --log_path "$RESULTS_DIR/ensemble.csv" \
        >> "$RESULTS_DIR/logs/ensemble.log" 2>&1

    echo "=== $DS: TABM ==="
    python run_tabm.py --dataset "$DS" --device "$DEVICE" \
        --save_dir "$CKPT_DIR/tabm" --log_path "$RESULTS_DIR/tabm.csv" \
        >> "$RESULTS_DIR/logs/tabm.log" 2>&1
done

echo "All experiments done."
