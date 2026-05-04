#!/usr/bin/env bash
# Numerical parity check between the old (sum-then-backward) and new
# (per-member backward) TABM training step. Run on a free GPU.
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda:1}"
LOG_DIR="ablation/logs_parity"
mkdir -p "$LOG_DIR"

uv run python -m ablation.parity_check \
  --device "$DEVICE" \
  --dataset roman-empire \
  --model SAGE \
  --num_layers 2 \
  --hidden_dim 128 \
  --k 4 \
  --num_steps 200 \
  --split 0 \
  2>&1 | tee "$LOG_DIR/parity.log"
