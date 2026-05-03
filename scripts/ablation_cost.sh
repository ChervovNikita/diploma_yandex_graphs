#!/usr/bin/env bash
# Cost analysis: per-(model, num_layers) profile of BASE / ENS_k4 / TABM_k{2,4,8}
# on split 0 (params, mean step time, peak GPU memory).
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda:0}"
LOG_DIR="ablation/results_cost"
STDOUT_DIR="ablation/logs_cost"
mkdir -p "$LOG_DIR" "$STDOUT_DIR"

DATASET="${DATASET:-roman-empire}"
MODELS=(GCN SAGE GAT-sep GT-sep)
LAYERS=(1 2 3 4 5)
HIDDEN_DIMS=(512)
LRS=(3e-5)

stdout="$STDOUT_DIR/${DATASET}.log"
log="$LOG_DIR/${DATASET}.csv"

python -m ablation.run_cost \
  --device "$DEVICE" \
  --dataset "$DATASET" \
  --models "${MODELS[@]}" \
  --layers "${LAYERS[@]}" \
  --hidden_dim "${HIDDEN_DIMS[@]}" \
  --lr "${LRS[@]}" \
  --log_path "$log" \
  2>&1 | tee "$stdout"

echo "Cost analysis finished. Results in $log"
