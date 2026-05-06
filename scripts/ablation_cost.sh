#!/usr/bin/env bash
# Cost analysis: per-(model, num_layers) profile of BASE / ENS_k4 / TABM_k{2,4,8}
# on split 0 (params, mean step time, peak GPU memory).
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda:0}"
LOG_DIR="ablation/results_cost"
STDOUT_DIR="ablation/logs_cost"
mkdir -p "$LOG_DIR" "$STDOUT_DIR"

DATASETS=(minesweeper questions tolokers amazon-ratings roman-empire)
MODELS=(ResNet GCN SAGE GAT GAT-sep GT GT-sep TAG)
LAYERS=(1 2 3 4 5)
HIDDEN_DIMS=(512)
LRS=(3e-5)

if [[ -n "${DATASET:-}" ]]; then
  DATASETS=("$DATASET")
fi

for DS in "${DATASETS[@]}"; do
  stdout="$STDOUT_DIR/${DS}.log"
  log="$LOG_DIR/${DS}.csv"

  echo "=== Cost analysis for dataset $DS ==="
  python -m ablation.run_cost \
    --device "$DEVICE" \
    --dataset "$DS" \
    --models "${MODELS[@]}" \
    --layers "${LAYERS[@]}" \
    --hidden_dim "${HIDDEN_DIMS[@]}" \
    --lr "${LRS[@]}" \
    --log_path "$log" \
    2>&1 | tee "$stdout"

  echo "Cost analysis for $DS finished. Results in $log"
done

echo "All cost analyses finished."
