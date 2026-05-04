#!/usr/bin/env bash
# k-ablation: TABMAblationModel for k in {2, 8} on the LAYERS x HIDDEN_DIMS x LRS
# grid for every split. Aggregation downstream picks the val-best config per split.
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda:0}"
LOG_DIR="ablation/results_k"
STDOUT_DIR="ablation/logs_k"
mkdir -p "$LOG_DIR" "$STDOUT_DIR"

DATASETS=(roman-empire)
MODELS=(SAGE GAT-sep GT-sep)
KS=(2 4 8)
LAYERS=(5)
HIDDEN_DIMS=(512)
LRS=(3e-5)
NUM_SPLITS=5

for ds in "${DATASETS[@]}"; do
  for m in "${MODELS[@]}"; do
    for k in "${KS[@]}"; do
      tag="${ds}_${m}_k${k}"
      log="$LOG_DIR/${ds}_k${k}.csv"
      stdout="$STDOUT_DIR/${tag}.log"
      echo "==> $tag"
      python -m ablation.run_k_ablation \
        --device "$DEVICE" \
        --dataset "$ds" \
        --model "$m" \
        --k "$k" \
        --layers "${LAYERS[@]}" \
        --hidden_dim "${HIDDEN_DIMS[@]}" \
        --lr "${LRS[@]}" \
        --num_splits "$NUM_SPLITS" \
        --log_path "$log" \
        --stdout_log "$stdout" \
        2>&1 | tee "$stdout"
    done
  done
done

echo "All k-ablation runs finished."
