#!/usr/bin/env bash
# Init-scheme ablation: TABM (k=4) with R~Xavier on the output BE projector,
# trained on the LAYERS x HIDDEN_DIMS x LRS grid for every split. Compared in
# aggregation against the main-table TABM (k=4, default scheme).
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda:0}"
LOG_DIR="ablation/results_init"
STDOUT_DIR="ablation/logs_init"
mkdir -p "$LOG_DIR" "$STDOUT_DIR"

DATASETS=(roman-empire)
MODELS=(SAGE GAT-sep GT-sep)
SCHEMES=(default xavier_all ones_all)
LAYERS=(5)
HIDDEN_DIMS=(512)
LRS=(3e-5)
NUM_SPLITS=5

for ds in "${DATASETS[@]}"; do
  for m in "${MODELS[@]}"; do
    for scheme in "${SCHEMES[@]}"; do
      tag="${ds}_${m}_${scheme}"
      log="$LOG_DIR/${ds}_${scheme}.csv"
      stdout="$STDOUT_DIR/${tag}.log"
      echo "==> $tag"
      python -m ablation.run_init_ablation \
        --device "$DEVICE" \
        --dataset "$ds" \
        --model "$m" \
        --init_scheme "$scheme" \
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

echo "All init-ablation runs finished."
