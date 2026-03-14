#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS=(tolokers)
MODELS=(ResNet GCN SAGE GAT GAT-sep GT GT-sep TAG)
NUM_SPLITS=10

ENSEMBLE_PER_DS=$((${#MODELS[@]} * NUM_SPLITS))
TASKS_PER_DS=$ENSEMBLE_PER_DS
TASK_COUNT=$((${#DATASETS[@]} * TASKS_PER_DS))

LOCK_FILE="/tmp/run_experiments_$(whoami)_$$.lock"
COUNTER_FILE="/tmp/run_experiments_counter_$(whoami)_$$.txt"
echo 0 > "$COUNTER_FILE"

for DS in "${DATASETS[@]}"; do
    DS_DIR=$(echo "$DS" | tr '-' '_')
    mkdir -p "$SCRIPT_DIR/results/${DS_DIR}/logs_ensemble"
    mkdir -p "$SCRIPT_DIR/all_checkpoints/${DS_DIR}/ensemble"
done

get_next_task() {
    (
        flock -x 200
        read idx < "$COUNTER_FILE"
        idx=$((idx))
        if [[ $idx -ge $TASK_COUNT ]]; then
            echo ""
            exit 0
        fi
        echo $((idx + 1)) > "$COUNTER_FILE"
        echo "$idx"
    ) 200>"$LOCK_FILE"
}

run_task() {
    local idx=$1
    local gpu=$2
    local ds_idx=$((idx / TASKS_PER_DS))
    local t=$((idx % TASKS_PER_DS))
    local DS="${DATASETS[$ds_idx]}"
    local DS_DIR=$(echo "$DS" | tr '-' '_')
    local RESULTS_DIR="$SCRIPT_DIR/results/${DS_DIR}"
    local CKPT_DIR="$SCRIPT_DIR/all_checkpoints/${DS_DIR}"
    local DEVICE="cuda:$gpu"

    local model_idx=$((t / NUM_SPLITS))
    local split=$((t % NUM_SPLITS))
    local MODEL="${MODELS[$model_idx]}"
    local LOG_FILE="$RESULTS_DIR/logs_ensemble/${MODEL}_ens_split${split}.log"
    echo "[GPU $gpu] $DS $MODEL ensemble split=$split"
    python run_base_ensemble.py --dataset "$DS" --device "$DEVICE" --models "$MODEL" --split $split \
        --save_dir "$CKPT_DIR/ensemble" --log_path "$RESULTS_DIR/ensemble.csv" \
        --stdout_log "$LOG_FILE" >> "$LOG_FILE" 2>&1
    echo "[GPU $gpu] Done"
}

worker() {
    local gpu=$1
    while true; do
        task=$(get_next_task)
        [[ -z "$task" ]] && break
        run_task "$task" "$gpu" || echo "[GPU $gpu] Task failed, continuing..."
    done
}

worker 0 &
worker 1 &
wait
rm -f "$LOCK_FILE" "$COUNTER_FILE"
echo "All experiments done."
