#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

# source activate /home/avshmelev/.conda/envs/nikita_neurips

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS=(questions tolokers minesweeper amazon-ratings)
MODELS=(ResNet GCN SAGE GAT GAT-sep GT GT-sep TAG)
LAYERS=(1 2 3 4 5)
NUM_SPLITS=10

BASE_PER_DS=$((${#MODELS[@]} * ${#LAYERS[@]} * NUM_SPLITS))
ENSEMBLE_PER_DS=$((${#MODELS[@]} * NUM_SPLITS))
TABM_PER_DS=$((${#MODELS[@]} * NUM_SPLITS))
TASKS_PER_DS=$((BASE_PER_DS + ENSEMBLE_PER_DS + TABM_PER_DS))
TASK_COUNT=$((${#DATASETS[@]} * TASKS_PER_DS))

LOCK_FILE="/tmp/run_experiments_$(whoami)_$$.lock"
COUNTER_FILE="/tmp/run_experiments_counter_$(whoami)_$$.txt"
echo 0 > "$COUNTER_FILE"

for DS in "${DATASETS[@]}"; do
    DS_DIR=$(echo "$DS" | tr '-' '_')
    mkdir -p "$SCRIPT_DIR/results/${DS_DIR}/logs_base"
    mkdir -p "$SCRIPT_DIR/results/${DS_DIR}/logs_ensemble"
    mkdir -p "$SCRIPT_DIR/results/${DS_DIR}/logs_tabm"
    mkdir -p "$SCRIPT_DIR/all_checkpoints/${DS_DIR}/base"
    mkdir -p "$SCRIPT_DIR/all_checkpoints/${DS_DIR}/ensemble"
    mkdir -p "$SCRIPT_DIR/all_checkpoints/${DS_DIR}/tabm"
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

    if [[ $t -lt $BASE_PER_DS ]]; then
        local base_t=$t
        local model_idx=$((base_t / (${#LAYERS[@]} * NUM_SPLITS)))
        local layer_split=$((base_t % (${#LAYERS[@]} * NUM_SPLITS)))
        local layer_idx=$((layer_split / NUM_SPLITS))
        local split=$((layer_split % NUM_SPLITS))
        local MODEL="${MODELS[$model_idx]}"
        local LAYER="${LAYERS[$layer_idx]}"
        local LOG_FILE="$RESULTS_DIR/logs_base/${MODEL}_${LAYER}L_split${split}.log"
        echo "[GPU $gpu] $DS $MODEL ${LAYER}L split=$split"
        python run_base.py --dataset "$DS" --device "$DEVICE" --models "$MODEL" --layers $LAYER --split $split \
            --save_dir "$CKPT_DIR/base" --log_path "$RESULTS_DIR/base.csv" \
            >> "$LOG_FILE" 2>&1
    elif [[ $t -lt $((BASE_PER_DS + ENSEMBLE_PER_DS)) ]]; then
        local ens_t=$((t - BASE_PER_DS))
        local model_idx=$((ens_t / NUM_SPLITS))
        local split=$((ens_t % NUM_SPLITS))
        local MODEL="${MODELS[$model_idx]}"
        local LOG_FILE="$RESULTS_DIR/logs_ensemble/${MODEL}_ens_split${split}.log"
        echo "[GPU $gpu] $DS $MODEL ensemble split=$split"
        python run_base_ensemble.py --dataset "$DS" --device "$DEVICE" --models "$MODEL" --split $split \
            --save_dir "$CKPT_DIR/ensemble" --log_path "$RESULTS_DIR/ensemble.csv" \
            >> "$LOG_FILE" 2>&1
    else
        local tabm_t=$((t - BASE_PER_DS - ENSEMBLE_PER_DS))
        local model_idx=$((tabm_t / NUM_SPLITS))
        local split=$((tabm_t % NUM_SPLITS))
        local MODEL="${MODELS[$model_idx]}"
        local LOG_FILE="$RESULTS_DIR/logs_tabm/${MODEL}_tabm_split${split}.log"
        echo "[GPU $gpu] $DS $MODEL tabm split=$split"
        python run_tabm.py --dataset "$DS" --device "$DEVICE" --models "$MODEL" --split $split \
            --save_dir "$CKPT_DIR/tabm" --log_path "$RESULTS_DIR/tabm.csv" \
            >> "$LOG_FILE" 2>&1
    fi
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
