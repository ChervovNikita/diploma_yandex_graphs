#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

# source activate /home/avshmelev/.conda/envs/nikita_neurips

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS=(tolokers)
# DATASETS=(minesweeper questions tolokers amazon-ratings roman-empire)
MODELS=(ResNet GCN SAGE GAT GAT-sep GT GT-sep TAG)
LAYERS=(1 2 3 4 5)
# Full grid inside each Python run: layers (per task) × hidden_dim × lr
HIDDEN_DIMS=(256 384 512)
LRS=(3e-5 2e-5 4e-5 1e-5)
NUM_SPLITS=10
MAX_TASKS=1

BASE_PER_DS=$((${#MODELS[@]} * ${#LAYERS[@]} * NUM_SPLITS))
ENSEMBLE_PER_DS=$((${#MODELS[@]} * NUM_SPLITS))
TABM_PER_DS=$((${#MODELS[@]} * NUM_SPLITS))
TASKS_PER_DS=$((BASE_PER_DS + ENSEMBLE_PER_DS + TABM_PER_DS))

LOCK_FILE="/tmp/run_experiments_$(whoami)_$$.lock"
COUNTER_FILE="/tmp/run_experiments_counter_$(whoami)_$$.txt"

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
        if [[ $idx -ge $PHASE_END ]]; then
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
            --hidden_dim "${HIDDEN_DIMS[@]}" --lr "${LRS[@]}" \
            --save_dir "$CKPT_DIR/base" --log_path "$RESULTS_DIR/base.csv" \
            --stdout_log "$LOG_FILE" >> "$LOG_FILE" 2>&1
    elif [[ $t -lt $((BASE_PER_DS + ENSEMBLE_PER_DS)) ]]; then
        local ens_t=$((t - BASE_PER_DS))
        local model_idx=$((ens_t / NUM_SPLITS))
        local split=$((ens_t % NUM_SPLITS))
        local MODEL="${MODELS[$model_idx]}"
        local LOG_FILE="$RESULTS_DIR/logs_ensemble/${MODEL}_ens_split${split}.log"
        echo "[GPU $gpu] $DS $MODEL ensemble split=$split"
        python run_base_ensemble.py --dataset "$DS" --device "$DEVICE" --models "$MODEL" --split $split \
            --layers "${LAYERS[@]}" \
            --hidden_dim "${HIDDEN_DIMS[@]}" --lr "${LRS[@]}" \
            --save_dir "$CKPT_DIR/ensemble" --log_path "$RESULTS_DIR/ensemble.csv" \
            --stdout_log "$LOG_FILE" >> "$LOG_FILE" 2>&1
    else
        local tabm_t=$((t - BASE_PER_DS - ENSEMBLE_PER_DS))
        local model_idx=$((tabm_t / NUM_SPLITS))
        local split=$((tabm_t % NUM_SPLITS))
        local MODEL="${MODELS[$model_idx]}"
        local LOG_FILE="$RESULTS_DIR/logs_tabm/${MODEL}_tabm_split${split}.log"
        echo "[GPU $gpu] $DS $MODEL tabm split=$split"
        python run_tabm.py --dataset "$DS" --device "$DEVICE" --models "$MODEL" --split $split \
            --layers "${LAYERS[@]}" \
            --hidden_dim "${HIDDEN_DIMS[@]}" --lr "${LRS[@]}" \
            --save_dir "$CKPT_DIR/tabm" --log_path "$RESULTS_DIR/tabm.csv" \
            --stdout_log "$LOG_FILE" >> "$LOG_FILE" 2>&1
    fi
    echo "[GPU $gpu] Done"
}

gpu_pool() {
    local gpu=$1
    local -a running=()
    local no_more_tasks=false

    while true; do
        while [[ $no_more_tasks == false && ${#running[@]} -lt $MAX_TASKS ]]; do
            task=$(get_next_task)
            if [[ -z "$task" ]]; then
                no_more_tasks=true
                break
            fi
            ( run_task "$task" "$gpu" || echo "[GPU $gpu] Task failed, continuing..." ) &
            running+=($!)
        done

        if [[ ${#running[@]} -eq 0 ]]; then
            break
        fi

        wait -n
        local -a still=()
        for pid in "${running[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                still+=("$pid")
            fi
        done
        running=("${still[@]}")
    done
}

for ds_idx in "${!DATASETS[@]}"; do
    PHASE_START=$((ds_idx * TASKS_PER_DS))
    PHASE_END=$(((ds_idx + 1) * TASKS_PER_DS))
    echo "$PHASE_START" > "$COUNTER_FILE"
    echo "=== Dataset ${DATASETS[$ds_idx]} (indices $PHASE_START..$((PHASE_END - 1))) ==="
    gpu_pool 0 &
    gpu_pool 1 &
    wait
done
rm -f "$LOCK_FILE" "$COUNTER_FILE"
echo "All experiments done."
