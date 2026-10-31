#!/usr/bin/env bash
set -euo pipefail
repo=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
cd "$repo"
mkdir -p validation_tuning_prepared/tmp validation_tuning_prepared/cache
export CUDA_VISIBLE_DEVICES=4
export TMPDIR="$repo/validation_tuning_prepared/tmp"
export XDG_CACHE_HOME="$repo/validation_tuning_prepared/cache"
export TORCH_HOME="$repo/validation_tuning_prepared/cache/torch"
.venv/bin/python -u validation_tuning_prepared/tuning.py check-freeze
.venv/bin/python -u validation_tuning_prepared/tuning.py run --dataset cora --device cuda
.venv/bin/python -u validation_tuning_prepared/tuning.py run --dataset wikics --device cuda
