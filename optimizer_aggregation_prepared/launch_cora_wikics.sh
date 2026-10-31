#!/usr/bin/env bash
set -euo pipefail

repo=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
cd "$repo"
study="$repo/optimizer_aggregation_prepared"
mkdir -p "$study/tmp" "$study/cache"
export CUDA_VISIBLE_DEVICES=4
export TMPDIR="$study/tmp"
export XDG_CACHE_HOME="$study/cache"
export TORCH_HOME="$study/cache/torch"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

.venv/bin/python -u "$study/mechanism.py" check-freeze
.venv/bin/python -u "$study/mechanism.py" preflight --device cuda
.venv/bin/python -u "$study/mechanism.py" run --dataset cora --device cuda
.venv/bin/python -u "$study/mechanism.py" run --dataset wikics --device cuda
