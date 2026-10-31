#!/usr/bin/env bash
set -euo pipefail

repo=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
cd "$repo"
study="$repo/optimizer_aggregation_prepared"
mkdir -p "$study/tmp" "$study/cache"
export TMPDIR="$study/tmp"
export XDG_CACHE_HOME="$study/cache"
export TORCH_HOME="$study/cache/torch"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
trap 'status=$?; printf "%s\n" "$status" > "$study/initial_update_diagnostic.exit"' EXIT

.venv/bin/python -u "$study/initial_update_diagnostic.py" check-freeze
.venv/bin/python -u "$study/initial_update_diagnostic.py" run
