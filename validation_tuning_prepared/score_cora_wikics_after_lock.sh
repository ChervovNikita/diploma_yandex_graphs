#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
study="$repo/validation_tuning_prepared"
cd "$repo"

test -s "$study/VALIDATION_SELECTION_LOCK.json"
test ! -e "$study/FINAL_SCORE_AUDIT.json"
test ! -e "$study/scores/cora"
test ! -e "$study/scores/wikics"
mkdir -p "$study/tmp" "$study/cache"
export CUDA_VISIBLE_DEVICES=4
export TMPDIR="$study/tmp"
export XDG_CACHE_HOME="$study/cache"
export TORCH_HOME="$study/cache/torch"

"$repo/.venv/bin/python" -u "$study/tuning.py" check-freeze
"$repo/.venv/bin/python" -u "$study/tuning.py" score --dataset cora --device cuda
"$repo/.venv/bin/python" -u "$study/tuning.py" score --dataset wikics --device cuda
