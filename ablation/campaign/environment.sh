#!/usr/bin/env bash
# Source this file after setting GNNM_ENV to the relocated packed environment.
GNNM_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export GNNM_REPO
if [[ -z "${GNNM_ENV:-}" || ! -x "$GNNM_ENV/bin/python" ]]; then
    printf '%s\n' 'Set GNNM_ENV to the existing conda environment directory before sourcing this file.' >&2
    return 1
fi
export PATH="$GNNM_ENV/bin:$PATH"
export PYTHONPATH="$GNNM_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export DGLBACKEND=pytorch
export GNNM_RUNTIME="$GNNM_REPO/.campaign_runtime"
export TMPDIR="$GNNM_RUNTIME/tmp"
export XDG_CACHE_HOME="$GNNM_RUNTIME/cache"
export XDG_CONFIG_HOME="$GNNM_RUNTIME/config"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export DGLDEFAULTDIR="$XDG_CACHE_HOME/dgl"
export DGL_DOWNLOAD_DIR="$DGLDEFAULTDIR"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export OGB_SKIP_UPDATE_CHECK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR="$GNNM_RUNTIME/wandb"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$TORCH_HOME" \
    "$DGLDEFAULTDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$WANDB_DIR"
# Slurm owns CUDA_VISIBLE_DEVICES. Preserve its assigned tokens unchanged.
