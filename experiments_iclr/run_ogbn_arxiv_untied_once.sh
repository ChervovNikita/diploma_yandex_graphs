#!/usr/bin/env bash
# Dormant one-shot control. Invoke only at a coordinator-approved GPU slot.
set -euo pipefail
cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
test -f experiments_iclr/logs/ogbn_arxiv_300.complete
test ! -e experiments_iclr/ogbn_arxiv_untied_results
test ! -e experiments_iclr/logs/ogbn_arxiv_untied.complete
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python -c 'from experiments_iclr import ogbn_arxiv_untied_control as c; c._check_source_lock(); c._check_reference_config(c.fixed_configuration())'
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$(printf '%s' "$GPU_PIDS" | tr -d '[:space:]')"
printf 'OGB untied control started at %s\n' "$(date -u +%FT%TZ)"
timeout --signal=TERM --kill-after=30s 20m \
  .venv/bin/python -u experiments_iclr/ogbn_arxiv_untied_control.py \
  --device cuda:0 \
  > experiments_iclr/logs/ogbn_arxiv_untied.log 2>&1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=30s 20m \
  .venv/bin/python experiments_iclr/verify_ogbn_arxiv_untied.py \
  > experiments_iclr/logs/ogbn_arxiv_untied_verify.log 2>&1
touch experiments_iclr/logs/ogbn_arxiv_untied.complete
printf 'OGB untied control verified at %s\n' "$(date -u +%FT%TZ)"
