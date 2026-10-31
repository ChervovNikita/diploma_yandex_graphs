#!/usr/bin/env bash
set -euo pipefail
# Prepared parent integration command; no remote launch by packet author.
# Future execution only after parent verifies original one-GPU allocation via
# anogena-2.ai0001053-01174. This shell script does not establish route identity.
source /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/protocols/repo_env.sh
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0
GNNM_LOW_RANK_PYTHON=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python
GNNM_LOW_RANK_PACKET="${GNNM_PHASE_ROOT}/closest_controls_v1/trainable_low_rank_gine_v2"
cd "${GNNM_PHASE_ROOT}"
"${GNNM_LOW_RANK_PYTHON}" "${GNNM_PHASE_ROOT}/protocols/run_logged.py" \
  --output "${GNNM_LOW_RANK_PACKET}/logged_runs/synthetic_gate_run01" -- \
  "${GNNM_LOW_RANK_PYTHON}" "${GNNM_LOW_RANK_PACKET}/verify_synthetic_cpu.py" \
  --output "${GNNM_LOW_RANK_PACKET}/engineering/synthetic_gate_run01"
