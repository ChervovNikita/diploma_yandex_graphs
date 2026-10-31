#!/usr/bin/env bash
set -euo pipefail
# Pending commands. No remote launch has been performed by this packet author.
source /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/protocols/repo_env.sh
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0
GNNM_PORTABLE_PYTHON=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python
GNNM_PORTABLE_PACKET="${GNNM_PHASE_ROOT}/portable_spectral_replay_v1"
cd "${GNNM_PHASE_ROOT}"

"${GNNM_PORTABLE_PYTHON}" "${GNNM_PHASE_ROOT}/protocols/run_logged.py" \
  --output "${GNNM_PORTABLE_PACKET}/logged_runs/synthetic_gate_run01" -- \
  "${GNNM_PORTABLE_PYTHON}" "${GNNM_PORTABLE_PACKET}/verify_synthetic_cpu.py" \
  --output "${GNNM_PORTABLE_PACKET}/engineering/synthetic_gate_run01"

# Predeclared by seed/known rank; no score-based selection. Export admission
# rejects missing, incomplete, mismatched or unreplayed source runs.
"${GNNM_PORTABLE_PYTHON}" "${GNNM_PHASE_ROOT}/protocols/run_logged.py" \
  --output "${GNNM_PORTABLE_PACKET}/logged_runs/production_export_gate_run01" -- \
  "${GNNM_PORTABLE_PYTHON}" "${GNNM_PORTABLE_PACKET}/export_verified_fit.py" \
  --run "${GNNM_PHASE_ROOT}/spectral_privacy/ppi_training_v1/fits_cpu_v3/seed0_label10_selected_run01" \
  --author-replay "${GNNM_PHASE_ROOT}/spectral_privacy/ppi_training_v1/replays_cpu_v3/seed0_label10_selected_run01.json" \
  --output "${GNNM_PORTABLE_PACKET}/exports/seed0_label10_selected_run01"
