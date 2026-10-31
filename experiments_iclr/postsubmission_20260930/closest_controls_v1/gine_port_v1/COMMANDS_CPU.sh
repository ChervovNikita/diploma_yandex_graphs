#!/usr/bin/env bash
set -euo pipefail
# Prepared commands only. They have not been executed by this packet's author.
source /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/protocols/repo_env.sh
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0
GNNM_CHEMISTRY_PYTHON=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python
GNNM_CHEMISTRY_PACKET="${GNNM_PHASE_ROOT}/closest_controls_v1/gine_port_v1"
cd "${GNNM_PHASE_ROOT}"

"${GNNM_CHEMISTRY_PYTHON}" "${GNNM_PHASE_ROOT}/protocols/run_logged.py" \
  --output "${GNNM_CHEMISTRY_PACKET}/logged_runs/source_gate_run01" -- \
  "${GNNM_CHEMISTRY_PYTHON}" "${GNNM_CHEMISTRY_PACKET}/source/verify_source_cpu.py" \
  --report "${GNNM_CHEMISTRY_PACKET}/engineering/SOURCE_CPU_run01.json"

"${GNNM_CHEMISTRY_PYTHON}" "${GNNM_PHASE_ROOT}/protocols/run_logged.py" \
  --output "${GNNM_CHEMISTRY_PACKET}/logged_runs/molhiv_all_layer_preflight_run01" -- \
  "${GNNM_CHEMISTRY_PYTHON}" "${GNNM_CHEMISTRY_PACKET}/source/new_tasks/molhiv_pilot.py" preflight \
  --arm all_layer_be_gine_adaptation_prob --seed 0 --label-fraction 0.1 --device cpu \
  --data-manifest "${GNNM_PHASE_ROOT}/new_tasks/data/molhiv/frozen_shareable_manifest_v2.json" \
  --output "${GNNM_CHEMISTRY_PACKET}/preflights/molhiv_all_layer_label10_run01"

"${GNNM_CHEMISTRY_PYTHON}" "${GNNM_PHASE_ROOT}/protocols/run_logged.py" \
  --output "${GNNM_CHEMISTRY_PACKET}/logged_runs/molhiv_runtime_freeze_run01" -- \
  "${GNNM_CHEMISTRY_PYTHON}" "${GNNM_CHEMISTRY_PACKET}/source/new_tasks/molhiv_pilot.py" freeze \
  --source-gate "${GNNM_CHEMISTRY_PACKET}/engineering/SOURCE_CPU_run01.json" \
  --data-manifest "${GNNM_PHASE_ROOT}/new_tasks/data/molhiv/frozen_shareable_manifest_v2.json" \
  --output "${GNNM_CHEMISTRY_PACKET}/runtime/molhiv_all_layer_amended_run01.json"
