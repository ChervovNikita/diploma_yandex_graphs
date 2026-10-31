#!/usr/bin/env bash
# CPU-only coordinator preparation. No CUDA context or training is requested.
set -euo pipefail
: "${PPI_PYTHON:?Use the existing authorized repository interpreter}"
PPI_PHASE=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930
source "$PPI_PHASE/protocols/repo_env.sh"
export CUDA_VISIBLE_DEVICES=''

"$PPI_PYTHON" "$PPI_PHASE/protocols/ppi_development_v3/prepare.py" check

"$PPI_PYTHON" "$PPI_PHASE/protocols/run_logged.py" --output "$PPI_PHASE/protocols/logged_runs/ppi_development_v3_freeze_sharing_label10" -- "$PPI_PYTHON" "$PPI_PHASE/protocols/ppi_study.py" freeze --kind sharing --label-fraction 0.1 --data-manifest "$PPI_PHASE/new_tasks/data/ppi/frozen_shareable_manifest_v2.json" --output "$PPI_PHASE/protocols/ppi_development_v3_runtime_v1/raw_sharing_label10.json"

"$PPI_PYTHON" "$PPI_PHASE/protocols/run_logged.py" --output "$PPI_PHASE/protocols/logged_runs/ppi_development_v3_freeze_architecture_label10" -- "$PPI_PYTHON" "$PPI_PHASE/protocols/ppi_study.py" freeze --kind architecture --label-fraction 0.1 --data-manifest "$PPI_PHASE/new_tasks/data/ppi/frozen_shareable_manifest_v2.json" --output "$PPI_PHASE/protocols/ppi_development_v3_runtime_v1/raw_architecture_label10.json"

"$PPI_PYTHON" "$PPI_PHASE/protocols/run_logged.py" --output "$PPI_PHASE/protocols/logged_runs/ppi_development_v3_freeze_sharing_label100" -- "$PPI_PYTHON" "$PPI_PHASE/protocols/ppi_study.py" freeze --kind sharing --label-fraction 1.0 --data-manifest "$PPI_PHASE/new_tasks/data/ppi/frozen_shareable_manifest_v2.json" --output "$PPI_PHASE/protocols/ppi_development_v3_runtime_v1/raw_sharing_label100.json"

"$PPI_PYTHON" "$PPI_PHASE/protocols/run_logged.py" --output "$PPI_PHASE/protocols/logged_runs/ppi_development_v3_freeze_architecture_label100" -- "$PPI_PYTHON" "$PPI_PHASE/protocols/ppi_study.py" freeze --kind architecture --label-fraction 1.0 --data-manifest "$PPI_PHASE/new_tasks/data/ppi/frozen_shareable_manifest_v2.json" --output "$PPI_PHASE/protocols/ppi_development_v3_runtime_v1/raw_architecture_label100.json"

"$PPI_PYTHON" "$PPI_PHASE/protocols/run_logged.py" --output "$PPI_PHASE/protocols/logged_runs/ppi_development_v3_selector_path_gate" -- "$PPI_PYTHON" "$PPI_PHASE/protocols/verify_ppi_repoenv_selector_v3.py" --output "$PPI_PHASE/protocols/ppi_development_v3_runtime_v1/SELECTOR_PATH_GATE.json"

# Preserve these raw freezes. Capture the actual authorized idle GPU and
# finalize the supplemental joint protocols only when that hardware is ready.
