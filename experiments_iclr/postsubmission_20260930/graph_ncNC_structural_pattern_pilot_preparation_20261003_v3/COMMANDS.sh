#!/usr/bin/env bash
# Prepared command sheet. Each stage requires its separately bound root release.
# Do not run this file as a batch launcher before those receipts/releases exist.
set -euo pipefail
PILOT_RESEARCH_ROOT=/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930
PILOT_EXECUTION_ROOT="$PILOT_RESEARCH_ROOT/graph_ncNC_structural_pattern_execution_root_20261003_v2"
PILOT_NUMERICAL_ROOT="$PILOT_RESEARCH_ROOT/graph_ncNC_structural_pattern_numerical_execution_root_20261003_v2"
PILOT_DRIVER="$PILOT_RESEARCH_ROOT/graph_ncNC_structural_pattern_pilot_preparation_20261003_v3/pattern_run.py"
PILOT_PYTHON=/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/.gnnm_runtime/buddy_extra_v1/site
export CUDA_VISIBLE_DEVICES=GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998

"$PILOT_PYTHON" "$PILOT_DRIVER" --root-release "$PILOT_NUMERICAL_ROOT/ROOT_RELEASE_NUMERICAL.json" --stage numerical --unit pair --base-seed 0 --output "$PILOT_NUMERICAL_ROOT/numerical/run01"
"$PILOT_PYTHON" "$PILOT_DRIVER" --root-release "$PILOT_EXECUTION_ROOT/ROOT_RELEASE_FULL_GRAPH.json" --stage full_graph --unit pair --base-seed 0 --output "$PILOT_EXECUTION_ROOT/full_graph/run01"
"$PILOT_PYTHON" "$PILOT_DRIVER" --root-release "$PILOT_EXECUTION_ROOT/ROOT_RELEASE_FITS.json" --stage fit --unit J --base-seed 0 --output "$PILOT_EXECUTION_ROOT/fit_J/run01"
"$PILOT_PYTHON" "$PILOT_DRIVER" --root-release "$PILOT_EXECUTION_ROOT/ROOT_RELEASE_FITS.json" --stage fit --unit F --base-seed 0 --output "$PILOT_EXECUTION_ROOT/fit_F/run01"
"$PILOT_PYTHON" "$PILOT_DRIVER" --root-release "$PILOT_EXECUTION_ROOT/ROOT_RELEASE_DIAGNOSTICS.json" --stage diagnostics --unit pair --base-seed 0 --output "$PILOT_EXECUTION_ROOT/diagnostics/run01"
"$PILOT_PYTHON" "$PILOT_DRIVER" --root-release "$PILOT_EXECUTION_ROOT/ROOT_RELEASE_CLOSE.json" --stage close --unit pair --base-seed 0 --output "$PILOT_EXECUTION_ROOT/close/run01"
