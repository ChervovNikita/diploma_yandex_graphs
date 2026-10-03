#!/usr/bin/env bash
# Prepared numerical-only command; not an authorization or executed launch.
set -euo pipefail
cd /disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git
PILOT_PHASE=/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930
PILOT_EXECUTION_ROOT="$PILOT_PHASE/graph_ncNC_structural_pattern_numerical_execution_root_20261003_v1"
PILOT_PYTHON=/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12
test -f "$PILOT_EXECUTION_ROOT/ROOT_RELEASE_NUMERICAL.json"
test ! -e "$PILOT_EXECUTION_ROOT/supervision/run01"
test ! -e "$PILOT_EXECUTION_ROOT/numerical/run01"
test ! -e "$PILOT_EXECUTION_ROOT/DETACHED_SUPERVISOR.log"
GNNM_SSH_DESTINATION=shmelev@192.168.18.77 PYTHONDONTWRITEBYTECODE=1 nohup "$PILOT_PYTHON" -B "$PILOT_PHASE/graph_ncNC_structural_pattern_normal_supervision_preparation_20261003_v1/supervise_numerical.py" --root-release "$PILOT_EXECUTION_ROOT/ROOT_RELEASE_NUMERICAL.json" --output "$PILOT_EXECUTION_ROOT/supervision/run01" </dev/null >"$PILOT_EXECUTION_ROOT/DETACHED_SUPERVISOR.log" 2>&1 &
PILOT_SUPERVISOR_PID=$!
printf '%s\n' "$PILOT_SUPERVISOR_PID"
