#!/usr/bin/env bash
# Prepared full-graph-only command; not an authorization or executed launch.
set -euo pipefail
cd /disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git
PILOT_PHASE=/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930
PILOT_FULL_GRAPH_EXECUTION_ROOT="$PILOT_PHASE/graph_ncNC_structural_pattern_full_graph_execution_root_20261003_v2"
PILOT_PYTHON=/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12
test -f "$PILOT_FULL_GRAPH_EXECUTION_ROOT/ROOT_RELEASE_FULL_GRAPH.json"
test ! -e "$PILOT_FULL_GRAPH_EXECUTION_ROOT/supervision/run01"
test ! -e "$PILOT_FULL_GRAPH_EXECUTION_ROOT/full_graph/run01"
test ! -e "$PILOT_FULL_GRAPH_EXECUTION_ROOT/DETACHED_SUPERVISOR.log"
GNNM_SSH_DESTINATION=shmelev@192.168.18.77 PYTHONDONTWRITEBYTECODE=1 nohup "$PILOT_PYTHON" -B "$PILOT_PHASE/graph_ncNC_structural_pattern_full_graph_normal_supervision_preparation_20261003_v2/supervise_full_graph.py" --root-release "$PILOT_FULL_GRAPH_EXECUTION_ROOT/ROOT_RELEASE_FULL_GRAPH.json" --output "$PILOT_FULL_GRAPH_EXECUTION_ROOT/supervision/run01" </dev/null >"$PILOT_FULL_GRAPH_EXECUTION_ROOT/DETACHED_SUPERVISOR.log" 2>&1 &
PILOT_SUPERVISOR_PID=$!
printf '%s\n' "$PILOT_SUPERVISOR_PID"
