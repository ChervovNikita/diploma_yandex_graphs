#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
ALL_LAYER_WAITER_PID=61180
ps -p "$ALL_LAYER_WAITER_PID" -o args= | grep -Fq 'bash experiments_iclr/logs/launch_all_layer_when_ready.sh'
ps -p "$ALL_LAYER_WAITER_PID" -o stat= | grep -q T
resume_all_layer() {
  if ps -p "$ALL_LAYER_WAITER_PID" -o args= 2>/dev/null | grep -Fq 'bash experiments_iclr/logs/launch_all_layer_when_ready.sh'; then
    kill -CONT "$ALL_LAYER_WAITER_PID"
    printf 'All-layer waiter resumed at %s\n' "$(date -u +%FT%TZ)"
  fi
}
trap resume_all_layer EXIT
DEADLINE_EPOCH=$(date -u -d '2026-09-26 01:00:00 UTC' +%s)
while true; do
  test "$(date -u +%s)" -lt "$DEADLINE_EPOCH"
  if ! ps -p 43238 -o args= 2>/dev/null | grep -Fq 'bash experiments_iclr/run_gat_failure_pair.sh'; then
    break
  fi
  sleep 30
done
.venv/bin/python -c 'from experiments_iclr.verify_gat_failure_pair import queue_terminal_marker; queue_terminal_marker()'
while true; do
  test "$(date -u +%s)" -lt "$DEADLINE_EPOCH"
  GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  if [ -z "$(printf '%s' "$GPU_PIDS" | tr -d '[:space:]')" ]; then
    break
  fi
  sleep 15
done
.venv/bin/python experiments_iclr/fixed_mask_seed_pair.py --preflight-only
printf 'Fixed-mask seed study started at %s\n' "$(date -u +%FT%TZ)"
timeout --signal=TERM --kill-after=30s 3h30m \
  .venv/bin/python -u experiments_iclr/fixed_mask_seed_pair.py \
  --device cuda:0 --seeds 0 1 2 \
  > experiments_iclr/fixed_mask_seed_pair_results/run.log 2>&1
.venv/bin/python -u experiments_iclr/verify_fixed_mask_seed_pair.py --device cuda:0 \
  > experiments_iclr/fixed_mask_seed_pair_results/verification.log 2>&1
printf 'Fixed-mask seed study verified at %s\n' "$(date -u +%FT%TZ)"
