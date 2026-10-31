#!/usr/bin/env bash
set -euo pipefail
while [ "$(date -u +%s)" -lt "$(date -u -d '2026-09-26 03:50:00 UTC' +%s)" ]; do
  sleep 60
done
cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
GAT_WAITER_PID=$(cat experiments_iclr/logs/gat_pair_launcher.pid)
if ps -p "$GAT_WAITER_PID" -o args= | grep -Fq 'bash experiments_iclr/run_gat_failure_pair.sh'; then
  if ps -p "$GAT_WAITER_PID" -o stat= | grep -q T; then
    kill -CONT "$GAT_WAITER_PID"
  fi
fi
