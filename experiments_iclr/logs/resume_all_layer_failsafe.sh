#!/usr/bin/env bash
set -euo pipefail
while [ "$(date -u +%s)" -lt "$(date -u -d '2026-09-26 05:15:00 UTC' +%s)" ]; do
  sleep 60
done
cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
if ps -p 61180 -o args= | grep -Fq 'bash experiments_iclr/logs/launch_all_layer_when_ready.sh'; then
  if ps -p 61180 -o stat= | grep -q T; then
    kill -CONT 61180
  fi
fi
