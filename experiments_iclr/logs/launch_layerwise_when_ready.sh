#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
CUTOFF=$(date -u -d '2026-09-26 09:30:00 UTC' +%s)
while true; do
  if (( $(date -u +%s) >= CUTOFF )); then
    printf 'LAYERWISE_LAUNCH_SKIPPED_CUTOFF\n'
    exit 0
  fi
  if ps -p 69305 -o args= 2>/dev/null | grep -Fq 'launch_mc_dropout_when_ready.sh'; then
    sleep 60
    continue
  fi
  if .venv/bin/python -c 'import sys; sys.path.insert(0, "experiments_iclr"); from all_layer_be_sage import require_queue_finished_and_gat_idle; require_queue_finished_and_gat_idle()' >/dev/null 2>&1; then
    break
  fi
  sleep 60
done
exec 9>experiments_iclr/gat_failure_pair_results/.run.lock
while ! flock -n 9; do
  if (( $(date -u +%s) >= CUTOFF )); then
    printf 'LAYERWISE_LAUNCH_SKIPPED_LOCK_CUTOFF\n'
    exit 0
  fi
  sleep 60
done
printf 'LAYERWISE_LAUNCH_START %s\n' "$(date -u +%FT%TZ)"
.venv/bin/python -u experiments_iclr/layerwise_member_diversity.py --model SAGE --device cuda:0
if .venv/bin/python experiments_iclr/verify_gat_failure_pair.py --complete >/dev/null 2>&1; then
  .venv/bin/python -u experiments_iclr/layerwise_member_diversity.py --model GAT --device cuda:0
else
  printf 'LAYERWISE_GAT_SKIPPED_INCOMPLETE\n'
fi
printf 'LAYERWISE_LAUNCH_COMPLETE %s\n' "$(date -u +%FT%TZ)"
