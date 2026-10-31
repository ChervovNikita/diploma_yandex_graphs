#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
CUTOFF=$(date -u -d '2026-09-26 05:30:00 UTC' +%s)
while true; do
  if .venv/bin/python -c 'import sys; sys.path.insert(0, "experiments_iclr"); from all_layer_be_sage import require_queue_finished_and_gat_idle; require_queue_finished_and_gat_idle()' >/dev/null 2>&1; then
    break
  fi
  if (( $(date -u +%s) >= CUTOFF )); then
    printf 'ALL_LAYER_LAUNCH_SKIPPED_CUTOFF\n'
    exit 0
  fi
  sleep 60
done
printf 'ALL_LAYER_LAUNCH_START %s\n' "$(date -u +%FT%TZ)"
.venv/bin/python -u experiments_iclr/all_layer_be_sage.py
.venv/bin/python experiments_iclr/all_layer_be_sage.py --verify-only --require-complete
printf 'ALL_LAYER_LAUNCH_COMPLETE %s\n' "$(date -u +%FT%TZ)"
