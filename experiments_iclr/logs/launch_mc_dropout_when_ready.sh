#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
CUTOFF=$(date -u -d '2026-09-26 09:00:00 UTC' +%s)
while true; do
  if (( $(date -u +%s) >= CUTOFF )); then
    printf 'MC_DROPOUT_LAUNCH_SKIPPED_CUTOFF\n'
    exit 0
  fi
  if ps -p 61180 -o args= 2>/dev/null | grep -Fq 'launch_all_layer_when_ready.sh'; then
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
    printf 'MC_DROPOUT_LAUNCH_SKIPPED_LOCK_CUTOFF\n'
    exit 0
  fi
  sleep 60
done
printf 'MC_DROPOUT_LAUNCH_START %s\n' "$(date -u +%FT%TZ)"
.venv/bin/python -u experiments_iclr/mc_dropout_control.py --device cuda:0 >> experiments_iclr/mc_dropout_results/run.log 2>&1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('experiments_iclr/mc_dropout_results')
assert all((root/f'split{i}.json').is_file() for i in range(5))
assert all((root/f'split{i}_predictions.npz').is_file() for i in range(5))
assert (root/'summary.json').is_file()
print('MC_DROPOUT_LAUNCH_COMPLETE')
PY
