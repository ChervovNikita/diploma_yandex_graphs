#!/usr/bin/env bash
# Dormant, guarded one-shot launcher. Never run concurrently with queued work.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$SCRIPT_DIR/.."

CUTOFF_UTC='2026-09-26 04:30:00 UTC'
CUTOFF_EPOCH=$(date -u -d "$CUTOFF_UTC" +%s)
before_cutoff() {
  if (( $(date -u +%s) >= CUTOFF_EPOCH )); then
    printf 'OGB1000 not started: %s cutoff passed\n' "$CUTOFF_UTC" >&2
    exit 1
  fi
}
before_cutoff

RESULT_ROOT=experiments_iclr/ogbn_arxiv_1000_results
test -f experiments_iclr/logs/ogbn_arxiv_300.complete
test ! -e "$RESULT_ROOT"
test ! -e experiments_iclr/logs/ogbn_arxiv_1000.complete
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python experiments_iclr/verify_ogbn_arxiv_1000.py --preflight
.venv/bin/python -c \
  'from experiments_iclr.verify_gat_failure_pair import queue_terminal_marker; queue_terminal_marker()'

# These queued launchers and jobs have priority. Their absence is required even
# when an earlier study failed or skipped and therefore has no success marker.
for pattern in \
  'launch_all_layer_when_ready.sh' \
  'all_layer_be_sage.py' \
  'launch_gat_fixed_after_all_layer.sh' \
  'run_ogbn_arxiv_untied_once.sh' \
  'ogbn_arxiv_untied_control.py' \
  'verify_ogbn_arxiv_untied.py' \
  'run_gat_failure_pair.sh' \
  'launch_fixed_mask_when_ready.sh' \
  'fixed_mask_seed_pair.py' \
  'verify_fixed_mask_seed_pair.py' \
  'launch_mc_dropout_when_ready.sh' \
  'mc_dropout_control.py' \
  'launch_layerwise_when_ready.sh' \
  'layerwise_member_diversity.py'; do
  if pgrep -af "$pattern" >/dev/null; then
    printf 'OGB1000 not started: queued process still exists: %s\n' "$pattern" >&2
    exit 1
  fi
done

# GAT, MC, and layerwise launchers use this repository-local GPU lock.
exec 9>experiments_iclr/gat_failure_pair_results/.run.lock
flock -n 9 || { printf 'OGB1000 not started: queued GPU lock is held\n' >&2; exit 1; }
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$(printf '%s' "$GPU_PIDS" | tr -d '[:space:]')" || {
  printf 'OGB1000 not started: GPU compute process exists\n' >&2
  exit 1
}
before_cutoff

printf 'OGB1000 training started at %s\n' "$(date -u +%FT%TZ)"
timeout --signal=TERM --kill-after=30s 60m \
  .venv/bin/python -u experiments_iclr/ogbn_arxiv_pilot.py \
  --device cuda:0 --seeds 0 1 2 --variants base ens gnnm \
  --max-epochs 1000 --min-epochs 1000 --patience 20 --eval-every 1 \
  --output-root "$RESULT_ROOT" \
  > experiments_iclr/logs/ogbn_arxiv_1000.log 2>&1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  timeout --signal=TERM --kill-after=30s 30m \
  .venv/bin/python experiments_iclr/verify_ogbn_arxiv_1000.py \
  > experiments_iclr/logs/ogbn_arxiv_1000_verify.log 2>&1
touch experiments_iclr/logs/ogbn_arxiv_1000.complete
printf 'OGB1000 complete artifact gate passed at %s\n' "$(date -u +%FT%TZ)"
