#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
GAT_WAITER_PID=$(cat experiments_iclr/logs/gat_pair_launcher.pid)
ps -p "$GAT_WAITER_PID" -o args= | grep -Fq 'bash experiments_iclr/run_gat_failure_pair.sh'
ps -p "$GAT_WAITER_PID" -o stat= | grep -q T
resume_gat() {
  if ps -p "$GAT_WAITER_PID" -o args= 2>/dev/null | grep -Fq 'bash experiments_iclr/run_gat_failure_pair.sh'; then
    kill -CONT "$GAT_WAITER_PID"
    printf 'GAT waiter resumed at %s\n' "$(date -u +%FT%TZ)"
  fi
}
trap resume_gat EXIT
DEADLINE_EPOCH=$(date -u -d '2026-09-26 03:15:00 UTC' +%s)
while true; do
  test "$(date -u +%s)" -lt "$DEADLINE_EPOCH"
  if .venv/bin/python -c 'from experiments_iclr.verify_gat_failure_pair import queue_terminal_marker; queue_terminal_marker()' >/dev/null 2>&1; then
    break
  fi
  .venv/bin/python -c 'from experiments_iclr.verify_gat_failure_pair import queue_processes; assert queue_processes()'
  sleep 30
done
while true; do
  test "$(date -u +%s)" -lt "$DEADLINE_EPOCH"
  GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  if [ -z "$(printf '%s' "$GPU_PIDS" | tr -d '[:space:]')" ]; then
    break
  fi
  sleep 15
done
test ! -e experiments_iclr/ogbn_arxiv_300_results
test "$(sha256sum experiments_iclr/ogbn_arxiv_pilot.py | awk '{print $1}')" = 7abf975fcf900f52a01a51652605b2d2519aaee02918948732ee5401952dd89b
test "$(sha256sum models.py | awk '{print $1}')" = 07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303
printf 'OGB 300 training started at %s\n' "$(date -u +%FT%TZ)"
timeout --signal=TERM --kill-after=30s 30m \
  .venv/bin/python -u experiments_iclr/ogbn_arxiv_pilot.py \
  --device cuda:0 --seeds 0 1 2 --variants base ens gnnm \
  --max-epochs 300 --min-epochs 300 --patience 20 --eval-every 1 \
  --output-root experiments_iclr/ogbn_arxiv_300_results \
  > experiments_iclr/logs/ogbn_arxiv_300.log 2>&1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python \
  experiments_iclr/verify_ogbn_arxiv_results.py \
  --profile fixed-300 --require-complete \
  > experiments_iclr/logs/ogbn_arxiv_300_verify.log 2>&1
touch experiments_iclr/logs/ogbn_arxiv_300.complete
printf 'OGB 300 verified at %s\n' "$(date -u +%FT%TZ)"
