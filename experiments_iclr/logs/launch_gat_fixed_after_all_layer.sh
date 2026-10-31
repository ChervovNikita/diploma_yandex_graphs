#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
MC_PID=69305
ALL_LAYER_PID=61180
ps -p "$MC_PID" -o args= | grep -Fq 'bash experiments_iclr/logs/launch_mc_dropout_when_ready.sh'
ps -p "$MC_PID" -o stat= | grep -q T
resume_mc() {
  if ps -p "$MC_PID" -o args= 2>/dev/null | grep -Fq 'bash experiments_iclr/logs/launch_mc_dropout_when_ready.sh'; then
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]; do
      sleep 15
    done
    kill -CONT "$MC_PID"
    printf 'MC waiter resumed at %s\n' "$(date -u +%FT%TZ)"
  fi
}
trap resume_mc EXIT
while ps -p "$ALL_LAYER_PID" -o args= 2>/dev/null | grep -Fq 'bash experiments_iclr/logs/launch_all_layer_when_ready.sh'; do
  sleep 30
done
printf 'All-layer launcher exited at %s\n' "$(date -u +%FT%TZ)"
if tail -n 1 experiments_iclr/logs/all_layer_launcher.log | grep -Fq 'ALL_LAYER_LAUNCH_COMPLETE'; then
  if CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python       experiments_iclr/all_layer_be_sage.py --verify-only --require-complete       > experiments_iclr/logs/all_layer_pre_gat_verify.log 2>&1; then
    printf 'All-layer pre-GAT complete gate passed\n'
    sha256sum experiments_iclr/all_layer_be_sage_results/protocol.json       experiments_iclr/all_layer_be_sage_results/projector_controls.csv       experiments_iclr/all_layer_be_sage_results/artifact_hashes.json       > experiments_iclr/logs/all_layer_pre_gat_artifact_hashes.sha256
  else
    printf 'All-layer pre-GAT complete gate FAILED\n'
  fi
else
  printf 'All-layer launcher did not record completion\n'
fi
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]; do
  sleep 15
done
for attempt in 1 2 3; do
  if CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python -c       'from experiments_iclr.verify_gat_failure_pair import queue_terminal_marker; queue_terminal_marker()'       >> experiments_iclr/logs/gat_fixed_relaunch.log 2>&1; then
    break
  fi
  sleep 15
done
printf 'OGB tied-versus-untied control slot at %s\n' "$(date -u +%FT%TZ)"
if bash experiments_iclr/run_ogbn_arxiv_untied_once.sh \
    >> experiments_iclr/logs/gat_fixed_relaunch.log 2>&1; then
  printf 'OGB tied-versus-untied complete gate passed at %s\n' "$(date -u +%FT%TZ)"
else
  printf 'OGB tied-versus-untied control failed or timed out at %s\n' "$(date -u +%FT%TZ)"
fi
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]; do
  sleep 15
done
printf 'GAT restart at %s\n' "$(date -u +%FT%TZ)"
if timeout --signal=TERM --kill-after=30s 5h     bash experiments_iclr/run_gat_failure_pair.sh     >> experiments_iclr/logs/gat_fixed_relaunch.log 2>&1; then
  printf 'GAT launcher succeeded at %s\n' "$(date -u +%FT%TZ)"
else
  printf 'GAT launcher failed at %s\n' "$(date -u +%FT%TZ)"
fi
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]; do
  sleep 15
done
FIXED_CUTOFF=$(date -u -d '2026-09-26 01:00:00 UTC' +%s)
if [ "$(date -u +%s)" -lt "$FIXED_CUTOFF" ]; then
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python     experiments_iclr/fixed_mask_seed_pair.py --preflight-only     > experiments_iclr/logs/fixed_mask_relaunch_preflight.log 2>&1
  printf 'Fixed-mask restart at %s\n' "$(date -u +%FT%TZ)"
  if timeout --signal=TERM --kill-after=30s 210m       .venv/bin/python -u experiments_iclr/fixed_mask_seed_pair.py       --device cuda:0 --seeds 0 1 2       > experiments_iclr/fixed_mask_seed_pair_results/run.log 2>&1; then
    .venv/bin/python -u experiments_iclr/verify_fixed_mask_seed_pair.py --device cuda:0       > experiments_iclr/fixed_mask_seed_pair_results/verification.log 2>&1
    printf 'Fixed-mask complete gate passed at %s\n' "$(date -u +%FT%TZ)"
  else
    printf 'Fixed-mask launcher failed or timed out at %s\n' "$(date -u +%FT%TZ)"
  fi
else
  printf 'Fixed-mask restart skipped after 01:00 UTC cutoff\n'
fi
