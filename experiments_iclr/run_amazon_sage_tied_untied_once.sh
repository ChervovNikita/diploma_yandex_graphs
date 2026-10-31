#!/usr/bin/env bash
# Dormant, one-shot Amazon extension. Explicit --execute is required.
set -euo pipefail
[[ $# -eq 1 && ( $1 == --status || $1 == --execute ) ]] || {
  printf 'Usage: bash experiments_iclr/run_amazon_sage_tied_untied_once.sh --status|--execute\n' >&2
  exit 2
}
MODE=$1
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$HERE/.."
ROOT=experiments_iclr/amazon_sage_tied_untied_results
CUTOFF_UTC='2026-09-26 03:00:00 UTC'
CUTOFF_EPOCH=$(date -u -d "$CUTOFF_UTC" +%s)
fail() { printf 'Amazon SAGE extension refused: %s\n' "$*" >&2; exit 1; }
before_cutoff() { (( $(date -u +%s) < CUTOFF_EPOCH )) || fail "start cutoff $CUTOFF_UTC passed"; }
pristine_root() {
  [[ -d $ROOT && -f $ROOT/source_manifest.json ]] || return 1
  [[ $(find "$ROOT" -mindepth 1 -maxdepth 1 -printf '%f\n') == source_manifest.json ]]
}
core_terminal() {
  local pid event
  [[ -s experiments_iclr/logs/gat_fixed_coordinator.pid && -s experiments_iclr/logs/gat_fixed_coordinator.log ]] || return 1
  pid=$(<experiments_iclr/logs/gat_fixed_coordinator.pid)
  [[ $pid =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "$pid" 2>/dev/null && return 1
  event=$(grep -E '^(Fixed-mask complete gate passed at|Fixed-mask launcher failed or timed out at|Fixed-mask restart skipped after 01:00 UTC cutoff)' experiments_iclr/logs/gat_fixed_coordinator.log | tail -n 1 || true)
  [[ -n $event ]]
}
mc_terminal() {
  [[ -s experiments_iclr/logs/mc_dropout_launcher.log ]] || return 1
  [[ $(tail -n 1 experiments_iclr/logs/mc_dropout_launcher.log) == MC_DROPOUT_LAUNCH_COMPLETE ]]
}
layerwise_terminal() {
  [[ -s experiments_iclr/logs/layerwise_launcher.log ]] || return 1
  tail -n 1 experiments_iclr/logs/layerwise_launcher.log | grep -Eq '^LAYERWISE_LAUNCH_COMPLETE '
}
ogb1000_terminal() {
  [[ -f experiments_iclr/logs/ogbn_arxiv_1000.complete && -f experiments_iclr/ogbn_arxiv_1000_results/summary.json ]]
}
queue_busy() {
  pgrep -af 'launch_all_layer_when_ready[.]sh|all_layer_be_sage[.]py|launch_gat_fixed_after_all_layer[.]sh|run_ogbn_arxiv_untied_once[.]sh|ogbn_arxiv_untied_control[.]py|verify_ogbn_arxiv_untied[.]py|run_gat_failure_pair[.]sh|launch_fixed_mask_when_ready[.]sh|fixed_mask_seed_pair[.]py|verify_fixed_mask_seed_pair[.]py|launch_mc_dropout_when_ready[.]sh|mc_dropout_control[.]py|launch_layerwise_when_ready[.]sh|layerwise_member_diversity[.]py|run_ogbn_arxiv_1000_after_core[.]sh|run_ogbn_arxiv_1000_once[.]sh|ogbn_arxiv_pilot[.]py|verify_ogbn_arxiv_1000[.]py' >/dev/null
}
gpu_idle() {
  command -v nvidia-smi >/dev/null || return 1
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || return 1
  [[ -z ${pids//[[:space:]]/} ]]
}
preflight() {
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE=0 \
    .venv/bin/python experiments_iclr/verify_amazon_sage_tied_untied.py --preflight >/dev/null
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE=0 \
    .venv/bin/python experiments_iclr/amazon_sage_tied_untied.py --preflight-only >/dev/null
}
ready() {
  before_cutoff
  pristine_root || fail 'result root is not pristine'
  preflight || fail 'pinned source or corrected data preflight failed'
  core_terminal || fail 'core coordinator is not terminal'
  mc_terminal || fail 'MC Dropout is not complete'
  layerwise_terminal || fail 'layerwise study is not complete'
  ogb1000_terminal || fail 'optional OGB1000 completion gate is absent'
  ! queue_busy || fail 'queued or optional GPU process is present'
  gpu_idle || fail 'GPU compute process is present or GPU status is unavailable'
  before_cutoff
}
if [[ $MODE == --status ]]; then
  before_cutoff
  if pristine_root; then printf 'PRISTINE_RESULT_ROOT\n'; else printf 'RESULT_ROOT_NOT_PRISTINE\n'; fi
  if preflight; then printf 'SOURCE_PREFLIGHT_PASS\n'; else printf 'SOURCE_PREFLIGHT_FAIL\n'; fi
  if core_terminal; then printf 'CORE_TERMINAL\n'; else printf 'CORE_PENDING\n'; fi
  if mc_terminal; then printf 'MC_COMPLETE\n'; else printf 'MC_PENDING\n'; fi
  if layerwise_terminal; then printf 'LAYERWISE_COMPLETE\n'; else printf 'LAYERWISE_PENDING\n'; fi
  if ogb1000_terminal; then printf 'OGB1000_COMPLETE\n'; else printf 'OGB1000_PENDING\n'; fi
  if queue_busy; then printf 'QUEUED_PROCESS_PRESENT\n'; else printf 'NO_QUEUED_PROCESS\n'; fi
  if gpu_idle; then printf 'GPU_IDLE\n'; else printf 'GPU_BUSY_OR_UNAVAILABLE\n'; fi
  exit 0
fi
command -v flock >/dev/null || fail 'flock is unavailable'
command -v timeout >/dev/null || fail 'timeout is unavailable'
[[ -f experiments_iclr/gat_failure_pair_results/.run.lock ]] || fail 'shared GPU queue lock is missing'
exec 9<experiments_iclr/gat_failure_pair_results/.run.lock
flock -n 9 || fail 'shared GPU queue lock is held'
ready
printf 'AMAZON_TIED_UNTIED_START %s\n' "$(date -u +%FT%TZ)"
PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE=0 timeout --signal=TERM --kill-after=30s 255m \
  .venv/bin/python -u experiments_iclr/amazon_sage_tied_untied.py \
  --device cuda:0 --splits 0 1 2 > "$ROOT/run.log" 2>&1
PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE=0 timeout --signal=TERM --kill-after=30s 30m \
  .venv/bin/python -u experiments_iclr/verify_amazon_sage_tied_untied.py \
  --device cuda:0 > "$ROOT/verification.log" 2>&1
[[ -f $ROOT/completion_audit.json ]] || fail 'independent completion audit is absent'
printf 'AMAZON_TIED_UNTIED_COMPLETE %s\n' "$(date -u +%FT%TZ)"
