#!/usr/bin/env bash
# Optional one-shot waiter. Start manually with --execute; never schedule itself.
set -euo pipefail

[[ $# -eq 1 && ( $1 == --status || $1 == --execute ) ]] || {
  printf 'Usage: bash experiments_iclr/logs/run_ogbn_arxiv_1000_after_core.sh --status|--execute\n' >&2
  exit 2
}
MODE=$1
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$HERE/../.."

CUTOFF_UTC='2026-09-26 04:30:00 UTC'
CUTOFF_EPOCH=$(date -u -d "$CUTOFF_UTC" +%s)
RESULT=experiments_iclr/ogbn_arxiv_1000_results
CORE_PID_FILE=experiments_iclr/logs/gat_fixed_coordinator.pid
CORE_LOG=experiments_iclr/logs/gat_fixed_coordinator.log
MC_LOG=experiments_iclr/logs/mc_dropout_launcher.log
MC_RESULT=experiments_iclr/mc_dropout_results

fail() { printf 'OGB1000 after-core refused: %s\n' "$*" >&2; exit 1; }
before_cutoff() {
  (( $(date -u +%s) < CUTOFF_EPOCH )) || fail "start cutoff $CUTOFF_UTC passed"
}
source_preflight() {
  [[ -f experiments_iclr/logs/ogbn_arxiv_300.complete ]] || fail '300-epoch prerequisite marker is absent'
  [[ ! -e "$RESULT" && ! -e experiments_iclr/logs/ogbn_arxiv_1000.complete ]] || fail '1,000-epoch attempt already exists'
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    .venv/bin/python experiments_iclr/verify_ogbn_arxiv_1000.py --preflight \
    || fail 'frozen source/config lock failed'
}
queue_busy() {
  pgrep -af 'launch_all_layer_when_ready[.]sh|all_layer_be_sage[.]py|launch_gat_fixed_after_all_layer[.]sh|run_ogbn_arxiv_untied_once[.]sh|ogbn_arxiv_untied_control[.]py|verify_ogbn_arxiv_untied[.]py|run_gat_failure_pair[.]sh|launch_fixed_mask_when_ready[.]sh|fixed_mask_seed_pair[.]py|verify_fixed_mask_seed_pair[.]py|launch_mc_dropout_when_ready[.]sh|mc_dropout_control[.]py|launch_layerwise_when_ready[.]sh|layerwise_member_diversity[.]py|run_after_sage_queue[.]sh|resume_all_layer_failsafe[.]sh|resume_gat_failsafe[.]sh' >/dev/null
}
core_finished() {
  [[ -s "$CORE_PID_FILE" && -s "$CORE_LOG" ]] || return 1
  local pid last_event
  pid=$(<"$CORE_PID_FILE")
  [[ $pid =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "$pid" 2>/dev/null && return 1
  last_event=$(grep -E '^(All-layer launcher exited|OGB tied-versus-untied control slot|GAT restart at|Fixed-mask restart at|Fixed-mask complete gate passed at|Fixed-mask launcher failed or timed out at|Fixed-mask restart skipped after 01:00 UTC cutoff)' "$CORE_LOG" | tail -n 1 || true)
  case "$last_event" in
    'Fixed-mask complete gate passed at '*|'Fixed-mask launcher failed or timed out at '*|'Fixed-mask restart skipped after 01:00 UTC cutoff') return 0 ;;
    *) return 1 ;;
  esac
}
mc_finished() {
  [[ -s "$MC_LOG" && -f "$MC_RESULT/summary.json" ]] || return 1
  [[ $(tail -n 1 "$MC_LOG") == 'MC_DROPOUT_LAUNCH_COMPLETE' ]] || return 1
  local split
  for split in 0 1 2 3 4; do
    [[ -f "$MC_RESULT/split${split}.json" && -f "$MC_RESULT/split${split}_predictions.npz" ]] || return 1
  done
}
gpu_idle() {
  command -v nvidia-smi >/dev/null || return 1
  local active
  active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || return 1
  [[ -z ${active//[[:space:]]/} ]]
}
ready() {
  core_finished && mc_finished && ! queue_busy && gpu_idle
}

before_cutoff
source_preflight
if [[ $MODE == --status ]]; then
  if core_finished; then printf 'CORE_TERMINAL_OK\n'; else printf 'CORE_NOT_TERMINAL\n'; fi
  if mc_finished; then printf 'MC_COMPLETE_OK\n'; else printf 'MC_NOT_COMPLETE\n'; fi
  if queue_busy; then printf 'QUEUED_PROCESS_PRESENT\n'; else printf 'NO_QUEUED_PROCESS\n'; fi
  if gpu_idle; then printf 'GPU_IDLE\n'; else printf 'GPU_BUSY_OR_UNAVAILABLE\n'; fi
  if ready; then printf 'READY_FOR_FINAL_GATES\n'; else printf 'WAITING_FOR_AFTER_CORE_SLOT\n'; fi
  exit 0
fi

command -v flock >/dev/null || fail 'flock is unavailable'
exec 8>experiments_iclr/logs/ogbn_arxiv_1000_after_core.lock
flock -n 8 || fail 'another after-core wrapper holds the one-shot lock'
printf 'OGB1000 after-core waiter started at %s\n' "$(date -u +%FT%TZ)"
while ! ready; do
  before_cutoff
  sleep 60
done

# Recheck every condition immediately before handing off. The pinned original
# launcher repeats source, queue, lock, GPU, and cutoff checks itself.
before_cutoff
source_preflight
core_finished || fail 'core coordinator terminal gate changed'
mc_finished || fail 'MC Dropout completion gate changed'
queue_busy && fail 'a queued process appeared'
gpu_idle || fail 'GPU is busy or unavailable'
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -c 'from experiments_iclr.verify_gat_failure_pair import queue_terminal_marker; queue_terminal_marker()' \
  || fail 'original queue lacks its final success marker'
before_cutoff
exec bash experiments_iclr/run_ogbn_arxiv_1000_once.sh
