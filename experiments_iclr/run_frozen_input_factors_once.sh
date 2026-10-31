#!/usr/bin/env bash
# Prepared only. Run manually with --execute; this file never schedules itself.
set -euo pipefail

[[ $# -eq 1 && $1 == --execute ]] || {
  echo 'Usage: experiments_iclr/run_frozen_input_factors_once.sh --execute' >&2
  exit 2
}

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$HERE/.." && pwd -P)
cd "$REPO"
RESULT=experiments_iclr/frozen_input_factor_results
PY=.venv/bin/python
CUTOFF='2026-09-26 03:00:00 UTC'
CUTOFF_EPOCH=$(date -u -d "$CUTOFF" +%s)

before_cutoff() {
  if (( $(date -u +%s) >= CUTOFF_EPOCH )); then
    echo "Frozen-input launch refused: $CUTOFF start cutoff passed" >&2
    exit 1
  fi
}

gpu_idle() {
  command -v nvidia-smi >/dev/null || { echo 'nvidia-smi unavailable' >&2; exit 1; }
  local active
  active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  if [[ -n ${active//[[:space:]]/} ]]; then
    echo 'GPU has a compute process; frozen-input launch refused' >&2
    exit 1
  fi
  if ps -eo args= | grep -E '(^|/)(run_after_sage|launch_[^ ]*_when_ready|run_gat_failure_pair|resume_[^ ]*_failsafe)[.]sh([[:space:]]|$)' | grep -v grep >/dev/null; then
    echo 'A queued GPU launcher is active; frozen-input launch refused' >&2
    exit 1
  fi
}

before_cutoff
command -v flock >/dev/null || { echo 'flock unavailable' >&2; exit 1; }
exec 9>"$RESULT/.launch.lock"
flock -n 9 || { echo 'Another frozen-input launcher holds the lock' >&2; exit 1; }

# Strict one-shot behavior: an interruption needs an explicit audit and reset.
[[ ! -e $RESULT/run_manifest.json && ! -e $RESULT/projector_controls.csv ]] || {
  echo 'Frozen-input run was already started; refusing automatic restart' >&2
  exit 1
}
"$PY" experiments_iclr/verify_frozen_input_factors.py --preflight
gpu_idle
before_cutoff

"$PY" - <<'PY'
import json
import os
from experiments_iclr.verify_frozen_input_factors import ROOT, expected_run_manifest

path = ROOT / 'run_manifest.json'
with path.open('x') as stream:
    json.dump(expected_run_manifest(), stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
PY

for split in 0 1 2 3 4; do
  gpu_idle
  "$PY" experiments_iclr/verify_frozen_input_factors.py --prefix-count "$split"
  "$PY" -u experiments_iclr/frozen_input_factors.py --split "$split" \
    >> "$RESULT/run.log" 2>&1
  "$PY" experiments_iclr/verify_frozen_input_factors.py --prefix-count "$((split + 1))"
done

"$PY" experiments_iclr/verify_frozen_input_factors.py --complete \
  | tee "$RESULT/complete_verification.log"
echo FROZEN_INPUT_FACTORS_COMPLETE | tee -a "$RESULT/run.log"
