#!/usr/bin/env bash
# Run only after the existing GPU queue has finished with its final marker.
# All generated files remain inside this repository.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
cd "$REPO"

# This deadline is deliberately fixed in source. There is no environment or
# command-line override: a later decision to run requires an explicit edit.
CUTOFF_UTC='2026-09-26 04:00:00 UTC'
CUTOFF_EPOCH=$(date -u -d "$CUTOFF_UTC" +%s)
require_before_cutoff() {
  if (( $(date -u +%s) >= CUTOFF_EPOCH )); then
    echo "GAT pair not started: $CUTOFF_UTC cutoff passed; preserve GPU and manuscript time before the 11:59 UTC submission deadline" >&2
    exit 1
  fi
}
require_before_cutoff

QUEUE_PID_FILE=experiments_iclr/logs/run_after_sage_queue.pid
if [[ ! -f "$QUEUE_PID_FILE" ]]; then
  echo 'Original queue PID file is missing; refusing to run GAT study' >&2
  exit 1
fi
while ! .venv/bin/python -c \
    'from experiments_iclr.verify_gat_failure_pair import queue_terminal_marker; queue_terminal_marker()' \
    >/dev/null 2>&1; do
  require_before_cutoff
  if ! .venv/bin/python -c \
      'from experiments_iclr.verify_gat_failure_pair import queue_processes; raise SystemExit(0 if queue_processes() else 1)' \
      >/dev/null 2>&1; then
    echo 'Original queue exited without its final success marker' >&2
    exit 1
  fi
  sleep 30
done
require_before_cutoff
.venv/bin/python -c \
  'from experiments_iclr.verify_gat_failure_pair import queue_terminal_marker; queue_terminal_marker()'

RESULT_ROOT=experiments_iclr/gat_failure_pair_results
mkdir -p "$RESULT_ROOT"
command -v flock >/dev/null || { echo 'flock is required' >&2; exit 1; }
exec 9>"$RESULT_ROOT/.run.lock"
flock -n 9 || { echo 'Another GAT pair launcher holds the result lock' >&2; exit 1; }

# Freeze source, data, environment, queue completion, and protocol once. A
# restart must match the frozen manifest exactly; completed CSV rows stay put.
.venv/bin/python - <<'PY'
import json
import os
from experiments_iclr.verify_gat_failure_pair import ROOT, expected_manifest

path = ROOT / 'protocol.json'
expected = expected_manifest()
if path.exists():
    if json.loads(path.read_text()) != expected:
        raise SystemExit('Frozen GAT protocol differs from current source or queue')
else:
    temp = ROOT / 'protocol.json.tmp'
    with temp.open('w') as stream:
        json.dump(expected, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
PY

.venv/bin/python experiments_iclr/verify_gat_failure_pair.py
require_before_cutoff

run_group() {
  local depth=$1
  shift
  local splits=("$@")
  .venv/bin/python -c \
    'from experiments_iclr.verify_gat_failure_pair import queue_terminal_marker; queue_terminal_marker()'
  echo "GAT pair: depth=$depth masks=${splits[*]}" | tee -a "$RESULT_ROOT/run.log"
  .venv/bin/python -u experiments_iclr/projector_controls.py \
    --dataset roman-empire --model GAT \
    --variants gnnm untied_backbone --splits "${splits[@]}" \
    --num_layers "$depth" --hidden_dim 512 --lr 3e-5 --m 4 \
    --num_steps 5000 --device cuda:0 --result_root "$RESULT_ROOT" \
    >> "$RESULT_ROOT/run.log" 2>&1
}

run_group 5 0 1
.venv/bin/python experiments_iclr/verify_gat_failure_pair.py --require-splits 0 1
run_group 4 2 3 4
.venv/bin/python experiments_iclr/verify_gat_failure_pair.py --complete
echo 'GAT_PAIR_COMPLETE' | tee -a "$RESULT_ROOT/run.log"
