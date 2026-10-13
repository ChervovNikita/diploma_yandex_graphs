#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf 'Usage: bash run_gpu.sh allocation|server77 GPU_INDEX EXACT_APPROVAL_RECEIPT [EXACT_WORKER_PYTHON]\n' >&2
  exit 2
fi
task_host="$1"
task_gpu="$2"
task_approval="$3"
case "$task_host/$task_gpu" in
  allocation/0|server77/0|server77/1) ;;
  *) printf 'Use allocation/0, server77/0 or server77/1.\n' >&2; exit 2 ;;
esac
if [[ "$task_host" == allocation ]]; then
  [[ "$(hostname)" == anogena-2-0 ]] || { printf 'Allocation hostname verification failed.\n' >&2; exit 1; }
  [[ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" == GPU-44039938-fd82-41d2-fefd-de71514e2fac ]] || { printf 'Allocation sole GPU UUID verification failed.\n' >&2; exit 1; }
fi
[[ -f "$task_approval" ]] || { printf 'An existing exact explicit-GO approval receipt is required.\n' >&2; exit 1; }
task_python="${4:-${GNNM_WORKER_PYTHON:-}}"
if [[ -z "$task_python" && "$task_host" == allocation ]]; then
  task_python=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python
fi
[[ -n "$task_python" && -x "$task_python" ]] || { printf 'Supply the exact compatible worker interpreter as argument 4 or GNNM_WORKER_PYTHON.\n' >&2; exit 1; }
task_wrapper_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
task_phase_directory="$(cd -- "$task_wrapper_directory/.." && pwd -P)"
"$task_python" -B - "$task_phase_directory" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
pins={
 'massive_practical_master_20261013_v29/PLAN.json':(44212006,'722c5234cfc50def2b3de334f4326a7df777290eea23e5f530f946b17fce246f'),
 'massive_practical_master_20261013_v29/SUMMARY.json':(68967953,'3c1bb09e9b0a3e4f4c26aa3b7dbc0d76137916a462ab321de5d60768947dde0b'),
 'practical_master_inputs_20261013_v29/INPUT_INDEX.json':(52902,'6d10effd59b1922627b3a7717e38b4fbf11e496fc9959e4a7fea8d7aae5efadc'),
 'master_controller_source_20261013_v29/MANIFEST.json':(12064,'108a9d3797bea32369afd78d063338e079ed1a52ac90252311ee862373b58072'),
}
for name,(size,digest) in pins.items():
 raw=(root/name).read_bytes()
 if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Exact frozen v29 metadata changed: '+name)
plan=json.loads((root/'massive_practical_master_20261013_v29/PLAN.json').read_bytes())
digest=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
if digest!='0a35daf95e7f1ad8e8df1037923a5a38638ed37e8c0a65c9dbcc3bf73f55488b':raise ValueError('Exact v29 canonical plan changed')
PY
exec "$task_python" -B "$task_phase_directory/master_controller_source_20261013_v29/runtime/gpu_command.py" start --project-root "$task_phase_directory" --plan "$task_phase_directory/massive_practical_master_20261013_v29/PLAN.json" --approval "$task_approval" --host "$task_host" --gpu-index "$task_gpu" --python "$task_python"
