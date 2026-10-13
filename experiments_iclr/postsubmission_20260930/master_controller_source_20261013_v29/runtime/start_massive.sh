#!/usr/bin/env bash
set -euo pipefail
gnnm_host=${1:?Use allocation or server77}
gnnm_package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gnnm_repo=$(git -C "$gnnm_package" rev-parse --show-toplevel)
gnnm_phase=$gnnm_package
while [[ "$gnnm_phase" != "$gnnm_repo" && ! -f "$gnnm_phase/AGENTS.md" ]]; do
  gnnm_phase=$(dirname -- "$gnnm_phase")
done
[[ -f "$gnnm_phase/AGENTS.md" ]] || { echo "Missing project root marker AGENTS.md" >&2; exit 1; }
gnnm_python=${GNNM_PYTHON:-"$gnnm_repo/.venv/bin/python"}
gnnm_plan=${2:-"$gnnm_package/configs/MASSIVE_PLAN.json"}
gnnm_approval=${3:-"$gnnm_package/configs/APPROVAL.json"}
gnnm_gpu_args=()
if [[ $# -ge 4 ]]; then gnnm_gpu_args=(--gpu-index "$4"); fi
gnnm_launcher=$("$gnnm_python" -B - "$gnnm_phase" "$gnnm_plan" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
plan_path = Path(sys.argv[2]).resolve()
if not plan_path.is_relative_to(root):
    raise ValueError("The plan must stay inside the project")
plan = json.loads(plan_path.read_text())
relative = Path(plan["runtime_launcher"])
if relative.is_absolute() or ".." in relative.parts:
    raise ValueError("Expected a confined frozen runtime launcher")
launcher = (root / relative).resolve()
runtime = (root / plan["runtime_package"]).resolve()
if not runtime.is_relative_to(root) or launcher != runtime / "launcher.py" or not launcher.is_file():
    raise ValueError("The plan's frozen runtime launcher is missing or unconfined")
print(launcher)
PY
)
exec "$gnnm_python" -B "$gnnm_launcher" \
  --project-root "$gnnm_phase" --plan "$gnnm_plan" --approval "$gnnm_approval" \
  --host "$gnnm_host" --python "$gnnm_python" --detach "${gnnm_gpu_args[@]}"
