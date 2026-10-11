#!/usr/bin/env bash
set -euo pipefail
gnnm_host=${1:?Use allocation or server77}
gnnm_package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gnnm_phase=$(cd -- "$gnnm_package/.." && pwd)
gnnm_repo=$(git -C "$gnnm_phase" rev-parse --show-toplevel)
gnnm_python=${GNNM_PYTHON:-"$gnnm_repo/.venv/bin/python"}
gnnm_plan=${2:-"$gnnm_package/configs/MASSIVE_PLAN.json"}
gnnm_approval=${3:-"$gnnm_package/configs/APPROVAL.json"}
exec "$gnnm_python" -B "$gnnm_package/launcher.py" \
  --project-root "$gnnm_phase" --plan "$gnnm_plan" --approval "$gnnm_approval" \
  --host "$gnnm_host" --python "$gnnm_python" --detach
