#!/usr/bin/env bash
# One explicit cell in existing code; no loop, queue, fitting or adoption by default.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Existing six-condition internal BE experiment, one cell only.
Required environment: INTERNAL_BE_ROOT_ADOPTED=1 TASK CONDITION SEED TRAIN VALID OUTPUT
TASK: molhiv | collab | wikics
CONDITION: single | independent4 | O | I | P | G
SEED: caller-frozen integer (proposed 7101, 7203, 7307; not adopted)
TRAIN/VALID: root-bound complete numeric role NPZ paths. OUTPUT: fresh phase path.
O/I/P/G constructor be_init and fixed lambda0.5 are prospective until root freeze.
No TEST reader, scheduler, automatic condition loop, retry or resume.
HELP
  exit 0
fi
if [[ "${INTERNAL_BE_ROOT_ADOPTED:-0}" != "1" ]]; then
  printf '%s\n' 'DISABLED: root must adopt this exact one-cell protocol before execution.' >&2
  exit 64
fi
: "${TASK:?Specify one task}" "${CONDITION:?Specify one condition}" "${SEED:?Specify the root-frozen seed}"
: "${TRAIN:?Specify the bound complete TRAIN role}" "${VALID:?Specify the bound complete development role}" "${OUTPUT:?Specify a fresh phase output}"
repo=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
phase="$repo/experiments_iclr/postsubmission_20260930"
python="$phase/native_ncn_runtime_20261005_v1/.venv/bin/python"
public="$phase/portable_internal_be_public_interface_20261007_v2"
adapter="$phase/public_internal_be_private_steering_adapter_20261007_v1"
interface="$phase/public_internal_be_private_steering_complete_interface_20261007_v1"
allocation="$phase/public_internal_be_allocation_controls_20261007_v1"
native=()
case "$TASK" in
  molhiv) ;;
  collab) native=(--ncn-model "$phase/graph_ncNC_member_completion_qualification_preparation_20261003_v1/private_evidence/native/model.py"
                  --ncn-utils "$phase/graph_ncNC_member_completion_qualification_preparation_20261003_v1/private_evidence/native/utils.py") ;;
  wikics) native=(--polynormer "$phase/wikics_native_polynormer_r_donor_preparation_20261007_v1/vendor/native_polynormer.py") ;;
  *) printf '%s\n' 'Unknown task' >&2; exit 64 ;;
esac
case "$CONDITION" in
  single|independent4) program="$public/train.py"; method=(--arm "$CONDITION") ;;
  O|I|P|G) program="$allocation/train.py"; method=(--policy "$CONDITION" --base-arm be_init --lambda-value 0.5
      --public-root "$public" --adapter-root "$adapter" --interface-root "$interface") ;;
  *) printf '%s\n' 'Unknown condition; this handoff adopts no extra grid' >&2; exit 64 ;;
esac
"$python" -I -S -B - "$phase" "$OUTPUT" <<'PY'
from pathlib import Path
import sys
phase = Path(sys.argv[1]).resolve(strict=True)
output = Path(sys.argv[2]).resolve()
if output == phase or not output.is_relative_to(phase):
    raise SystemExit('Output must resolve inside the project phase')
PY
cd "$repo"
export CUDA_VISIBLE_DEVICES=GPU-44039938-fd82-41d2-fefd-de71514e2fac
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$phase/native_ncn_dependency_overlay_20261005_v1:$repo/.venv/lib/python3.11/site-packages"
exec "$python" -B "$program" --task "$TASK" "${method[@]}" --seed "$SEED" --device cuda:0 \
  --train "$TRAIN" --valid "$VALID" --output "$OUTPUT" "${native[@]}"
