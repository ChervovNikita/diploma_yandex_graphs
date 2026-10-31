#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
while ps -p 23261 -o args= 2>/dev/null | grep -Fq 'projector_controls.py --model SAGE'; do
  sleep 30
done
mkdir -p experiments_iclr/logs experiments_iclr/strong_base_results
COMPLETED_ROWS=$(( $(wc -l < experiments_iclr/results/projector_controls.csv) - 1 ))
if [[ "$COMPLETED_ROWS" -ne 35 ]]; then
  echo "Expected 35 complete SAGE control rows, found $COMPLETED_ROWS" >&2
  exit 1
fi
.venv/bin/python - <<'PY'
import csv
import math
from pathlib import Path

path = Path('experiments_iclr/results/projector_controls.csv')
with path.open(newline='') as stream:
    rows = list(csv.DictReader(stream))
expected = {(dataset, 'SAGE', variant, str(split))
            for dataset in ('roman-empire',)
            for variant in ('gnnm', 'independent_projectors', 'heads_only',
                            'input_only', 'output_only', 'base', 'gnnm_m1')
            for split in range(5)}
keys = [(row['dataset'], row['model'], row['variant'], row['split'])
        for row in rows]
if len(keys) != len(expected) or set(keys) != expected:
    raise SystemExit('Control matrix has missing, extra, or duplicate variant/split keys')
root = Path.cwd().resolve()
result_root = (root / 'experiments_iclr/results').resolve()
for row in rows:
    split = int(row['split'])
    variant = row['variant']
    member_count = 1 if variant in ('base', 'gnnm_m1') else 4
    if (int(row['seed']) != split or int(row['num_layers']) != 5
            or int(row['hidden_dim']) != 512
            or not math.isclose(float(row['lr']), 3e-5, rel_tol=0, abs_tol=1e-12)
            or int(row['m']) != member_count
            or int(row['num_steps']) != 5000
            or not 1 <= int(row['best_step']) <= 5000
            or not math.isfinite(float(row['val_metric']))
            or not math.isfinite(float(row['test_metric']))):
        raise SystemExit(f'Mixed or nonfinite protocol in {variant} split {split}')
    for field in ('checkpoint', 'prediction_file'):
        relative = Path(row[field])
        if relative.is_absolute() or '..' in relative.parts:
            raise SystemExit(f'Unsafe {field} in {variant} split {split}')
        artifact = (root / relative).resolve()
        if not artifact.is_relative_to(result_root) or not artifact.is_file():
            raise SystemExit(f'Missing or external {field} in {variant} split {split}')
PY
VERIFICATION_ROOT=experiments_iclr/gnnm_split0_verification
mkdir -p "$VERIFICATION_ROOT"
SOURCE_HASH=$(sha256sum experiments_iclr/projector_controls.py | cut -d ' ' -f 1)
VERIFIER_HASH=$(sha256sum experiments_iclr/verify_gnnm_split0.py | cut -d ' ' -f 1)
MATRIX_HASH=$(sha256sum experiments_iclr/results/projector_controls.csv | cut -d ' ' -f 1)
if [[ -f "$VERIFICATION_ROOT/runner.sha256" ]]; then
  [[ $(cat "$VERIFICATION_ROOT/runner.sha256") == "$SOURCE_HASH" ]] || {
    echo 'Verification source changed; refusing to reuse an existing run' >&2
    exit 1
  }
  [[ -f "$VERIFICATION_ROOT/verifier.sha256" ]] &&
  [[ $(cat "$VERIFICATION_ROOT/verifier.sha256") == "$VERIFIER_HASH" ]] || {
    echo 'Verification decision script changed or manifest is missing' >&2
    exit 1
  }
  if [[ ! -f experiments_iclr/results/gnnm_split0_verification_audit.json ]]; then
    [[ -f "$VERIFICATION_ROOT/matrix.sha256" ]] &&
    [[ $(cat "$VERIFICATION_ROOT/matrix.sha256") == "$MATRIX_HASH" ]] || {
      echo 'Control matrix changed before verification; refusing stale reuse' >&2
      exit 1
    }
  fi
else
  [[ ! -e "$VERIFICATION_ROOT/projector_controls.csv" ]] || {
    echo 'Verification result exists without a source manifest' >&2
    exit 1
  }
  if [[ -f "$VERIFICATION_ROOT/matrix.sha256" ]]; then
    [[ $(cat "$VERIFICATION_ROOT/matrix.sha256") == "$MATRIX_HASH" ]] || {
      echo 'Incomplete manifest belongs to a different matrix' >&2
      exit 1
    }
  fi
  printf '%s\n' "$MATRIX_HASH" > "$VERIFICATION_ROOT/matrix.sha256.tmp"
  mv "$VERIFICATION_ROOT/matrix.sha256.tmp" "$VERIFICATION_ROOT/matrix.sha256"
  printf '%s\n' "$VERIFIER_HASH" > "$VERIFICATION_ROOT/verifier.sha256.tmp"
  mv "$VERIFICATION_ROOT/verifier.sha256.tmp" "$VERIFICATION_ROOT/verifier.sha256"
  printf '%s\n' "$SOURCE_HASH" > "$VERIFICATION_ROOT/runner.sha256.tmp"
  mv "$VERIFICATION_ROOT/runner.sha256.tmp" "$VERIFICATION_ROOT/runner.sha256"
fi
.venv/bin/python -u experiments_iclr/projector_controls.py \
  --model SAGE --variants gnnm --splits 0 \
  --result_root "$VERIFICATION_ROOT" \
  >> experiments_iclr/logs/gnnm_split0_verification.log 2>&1
.venv/bin/python experiments_iclr/verify_gnnm_split0.py \
  >> experiments_iclr/logs/gnnm_split0_verification_audit.log 2>&1
.venv/bin/python experiments_iclr/summarize_controls.py \
  > experiments_iclr/logs/SAGE_summary.log 2>&1
.venv/bin/python experiments_iclr/analyze_decisions.py \
  > experiments_iclr/logs/SAGE_decisions.log 2>&1
.venv/bin/python -u experiments_iclr/profile_strong_base.py \
  > experiments_iclr/logs/strong_base_profile.log 2>&1
WIDTH=$(.venv/bin/python -c 'import json; print(json.load(open("experiments_iclr/strong_base_results/strong_base_selection.json"))["selected_width"])')
.venv/bin/python -u experiments_iclr/projector_controls.py \
  --model SAGE --variants base --hidden_dim "$WIDTH" --splits 0 1 2 3 4 \
  --result_root experiments_iclr/strong_base_results \
  > experiments_iclr/logs/strong_base_training.log 2>&1
.venv/bin/python experiments_iclr/summarize_strong_base.py \
  > experiments_iclr/logs/strong_base_summary.log 2>&1
.venv/bin/python -u experiments_iclr/projector_controls.py \
  --model SAGE --variants untied_backbone --splits 0 1 2 3 4 \
  > experiments_iclr/logs/SAGE_untied_backbone.log 2>&1
.venv/bin/python -u experiments_iclr/projector_controls.py \
  --model SAGE --variants freeze_output_factors --splits 0 1 2 3 4 \
  > experiments_iclr/logs/SAGE_freeze_output_factors.log 2>&1
.venv/bin/python -u experiments_iclr/projector_controls.py \
  --model SAGE --variants ens_pooled --splits 0 1 2 3 4 \
  > experiments_iclr/logs/SAGE_ens_pooled.log 2>&1
.venv/bin/python experiments_iclr/summarize_controls.py \
  > experiments_iclr/logs/SAGE_final_summary.log 2>&1
.venv/bin/python experiments_iclr/analyze_decisions.py \
  > experiments_iclr/logs/SAGE_final_decisions.log 2>&1
{
  printf 'COMMAND .venv/bin/python -u experiments_iclr/ogbn_arxiv_pilot.py --device cuda:0 --seeds 0 --variants base ens gnnm\n'
  .venv/bin/python -u experiments_iclr/ogbn_arxiv_pilot.py \
    --device cuda:0 --seeds 0 --variants base ens gnnm
} > experiments_iclr/logs/ogbn_arxiv_seed0.log 2>&1
.venv/bin/python experiments_iclr/ogbn_arxiv_diagnostics.py \
  > experiments_iclr/logs/ogbn_arxiv_seed0_diagnostics.log 2>&1
printf 'OGB_SEED0_COMPLETE\n'
{
  printf 'COMMAND .venv/bin/python -u experiments_iclr/ogbn_arxiv_pilot.py --device cuda:0 --seeds 1 2 --variants base ens gnnm\n'
  .venv/bin/python -u experiments_iclr/ogbn_arxiv_pilot.py \
    --device cuda:0 --seeds 1 2 --variants base ens gnnm
} > experiments_iclr/logs/ogbn_arxiv_seeds1_2.log 2>&1
.venv/bin/python experiments_iclr/ogbn_arxiv_diagnostics.py \
  > experiments_iclr/logs/ogbn_arxiv_final_diagnostics.log 2>&1
printf 'OGB_FULL_COMPLETE\n'
if [[ $(date -u +%s) -lt $(date -u -d '2026-09-26 06:00:00 UTC' +%s) ]]; then
  .venv/bin/python experiments_iclr/select_parameter_matched_ens.py \
    > experiments_iclr/logs/parameter_matched_ens_selection.log 2>&1
  MATCHED_WIDTH=$(.venv/bin/python -c 'import json; print(json.load(open("experiments_iclr/parameter_matched_ens_results/selection.json"))["selected_width"])')
  .venv/bin/python -u experiments_iclr/projector_controls.py \
    --model SAGE --variants ens_pooled --splits 0 1 2 3 4 \
    --num_layers 5 --hidden_dim "$MATCHED_WIDTH" --lr 3e-5 --m 4 \
    --num_steps 5000 --device cuda:0 \
    --result_root experiments_iclr/parameter_matched_ens_results \
    > experiments_iclr/logs/parameter_matched_ens_training.log 2>&1
  .venv/bin/python experiments_iclr/summarize_parameter_matched_ens.py \
    > experiments_iclr/logs/parameter_matched_ens_summary.log 2>&1
  printf 'PARAM_MATCHED_ENS_COMPLETE\n'
  .venv/bin/python -u experiments_iclr/profile_ensemble_inference.py \
    > experiments_iclr/logs/ensemble_inference_profile.log 2>&1
  printf 'ENSEMBLE_INFERENCE_PROFILE_COMPLETE\n'
else
  printf 'PARAM_MATCHED_ENS_SKIPPED_AFTER_06_UTC\n'
fi
