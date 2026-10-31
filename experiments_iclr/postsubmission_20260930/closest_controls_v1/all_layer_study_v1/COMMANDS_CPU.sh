#!/usr/bin/env bash
# Prepared commands only; never invoked during source preparation.
# Run from the finalized phase root, under the existing confinement wrapper.
set -euo pipefail
: "${GNNM_AMEND_PYTHON:?Set the approved repository interpreter}"
GNNM_AMEND_DIR=closest_controls_v1/all_layer_study_v1
GNNM_AMEND_PORT_GATE="$GNNM_AMEND_DIR/receipts/CPU_GATE_run02.json"
GNNM_AMEND_INTEGRATION_GATE="$GNNM_AMEND_DIR/engineering/INTEGRATION_CPU_run01.json"

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$GNNM_AMEND_PYTHON" \
  "$GNNM_AMEND_DIR/source/verify_integration_cpu.py" \
  --report "$GNNM_AMEND_INTEGRATION_GATE"

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$GNNM_AMEND_PYTHON" \
  "$GNNM_AMEND_DIR/source/protocols/ppi_study.py" preflight \
  --arm all_layer_be_sage_adaptation_prob --seed 0 --label-fraction 0.1 --device cpu \
  --output "$GNNM_AMEND_DIR/preflights/ppi_all_layer_label10_run01"

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$GNNM_AMEND_PYTHON" \
  "$GNNM_AMEND_DIR/source/new_tasks/mnist_pilot.py" preflight \
  --arm all_layer_be_sage_adaptation_prob --seed 0 --label-fraction 0.1 --device cpu \
  --output "$GNNM_AMEND_DIR/preflights/mnist_all_layer_label10_run01"

for GNNM_AMEND_KIND in sharing architecture; do
  for GNNM_AMEND_LABEL in 10 100; do
    GNNM_AMEND_FRACTION=0.1
    if [[ "$GNNM_AMEND_LABEL" == 100 ]]; then GNNM_AMEND_FRACTION=1.0; fi
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$GNNM_AMEND_PYTHON" \
      "$GNNM_AMEND_DIR/source/protocols/ppi_study.py" freeze \
      --kind "$GNNM_AMEND_KIND" --label-fraction "$GNNM_AMEND_FRACTION" \
      --port-gate "$GNNM_AMEND_PORT_GATE" --integration-gate "$GNNM_AMEND_INTEGRATION_GATE" \
      --output "$GNNM_AMEND_DIR/runtime/ppi_${GNNM_AMEND_KIND}_label${GNNM_AMEND_LABEL}.json"
  done
done

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$GNNM_AMEND_PYTHON" \
  "$GNNM_AMEND_DIR/source/new_tasks/mnist_pilot.py" freeze \
  --port-gate "$GNNM_AMEND_PORT_GATE" --integration-gate "$GNNM_AMEND_INTEGRATION_GATE" \
  --output "$GNNM_AMEND_DIR/runtime/mnist_all_layer_amended.json"
