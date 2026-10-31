# MC Dropout comparator protocol

This is an exploratory inference-only comparator for the Roman Empire SAGE
five-mask study. It is separate from the fixed seven-variant training matrix.
Its results should enter the manuscript only after they have been generated,
audited, and interpreted alongside the trained controls.

## Fixed procedure

1. Read the five `base` rows for official masks 0–4 from
   `experiments_iclr/results/projector_controls.csv`. Each row names the
   checkpoint selected by validation accuracy during the original BASE run.
   The architecture is SAGE, five layers, width 512, learning rate 3e-5,
   with the original dropout probability 0.2. No training or new checkpoint
   selection takes place here.
2. Load the same Roman Empire graph with self loops and the same preprocessing
   as `projector_controls.py`. For each mask, restore the row's checkpoint and
   verify parameter count, validation accuracy, test accuracy, and test loss
   against the archived row. A mismatch stops the run before MC Dropout for
   that mask.
3. Keep the restored model's normalization and all other modules in eval mode.
   Enable its existing dropout modules only. Run **exactly four** full-graph
   stochastic passes per mask. Pass index `j` in `{0,1,2,3}` uses PyTorch seed
   `750000 + 1000 * split + j`. The seed is reset for every pass and the
   surrounding random state is restored afterward.
4. Average the four raw logit arrays elementwise, then take argmax for each
   node. This matches the logit-pooling rule for the GNNM and ENS controls.
   Report the official held-out accuracy for each mask and the paired
   difference from that same BASE checkpoint. The MC validation score is
   diagnostic only and cannot select a checkpoint, pass count, seed, or
   dropout rate.
5. Save per-mask JSON and prediction arrays, followed by a five-mask summary,
   only in `experiments_iclr/mc_dropout_results`. The JSON fingerprints the
   source row, data, checkpoint, code, explicit CUDA device, and Python,
   PyTorch, CUDA, PyG, DGL, NumPy, and scikit-learn versions. An existing
   result is reused only if those inputs and its prediction archive still
   match. An exclusive output lock prevents concurrent runs from mixing files.

This compares four sampled subnetworks of one trained BASE model with the four
members of GNNM and ENS. It isolates whether repeated stochastic inference
from the BASE checkpoint is sufficient to explain an observed improvement.
It does not equalize training objectives, parameter counts, or training cost.

## Execution

The synthetic CPU smoke check has passed. It checks seeded pass equality,
RNG restoration, unchanged weights, runtime fingerprint changes, and the
exclusive output lock; its temporary lock files are removed. Do not run the
full-graph command until the primary GPU training queue has ended, all five
BASE rows and checkpoints are present, and other queued GPU work is idle.
Run from the repository root:

```sh
mkdir -p experiments_iclr/mc_dropout_results
.venv/bin/python -u experiments_iclr/mc_dropout_control.py --device cuda:0 \
  >> experiments_iclr/mc_dropout_results/run.log 2>&1
```

The full-graph program refuses CPU and requires an explicit CUDA device. It
does not write into the matrix CSV, checkpoints, or queue. A completed run
contains `split0.json` through `split4.json`, the five matching
`splitN_predictions.npz` archives, `summary.json`, `run.log`, and
`.run.lock` in `experiments_iclr/mc_dropout_results`. Treat the fixed
four-pass result as one descriptive stochastic draw per official mask.
