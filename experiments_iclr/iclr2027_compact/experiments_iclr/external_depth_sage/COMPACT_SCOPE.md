# External SAGE depth grid: compact evidence

This is a post hoc, fully enumerated 24-cell check of tied versus untied SAGE
at depths 2 and 5 on WikiCS and Actor, official split 0, three optimizer seeds.
It was chosen after observing the Roman Empire depth result. The saved
`selected_decisions.npz` files contain official validation/test indices and
labels, all four member hard predictions, and the pooled hard prediction.
These suffice to recalculate all reported accuracies and paired differences.
The 300-row validation traces, result records, initial-state hashes, source and
data manifests, and four CUDA replay audit records are also included.

For privacy and archive size, the upload omits the public raw graph bytes,
initial-logit arrays, float32 selected logits, optimizer checkpoints, and
training logs. Thus `verify_compact.py` checks the uploaded decisions and
traces, while `verify_external_depth_grid.py` requires rerunning the models to
recreate omitted checkpoint artifacts. The original full evidence is retained
separately by the authors. `fetch_data.py` downloads public graph files and
checks their hashes against the frozen manifest. Run `external_depth_grid.py`
for each dataset and depth after installing the software described in the
repository; it creates the full artifacts in its own `results/` tree. Existing
compact results must be moved aside before doing that reproduction.
