# Roman Empire no-explicit-self-loop depth check: compact evidence

This is a post hoc 12-cell check on Roman Empire's official mask 0 and three
optimizer seeds. It compares tied and untied propagation at 2 and 5 SAGE
layers under the same edge convention as the external WikiCS and Actor grid:
undirected edges, with no explicitly added self-loops. All depth-5 selected
epochs are near the 300-epoch ceiling, so the check does not establish
convergence.

The saved `selected_decisions.npz` files hold validation/test indices and
labels, four member hard predictions and the pooled hard prediction. They
permit independent accuracy and paired-difference calculations. Source/data
hashes, all 300-row validation traces, initialization/result records, and
both full CUDA replay audit summaries are included. `verify_compact.py`
checks the uploaded decisions and traces.

This compact upload omits the public Roman NPZ bytes, initial-logit arrays,
float32 selected logits, optimizer checkpoints and training logs. Therefore
`verify_roman_noloop_depth.py` requires a fresh run to recreate the omitted
artifacts. Place the publicly available `data/roman_empire.npz` under this
study folder and check its SHA-256 against `PRETRAIN_FREEZE.json` before using
`roman_noloop_depth.py`. Existing compact results must be moved aside before
reproduction. The original full evidence is retained separately by the
authors.
