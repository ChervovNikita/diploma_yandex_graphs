# Roman Empire four-mask, two-depth extension: frozen post hoc protocol

This study was designed after observing the favorable fixed-mask width-512
result and other Roman depth and budget experiments. It tests whether the
direction of the tied-versus-untied SAGE difference recurs across four more
published Roman Empire masks under one fixed width-128, 1,000-epoch recipe.
It is exploratory replication on one graph, not four independent datasets.

## Complete grid and analysis

Run all 48 cells: official masks 1, 2, 3, 4; residual depths 2 and 5;
optimization seeds 0, 1, 2; tied GNNM and initially matched untied
propagation. Report all 24 paired test-accuracy differences, including
negative signs, after every cell passes the full audit. Report per-mask and
per-depth descriptive means. Do not select a mask, depth, epoch budget,
or seed from test performance. No p-value treats seeds or masks as
independent graphs.

## Fixed data, model, and selection

Use the corrected public Roman Empire NPZ (`data/roman_empire.npz`) and
each designated published train, validation, and test mask. Symmetrize and
coalesce the raw edges with PyG `to_undirected(coalesce(...))`; require
65,854 directed entries and add no self-loops. Use the original
`models.TABMModel` SAGE, four members, width 128, LayerNorm, dropout 0.2,
AdamW learning rate 0.001 and zero weight decay. Train each arm for exactly
1,000 full-batch epochs with mean member cross-entropy. Validate each epoch
using pooled-logit accuracy; break ties by lower pooled validation
cross-entropy, then earliest epoch. Restore the selected checkpoint before
scoring the published test mask once. No early stopping or tuning.

The untied arm deep-copies the tied residual SAGE stack once per member;
the boundary BatchEnsemble projectors remain the same. Canonical starting
parameters and post-construction Python, NumPy, CPU, and CUDA RNG states
must match within each pair. Initial member logits must agree within
`1e-5` maximum absolute difference. A failed gate invalidates the planned
complete grid until reviewed; it is not replaced silently.
Run the initial-function smoke for every planned mask, depth, and seed on
the assigned CUDA device before any production training.

## Provenance and completion

Pin the runner, verifier, orchestrator, protocol, original `models.py`,
and public NPZ by SHA-256 in each mask/depth source manifest before
training. Keep each mask/depth in a separate result directory. Save the
full 1,000-row validation trace, initial audit and logits, selected
checkpoint, and selected validation and test member logits with exact
node IDs and labels. An independent CUDA verifier must replay all selected
checkpoints and recalculate every score, check initial matching, and reject
missing or extra arms. Only a complete 48-cell audit may be summarized.
The depth-2 and depth-5 workers run sequentially on GPU 0. The complete-grid
audit is written only after both workers pass.
