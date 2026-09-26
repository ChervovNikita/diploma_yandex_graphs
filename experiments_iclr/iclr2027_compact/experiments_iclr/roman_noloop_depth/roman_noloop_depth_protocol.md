# Roman Empire no-explicit-self-loop depth check, frozen protocol v1

This is a post hoc matched-convention extension. Roman Empire depth 2 under
this graph convention and Roman Empire depths 2/5 under a different explicit
self-loop convention were already known. The depth-5 result under this
no-explicit-self-loop convention is unobserved at freeze time. All cells are
reported regardless of sign. This study does not retroactively create a
prospective test or make the three optimizer seeds independent graph splits.

Use the original `data/roman_empire.npz` public bytes, official mask 0, and
optimizer seeds 0, 1, 2. Symmetrize stored edges with PyG `to_undirected`
without adding explicit self-loops. Use transductive node features and graph
structure, but only train labels in the objective. The exactly fixed grid is
2 depths (2, 5) x 2 arms (tied, untied residual propagation) x 3 seeds =
12 runs. The data loader, architecture, optimizer, objective, and checkpoint
rule are copied from the prior Roman bridge and from the WikiCS/Actor SAGE
study; only depth, arm set, protocol ID, and isolated result path differ.

Every run uses original `models.TABMModel` SAGE, four members, width 128,
LayerNorm, dropout 0.2, AdamW learning rate 0.001, weight decay zero, and
300 full-batch epochs. Loss is the mean of member cross-entropies. Select
one joint checkpoint each epoch by highest pooled-logit validation accuracy,
then lowest pooled validation cross-entropy, then earliest epoch. Score the
test mask only after restoring the selected checkpoint. Untied propagation
copies the initially tied residual stack for each member while retaining
shared boundary maps and output normalization. Canonical initialization and
post-construction Python/NumPy/CPU Torch/CUDA Torch RNG must match; initial
member logits must be within 1e-5 maximum absolute difference (the earlier
A100 floating-point tolerance) before training.

Pin source, protocol, model, raw NPZ bytes, parsed tensor fingerprints and
software versions before training. Pass CPU data preflight and GPU
initialization smoke for both depths. Each run must save a 300-row validation
trace, initial-state audit, selected checkpoint, selected validation/test
member logits, official indices/labels, parameter count and artifact hashes.
A separate CUDA verifier must replay every selected checkpoint. An
independent Mac audit must check official labels/masks and all 12 archived
scores. Report every paired tied-minus-untied seed difference for validation
and test and both depth means. Existing Roman bridge and Roman depth-grid
result trees are untouched. The 300-epoch ceiling must be described wherever
selected epochs are near it; this protocol does not establish convergence.
