# Compact public evidence for factor placement and same-runtime TIED

This is a **post-outcome transport derivative** of two frozen studies. It does
not select runs or change any training or test result. Build only after the
all-layer 72-cell and same-runtime TIED36 36-cell validation locks and their
complete independent selected/default score audits exist. The original
432-cell validation lock is the common source anchor.

The public stage contains all 108 original `result.json` and full 1000-epoch
`validation_trace.csv` files byte-for-byte inside one standard-library
`CELL_RECORDS.tar.xz`. It also contains the complete original freezes,
validation locks, final score audits, source and protocols, and all allowed
selected/default `score.json` files byte-for-byte. Source SHA-256 digests in
the freezes and result/audit hashes must match at build time and are checked
again by the public verifier.

For **every one of the 108 trained cells**, the builder projects the saved
selected-checkpoint validation predictions to exact uint8 pooled and
four-member class decisions. Validation node IDs and labels are stored once
per graph. For **every allowed selected/default test score** in either
study, it stores the original float32 pooled test logits and exact uint8
four-member class decisions. Test IDs and labels come from the sibling
primary compact study's verified graph reference. All original float
prediction/checkpoint SHA-256 values are retained in the manifest. The
author evidence retains the original checkpoints, full member float logits,
and original prediction archives.

The public verifier recomputes checkpoint-selection epochs from every full
validation trace, validation-selected candidate choices from the locks,
selected validation accuracies from the exact hard decisions, and every
allowed test pooled accuracy and cross entropy from exact float32 pooled
logits and frozen test labels. It checks the test member decisions and their
mean accuracy where recorded. It checks complete key sets, source hashes,
and all audit/lock relationships. It cannot replay model weights or
training, reproduce validation cross entropy from hard decisions, or
recompute individual member probability scores without the omitted full
float logits/checkpoints. Those remain available to authors for checkpoint
replay and can be regenerated from the frozen source and public datasets.

The all-layer factor placement was chosen after earlier outcomes and carries
additional parameters. The separate TIED36 same-runtime control addresses
runtime comparability on Cora and WikiCS; neither comparison is claimed to
isolate placement from parameter capacity on every graph.
