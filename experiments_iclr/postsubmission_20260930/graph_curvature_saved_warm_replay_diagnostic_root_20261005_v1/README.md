# Native initialization replay diagnosis

These checks concern implementation repeatability, not predictive performance.
No new model fit, model forward, heldout prediction, or host-setting change was
performed. All remote operations used the authorized one-GPU repository.

`RESULT.json` compares the two already saved Squirrel17 engineering warm
checkpoints. Their model tensors and Adam moments differ; their recorded RNG,
specification and mode fields agree. The comparison reports 132 differing model
tensors, 264 differing optimizer tensors and one metadata field. It does not
establish which operation caused the difference.

`PREPROCESSING_REPLAY_RESULT.json` records two executions of the unchanged native
preprocessing on the same loaded features and edges under the recorded
deterministic CUDA policy. The preprocessed features differ in 6,232,027 FP32
elements, with maximum absolute difference 5.960464477539063e-08. The edges agree
exactly. Native preprocessing therefore is not bitwise repeatable under this
policy. The saved source uses repeated CUDA sparse propagation; attributing the
warm checkpoint differences to this operation requires further evidence.

The preprocessing probe reused the bound engineering loader, which reads both
TRAIN and VALIDATION compact label files. Their values were unused and not
reported. TEST labels were not read. The result's `heldout_labels_read=false`
field refers only to TEST and is too broad a field name; this note records the
precise access scope and preserves the executed source and original receipt.

The earlier returned-head reconstruction discrepancy is a separate issue:
recomputation differed by at most 1.1920928955078125e-07 in the named fixed arm.
The installation-witness successor is intended to test actual state custody
independently, without changing scientific settings or tolerances. No current
diagnostic qualifies the method or supplies a predictive gain.

For reproducible new comparisons, preserve the actual preprocessed inputs
alongside each fresh warm state and charge their capture/storage. Every paired
continuation must use that same recorded input. This is an implementation
requirement, not a retrospectively selected favorable experiment.
